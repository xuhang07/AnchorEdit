import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple

# 假设相关的封装类已经导入
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

class MultiTurnI2VPipeline(nn.Module):
    def __init__(
            self,
            args,
            generator=None,
            text_encoder=None,
            vae=None,
            device="cuda"
    ):
        super().__init__()
        self.args = args
        self.device = device
        
        # 1. 初始化模型组件
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {})) if generator is None else generator
        self.text_encoder = WanTextEncoder(self.args.model_kwargs.model_name) if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper(self.args.model_kwargs.model_name) if vae is None else vae

        # 2. 超参数设置
        self.num_transformer_blocks = getattr(args.model_kwargs, "num_transformer_blocks", 40)
        self.frame_seq_length = getattr(args, "frame_seq_length", 1560)
        self.local_attn_size = self.generator.model.local_attn_size
        self.num_frame_per_block = getattr(args.model_kwargs, "num_frame_per_block", 1)
        
        # 采样参数
        self.scheduler = self.generator.get_scheduler()
        
        # 对应参考代码中的 denoising_step_list 逻辑
        # 如果 args 提供了 list, 映射到 scheduler 的 timesteps 上
        self.denoising_step_list = torch.tensor(getattr(args, "denoising_step_list", [1000, 750, 500, 250]), dtype=torch.int64, device=device)
        if getattr(args, "warp_denoising_step", False):
            timesteps = torch.cat((self.scheduler.timesteps.to(device), torch.tensor([0], dtype=torch.float32, device=device)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list].long()
        
        # 缓存容器
        self.kv_cache = None
        self.crossattn_cache = None

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """初始化图像 KV Cache"""
        kv_cache = []
        kv_cache_size = self.local_attn_size * self.frame_seq_length if self.local_attn_size != -1 else 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, kv_cache_size, self.generator.model.num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, self.generator.model.num_heads, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.int64, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.int64, device=device)
            })
        self.kv_cache = kv_cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """初始化 Cross-Attention (Text) Cache"""
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def encode_multi_prompts(self, prompts_list: List[str]) -> torch.Tensor:
        """将 Prompt 编码为 Embeds"""
        text_dict = self.text_encoder(prompts_list)
        return text_dict['prompt_embeds'] if isinstance(text_dict, dict) else text_dict

    @torch.no_grad()
    def multi_turn_inference(
        self,
        initial_image_latent: torch.Tensor,  # [B, 1, C, H, W]
        prompts: List[str],                  # 每一轮的新 Prompt (长度为 num_turns)
        negative_prompt: str = "",           
        seed: int = 42,
        guidance_scale: float = 7.5,         
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        执行多轮自回归生成，每轮生成 num_frame_per_block 帧。
        """
        device = self.device
        generator_dtype = next(self.generator.parameters()).dtype
        vae_weight_dtype = next(self.vae.model.parameters()).dtype
        
        batch_size = initial_image_latent.shape[0]
        num_turns = len(prompts)
        _, _, num_channels, height, width = initial_image_latent.shape
        
        # 1. 初始化 Cache
        actual_bs = batch_size * 2 if guidance_scale > 1.0 else batch_size
        self._initialize_kv_cache(actual_bs, generator_dtype, device)
        self._initialize_crossattn_cache(actual_bs, generator_dtype, device)
        
        # 2. 准备负向 Embeds
        if guidance_scale > 1.0:
            uncond_embeds = self.encode_multi_prompts([negative_prompt] * batch_size).to(generator_dtype)

        # 3. 注入初始参考帧 (Turn 0 Context)
        # 初始帧通常只占 1 个 Token Offset 位置
        current_offset_tokens = 0
        init_prompt_embeds = self.encode_multi_prompts([prompts[0]] * batch_size).to(generator_dtype)
        
        if guidance_scale > 1.0:
            combined_context = {'prompt_embeds': torch.cat([init_prompt_embeds, uncond_embeds], dim=0).to(generator_dtype)}
            combined_latent = torch.cat([initial_image_latent] * 2, dim=0)
        else:
            combined_context = {'prompt_embeds': init_prompt_embeds}
            combined_latent = initial_image_latent

        self.generator(
            noisy_image_or_video=combined_latent.to(generator_dtype),
            conditional_dict=combined_context,
            timestep=torch.zeros([actual_bs, 1], device=device, dtype=torch.int64),
            kv_cache=self.kv_cache,
            crossattn_cache=None,
            current_start=current_offset_tokens
        )
        current_offset_tokens += self.frame_seq_length # 第一帧占位

        output_latents = [initial_image_latent]
        decoded_images = []
        
        # 4. 开始多轮 Block 生成
        for i in range(num_turns):
            print(f"--- Generating Turn {i+1}/{num_turns} (Frames: {self.num_frame_per_block}) ---")
            
            # 4.1 准备当前轮 Embeds
            turn_prompt_embeds = self.encode_multi_prompts([prompts[i]] * batch_size).to(generator_dtype)
            if guidance_scale > 1.0:
                current_context = {'prompt_embeds': torch.cat([turn_prompt_embeds, uncond_embeds], dim=0)}
            else:
                current_context = {'prompt_embeds': turn_prompt_embeds}
            
            # 4.2 初始化这一轮 Block 的噪声 [B, num_frame_per_block, C, H, W]
            torch.manual_seed(seed + i)
            noisy_input = torch.randn(
                [batch_size, self.num_frame_per_block, num_channels, height, width], 
                device=device, dtype=generator_dtype
            )

            # 4.3 空间去噪循环 (Temporal loop for the current block)
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                if guidance_scale > 1.0:
                    model_input = torch.cat([noisy_input] * 2, dim=0)
                    ts_tensor = torch.ones([actual_bs, self.num_frame_per_block], device=device, dtype=torch.int64) * current_timestep
                else:
                    model_input = noisy_input
                    ts_tensor = torch.ones([actual_bs, self.num_frame_per_block], device=device, dtype=torch.int64) * current_timestep

                _, model_output = self.generator(
                    noisy_image_or_video=model_input,
                    conditional_dict=current_context,
                    timestep=ts_tensor,
                    kv_cache=self.kv_cache,
                    crossattn_cache=None,
                    current_start=current_offset_tokens
                )

                if guidance_scale > 1.0:
                    out_cond, out_uncond = model_output.chunk(2, dim=0)
                    denoised_pred = out_uncond + guidance_scale * (out_cond - out_uncond)
                else:
                    denoised_pred = model_output

                # Scheduler 步进
                if step_idx < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[step_idx + 1]
                    # Flatten batch and frames for scheduler
                    flat_denoised = denoised_pred.flatten(0, 1)
                    noisy_input = self.scheduler.add_noise(
                        flat_denoised,
                        torch.randn_like(flat_denoised),
                        next_timestep * torch.ones([batch_size * self.num_frame_per_block], device=device, dtype=torch.int64)
                    ).unflatten(0, (batch_size, self.num_frame_per_block))
                else:
                    final_block_latent = denoised_pred

            # 4.4 记录本轮 Block 结果
            output_latents.append(final_block_latent)

            # 4.5 关键：更新 KV Cache (使用干净帧和 context_noise)
            # 根据参考代码，这里使用 args.context_noise (通常是 0 或极小值)
            context_noise_val = getattr(self.args, "context_noise", 0)
            if guidance_scale > 1.0:
                update_input = torch.cat([final_block_latent] * 2, dim=0)
                update_ts = torch.ones([actual_bs, self.num_frame_per_block], device=device, dtype=torch.int64) * context_noise_val
            else:
                update_input = final_block_latent
                update_ts = torch.ones([actual_bs, self.num_frame_per_block], device=device, dtype=torch.int64) * context_noise_val

            self.generator(
                noisy_image_or_video=update_input,
                conditional_dict=current_context,
                timestep=update_ts,
                kv_cache=self.kv_cache,
                crossattn_cache=None,
                current_start=current_offset_tokens
            )

            # 4.6 更新偏移量：每轮增加 num_frame_per_block 个 Block 的 Token 长度
            current_offset_tokens += self.num_frame_per_block * self.frame_seq_length
            
            # 4.7 实时解码预览 (按单帧展平解码)
            # final_block_latent 形状: [B, num_frame_per_block, C, H, W]
            with torch.no_grad():
                # 1. 将 [B, T, C, H, W] 展平为 [B*T, 1, C, H, W]
                # 这里的 1 是为了保持 VAE 预期的维度结构 (Batch, Time, Channel, H, W)
                b, t, c, h, w = final_block_latent.shape
                flat_latent = final_block_latent.view(b * t, 1, c, h, w)
                
                # 2. 调用 VAE 解码
                # 此时 VAE 会把每一帧当成一个独立的 Batch 成员处理
                # 返回形状通常为 [B*T, 3, 1, H, W] 或 [B*T, 1, 3, H, W]
                decoded_flat = self.vae.decode_to_pixel(flat_latent.to(vae_weight_dtype))
                
                # 3. 归一化并恢复形状 [B, T, 3, H, W]
                decoded_flat = (decoded_flat * 0.5 + 0.5).clamp(0, 1)
                
                # 兼容性检查：确保 Channel 在 dim=2 (即 [B*T, 1, 3, H, W])
                if decoded_flat.shape[1] == 3: # 如果是 [B*T, 3, 1, H, W]
                    decoded_flat = decoded_flat.permute(0, 2, 1, 3, 4)
                
                # 重新 view 回 [B, T, 3, H, W]
                decoded_block = decoded_flat.view(b, t, 3, h*8, w*8)

                # 4. 逐帧保存 (只取 Batch 0)
                for frame_idx in range(t):
                    single_frame = decoded_block[0, frame_idx] # [3, H, W]
                    decoded_images.append(single_frame.cpu()) # 建议转到 CPU 节省显存

        # 5. 最终输出
        full_latents = torch.cat(output_latents, dim=1) 
        full_video = self.vae.decode_to_pixel(full_latents.to(vae_weight_dtype))
        full_video = (full_video * 0.5 + 0.5).clamp(0, 1)
        
        return full_video, decoded_images