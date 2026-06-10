import gc
import json
import logging
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
import wandb
import time
import os
import random
from typing import Dict, List, Tuple, Optional

from utils.misc import cycle
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed, merge_dict_list
from utils.multi_turn_dataset import create_multi_turn_dataloader
from utils.multi_turn_tokenizer import MultiTurnTokenizer, MultiTurnCollator

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# from wan.modules.clip import clip_xlm_roberta_vit_h_14, CLIPModel


class MultiTurnI2VTrainer:
    """
    Trainer for multi-turn image-to-video editing using AR model.
    Implements teacher forcing training with random timestep selection.
    """
    
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Initialize distributed training environment
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.training.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.enable_tensorboard = config.logging.enable_tensorboard

        # Set random seed
        if config.system.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.system.seed = random_seed.item()

        set_seed(config.system.seed + global_rank)

        # Initialize wandb
        if self.is_main_process and self.enable_tensorboard:
            os.makedirs(config.logging.logdir, exist_ok=True)
            os.makedirs(config.logging.tensorboard_logdir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=config.logging.tensorboard_logdir)

            config_dict = OmegaConf.to_container(config, resolve=True) # resolve=True 会解析配置中的变量引用
            self.writer.add_text("config", json.dumps(config_dict, indent=2, default=str), 0)

        self.output_path = config.logging.ckpt_logdir

        # Initialize model components
        self._setup_model()
        self._setup_optimizer()
        self._setup_dataloader()
        self._setup_training_params()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "generator" in checkpoint:
                state_dict = checkpoint["generator"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        prefixes = ["_orig_mod.", "_checkpoint_wrapped_module.", "module."]
        normalized_state_dict = {}
        for k, v in state_dict.items():
            nk = k
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                        changed = True
            normalized_state_dict[nk] = v

        missing, unexpected = self.generator.load_state_dict(normalized_state_dict, strict=False)
        if self.is_main_process:
            print(f"[Checkpoint] loaded from {checkpoint_path}")
            print(f"[Checkpoint] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    def _setup_model(self):
        """Initialize model components"""
        # Use CausVid as base model for AR training
        self.generator = WanDiffusionWrapper(**self.config.model_kwargs)
        self.generator.model.requires_grad_(True)


        self.text_encoder = WanTextEncoder(self.config.model_kwargs.model_name)
        self.text_encoder.requires_grad_(False)

        # self.clip = CLIPModel(**self.config.clip_model_kwargs)

        self.vae = WanVAEWrapper(self.config.model_kwargs.model_name)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
        
        # Load pretrained checkpoint if specified
        ckpt_path = getattr(self.config.checkpoint, "pretrained_pt", None) or getattr(self.config.checkpoint, "generator_ckpt", None)
        if ckpt_path:
            self.load_checkpoint(ckpt_path)
        
        # Wrap models with FSDP
        self.generator = fsdp_wrap(
            self.generator,
            sharding_strategy=self.config.fsdp.sharding_strategy,
            mixed_precision=self.config.training.mixed_precision,
            wrap_strategy=self.config.fsdp.generator_fsdp_wrap_strategy
        )

        self.text_encoder = fsdp_wrap(
            self.text_encoder,
            sharding_strategy=self.config.fsdp.sharding_strategy,
            mixed_precision=self.config.training.mixed_precision,
            wrap_strategy=self.config.fsdp.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(self.config.fsdp, "text_encoder_cpu_offload", False)
        )

        # Setup VAE
        if not getattr(self.config.checkpoint, "no_visualize", False):
            self.vae = self.vae.to(
                device=self.device,
                dtype=torch.bfloat16 if self.config.training.mixed_precision else torch.float32
            )

        # Initialize tokenizer for multi-turn processing
        self.tokenizer = MultiTurnTokenizer(
            text_encoder=self.text_encoder,
            vae=self.vae,
            dtype=self.dtype,
            device=self.device,
            config=self.config,
        )

    def _setup_optimizer(self):
        """Initialize optimizers"""
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.generator.parameters() if param.requires_grad],
            lr=self.config.training.lr,
            betas=(self.config.training.beta1, self.config.training.beta2),
            weight_decay=self.config.training.weight_decay
        )

    def _setup_dataloader(self):
        """Initialize dataloader for multi-turn dataset"""
        dataloader = create_multi_turn_dataloader(
            data_path=self.config.data.data_path,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            num_frame_per_block=self.config.model_kwargs.num_frame_per_block,
            max_turns=getattr(self.config.data, "max_turns", None),
            shuffle=True,
            image_size=tuple(self.config.data.image_size),
            bucket_step_width=getattr(self.config.data, "bucket_step_width", 16),
            bucket_step_height=getattr(self.config.data, "bucket_step_height", 16),
            bucket_max_ratio=getattr(self.config.data, "bucket_max_ratio", 4.0),
        )
        
        self.dataloader = cycle(dataloader)

    def _setup_training_params(self):
        """Setup training hyperparameters"""
        self.num_train_timesteps = self.config.training.num_train_timesteps
        self.min_step = int(self.config.training.min_step_ratio * self.num_train_timesteps)
        self.max_step = int(self.config.training.max_step_ratio * self.num_train_timesteps)
        
        
        # Setup EMA
        ema_weight = getattr(self.config.training, "ema_weight", 0.0)
        self.generator_ema = None
        if ema_weight > 0.0:
            if self.is_main_process:
                print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.generator, decay=ema_weight)

        # Gradient clipping
        self.max_grad_norm_generator = getattr(self.config.training, "max_grad_norm_generator", 10.0)
        self.same_timestep_across_sequence = getattr(
            self.config.training, "same_timestep_across_sequence", False
        )

        # Gradient accumulation
        self.gradient_accumulation_steps = max(
            1, int(getattr(self.config.training, "gradient_accumulation_steps", 1))
        )
        # 记录当前处于累计周期内的第几个 micro-step（0-based）
        self._accum_step_idx = 0
        if self.is_main_process:
            print(
                f"[GradAccum] gradient_accumulation_steps={self.gradient_accumulation_steps} "
                f"(effective batch = batch_size * grad_accum * world_size = "
                f"{self.config.training.batch_size} * {self.gradient_accumulation_steps} * {self.world_size})"
            )

        # ---- 训练策略开关（均可在 config.training 下设置；默认等价于原 diffusion forcing） ----
        # Exposure-bias / scheduled sampling：以概率 p_self 用模型自己生成的 x0 替换中间历史帧
        self.self_forcing_prob = float(getattr(self.config.training, "self_forcing_prob", 0.0))
        self.self_forcing_warmup_steps = int(
            getattr(self.config.training, "self_forcing_warmup_steps", 0)
        )
        # Turn-weighted loss：每帧权重 1 + alpha * p_i，alpha=0 即原始 loss
        self.turn_weight_alpha = float(getattr(self.config.training, "turn_weight_alpha", 0.0))

        self.previous_time = None

    def _prepare_multi_turn_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Teacher Forcing 模式：准备全序列数据。
        """
        # 1. 编码全量图片 [B, C, T, H, W] 和全量文本 [B, (T-1)*L, D]
        processed_batch = self.tokenizer(batch) # 内部不再需要 step 截断
        
        image_latents = processed_batch['image_latents']
        prompt_embeds = processed_batch['prompt_embeds']
        
        # 2. 构造条件字典
        conditional_dict = {
            'prompt_embeds': prompt_embeds,
        }
        
        return {
            'conditional_dict': conditional_dict,
            'image_latents': image_latents,  # 这里的 T 是完整的序列长度
        }


    # ------------------------------------------------------------------
    # VAE decode→encode 往返（无梯度）：用于在 latent 域复现"真实推理"经过
    # 一次 VAE 编解码后引入的非高斯累积误差。仅对 sel_idx 给出的 (b, t) 帧执行，
    # 一次性 batch 化送入 VAE，保持开销可控。
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _vae_roundtrip_latents(
        self, latents: torch.Tensor, sel_idx: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Args:
            latents: [B, T, C, H, W]，当前已部分退化的 latent。
            sel_idx: [N_sel, 2]，每行 (b, t)，需要往返的位置索引。
        Returns:
            [N_sel, C, H, W]，对应位置往返后的 latent；
            若 VAE 不可用或 sel_idx 为空返回 None。
        """
        if not hasattr(self, "vae") or self.vae is None or sel_idx.numel() == 0:
            return None

        device = latents.device
        target_dtype = latents.dtype

        # 选中的帧打包成单帧 video：[N, T'=1, C, H, W]
        bs_i, ts_i = sel_idx[:, 0], sel_idx[:, 1]
        sel = latents[bs_i, ts_i].unsqueeze(1)  # [N, 1, C, H, W]

        try:
            # WanVAEWrapper 接口约定：
            #   - decode_to_pixel(latent[B,T,C,H,W])  → pixel[B,T,C_pix,H,W]（float32）
            #   - encode_to_latent(pixel[B,C,T,H,W])  → latent[B,T,C,H,W]
            # 注意 encode 输入是 [B,C,T,H,W]（C-T 与 decode 输出顺序相反），需要 permute。
            sel_in = sel.to(dtype=self.dtype)                                # [N, 1, C_lat, H, W]
            pixel = self.vae.decode_to_pixel(sel_in, use_cache=False)        # [N, 1, C_pix=3, H_pix, W_pix] float32
            pixel = pixel.to(dtype=self.dtype)                               # -> bf16
            pixel_for_enc = pixel.permute(0, 2, 1, 3, 4).contiguous()        # [N, C_pix, 1, H_pix, W_pix]
            latent_rt = self.vae.encode_to_latent(pixel_for_enc)             # [N, 1, C_lat, H, W]
            latent_rt = latent_rt.squeeze(1)                                 # [N, C_lat, H, W]
            return latent_rt.to(dtype=target_dtype, device=device)
        except Exception as e:
            # VAE 出问题不应影响训练，跳过往返退化
            if self.is_main_process:
                print(f"[Warn] VAE roundtrip skipped due to error: {e}")
            return None

    # ------------------------------------------------------------------
    # Self-forcing / Scheduled sampling：以概率 self_forcing_prob 用模型自己
    # 一次 no_grad 前向得到的 x0 替换中间历史帧的 latent，模拟推理时的 exposure bias。
    #   - 首帧（GT，i2v reference）和末帧（当前要预测的目标）不替换。
    #   - 概率为 0 时退化为 diffusion forcing（保持原 GT 历史）。
    #   - target 始终用原始 GT，让模型学着"哪怕历史走偏，也要把目标拉回真值"。
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _self_forcing_rollout(
        self,
        image_latents: torch.Tensor,
        conditional_dict: Dict,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        以一次 no_grad 前向得到 pred_x0，并对中间历史帧按概率替换 image_latents。

        **训练语义**：
          - 被替换的帧：clean latent 不再是 GT，而是 detach(pred_x0)。
            该帧仍然会经历"退化 + 加噪 + forward"，作为后续帧的上下文存在；
            但**其自身 loss 不应被计算**（target 用的是 GT，与已经替换掉的 clean 不一致）。
          - 未被替换的帧：保持 GT，正常算 loss。

        **FSDP 安全**：无论是否真正需要替换，所有 rank 都必须执行同一次 generator
        forward，因为 FSDP forward 内部有 all-gather 集合通信，跨 rank 必须同步。

        Args:
            image_latents: [B, T, C, H, W]，原始 GT latent。
            conditional_dict: 文本条件。
            timesteps: [B, T]，当前 step 采样的 timestep。
            noise: [B, T, C, H, W]，与正式前向使用相同的 noise。
        Returns:
            new_latents: [B, T, C, H, W]，被部分替换后的 latent。
            replace_mask: [B, T] bool，True 表示该帧被替换（loss 应忽略）。
        """
        B, T, C, H, W = image_latents.shape
        device = image_latents.device

        # 判断是否真正需要替换（但不能提前 return，必须走完 forward 保持 FSDP 同步）
        need_replace = (
            self.self_forcing_prob > 0.0
            and self.step >= self.self_forcing_warmup_steps
            and T > 2
        )

        # 用与正式前向一致的加噪输入
        noisy_input = self.scheduler.add_noise(
            original_samples=image_latents.flatten(0, 1),
            noise=noise.flatten(0, 1),
            timestep=timesteps.flatten(),
        ).view(B, T, C, H, W)

        # 必须执行 forward 以保持 FSDP 全 rank 同步（即使结果会被丢弃）
        _, pred_x0 = self.generator(
            noisy_input,
            timestep=timesteps,
            conditional_dict=conditional_dict,
        )

        # 默认无替换
        replace_mask = torch.zeros((B, T), device=device, dtype=torch.bool)

        if not need_replace:
            return image_latents, replace_mask

        pred_x0 = pred_x0.detach().to(dtype=image_latents.dtype)

        # 构造替换 mask：仅对中间帧 [1, T-1) 以 p_self 概率替换
        # 首帧（i2v reference）和末帧（当前要预测目标）保留 GT
        if T > 2:
            mid = torch.rand((B, T - 2), device=device) < self.self_forcing_prob
            replace_mask[:, 1:T - 1] = mid

        replace_mask_5d = replace_mask.view(B, T, 1, 1, 1).to(image_latents.dtype)
        new_latents = image_latents * (1 - replace_mask_5d) + pred_x0 * replace_mask_5d
        return new_latents, replace_mask

    # ------------------------------------------------------------------
    # 帧位退化（latent 域）：沿时间维度单调增强，
    # 模拟真实多轮推理"越晚越脏"的累积视觉劣化。
    # 设计目标：
    #   1) 退化强度按帧位置 p_i = i / (T-1) 单调插值，保证"越早越干净"；
    #   2) 避免过于高斯化——采用低频色偏 / 块压缩 / Laplace 乘性 / 弱模糊 /
    #      通道颜色漂移 的随机组合，更接近真实 codec / VAE 的非高斯误差。
    # ------------------------------------------------------------------
    def _degrade_history_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, T, C, H, W]，clean latent。
        Returns:
            退化后的 latent，shape 同输入。
            每帧强度 = base_strength * p_i，其中 p_i = i / (T-1)。
            当 T == 1 时不做退化。
        """
        cfg = getattr(self.config.training, "history_degradation", None)
        if cfg is not None and getattr(cfg, "enable", True) is False:
            return latents

        B, T, C, H, W = latents.shape
        if T <= 1:
            return latents

        device = latents.device
        dtype = latents.dtype

        def _get(key, default):
            if cfg is None:
                return default
            return getattr(cfg, key, default)

        # 每种退化的触发概率（帧位置系数会进一步调制）
        p_color    = _get("p_color", 0.5)     # 通道颜色漂移（乘加）
        p_lowfreq  = _get("p_lowfreq", 0.5)   # 低频色偏/晕染（模拟 VAE 编解码误差）
        p_block    = _get("p_block", 0.35)    # 块均值化（模拟 JPEG / VAE 压缩）
        p_laplace  = _get("p_laplace", 0.5)   # Laplace 乘性扰动（重尾）
        p_blur     = _get("p_blur", 0.3)      # 轻度空间模糊
        p_vae_rt   = _get("p_vae_rt", 0.0)    # VAE decode→encode 往返（真实累积误差，无梯度）

        # 强度上限（都保持轻度）。实际 per-frame 强度 = 上限 * p_i
        color_scale_eps = _get("color_scale_eps", 0.06)  # 通道乘性 ±6%
        color_bias_eps  = _get("color_bias_eps", 0.04)   # 通道加性 ±4%
        lowfreq_amp_max = _get("lowfreq_amp_max", 0.08)  # 低频场最大振幅（latent std ~ 1）
        block_sizes     = _get("block_sizes", [2, 4])    # 块均值化候选块大小
        laplace_b_max   = _get("laplace_b_max", 0.05)    # Laplace 乘性扰动的 b（scale）
        blur_sigma_max  = _get("blur_sigma_max", 0.6)    # 空间模糊最大 sigma
        vae_rt_mix_max  = _get("vae_rt_mix_max", 0.6)    # VAE 往返输出与原 latent 的最大混合比

        # 每帧强度系数 s_i ~ Uniform(lower_frac, upper_i)：
        #   - monotonic=True（默认）：upper_i = lower_frac + (1 - lower_frac) * p_i，
        #     p_i ∈ [0,1] 随位置递增；首帧上限 = lower_frac（区间退化成单点），末帧上限 = 1.0。
        #     语义：后帧"期望更脏"，但仍有概率采到接近 lower_frac 的低退化。
        #   - monotonic=False：所有帧共享同一区间 [lower_frac, 1.0]，无位置偏置。
        #     适用于双向 self-attn（不存在因果"越晚越脏"假设）。
        # 每个 (b, t) 独立采样，保持多样性。
        monotonic_deg = bool(_get("monotonic_degradation", True))
        if monotonic_deg:
            raw_pos = torch.linspace(0.0, 1.0, T, device=device, dtype=torch.float32)
        else:
            # 关闭位置偏置：所有帧 p_i = 1.0
            raw_pos = torch.ones((T,), device=device, dtype=torch.float32)
        lower_frac = float(_get("degrade_lower_frac", 0.0))
        lower_frac = max(0.0, min(lower_frac, 1.0))
        upper_per_t = lower_frac + (1.0 - lower_frac) * raw_pos  # [T]
        # [B, T, 1, 1, 1]
        upper_bt = upper_per_t.view(1, T, 1, 1, 1).expand(B, T, 1, 1, 1).to(dtype)
        rand_bt = torch.rand((B, T, 1, 1, 1), device=device, dtype=dtype)
        pos_ratio = lower_frac + (upper_bt - lower_frac) * rand_bt  # [B, T, 1, 1, 1]

        out = latents.clone()

        # ---------- 0) VAE decode→encode 往返（无梯度） ----------
        # 对被选中的 (b, t) 帧做 VAE decode→encode，按采样到的 pos_ratio 混合回原 latent。
        # 注意：VAE 较重，默认 p_vae_rt 较低。首帧通常保留，避免 ref 图过度劣化。
        if p_vae_rt > 0 and vae_rt_mix_max > 0:
            sel_mask = (torch.rand((B, T), device=device) < p_vae_rt)
            # 首帧不参与 VAE 往返（保护 ref 图，且节省计算）
            sel_mask[:, 0] = False
            sel_idx = sel_mask.nonzero(as_tuple=False)  # [N_sel, 2]  (b, t)
            if sel_idx.shape[0] > 0:
                rt_out = self._vae_roundtrip_latents(out, sel_idx)  # [N_sel, C, H, W]
                if rt_out is not None:
                    bs_i, ts_i = sel_idx[:, 0], sel_idx[:, 1]
                    # 每个选中帧各自的混合比 = vae_rt_mix_max * 该 (b, t) 的 pos_ratio
                    mix_per = (vae_rt_mix_max * pos_ratio[bs_i, ts_i, 0, 0, 0]).view(-1, 1, 1, 1)
                    original_sel = out[bs_i, ts_i]  # [N_sel, C, H, W]
                    mixed = original_sel * (1 - mix_per) + rt_out.to(dtype) * mix_per
                    out[bs_i, ts_i] = mixed

        # ---------- 1) 通道颜色漂移（乘 + 加） ----------
        # 每帧/每通道独立采样一个小偏移；强度由该帧采样到的 pos_ratio 决定
        mask_color = (torch.rand((B, T, 1, 1, 1), device=device) < p_color).to(dtype)
        scale = 1.0 + (torch.rand((B, T, C, 1, 1), device=device) * 2 - 1) * color_scale_eps * pos_ratio
        bias  = (torch.rand((B, T, C, 1, 1), device=device) * 2 - 1) * color_bias_eps * pos_ratio
        out = out * (1 - mask_color) + (out * scale + bias) * mask_color

        # ---------- 2) 低频色偏场（非高斯、结构化） ----------
        # 采样 low-res 噪声再 bilinear 上采样，得到平滑的 spatial 偏置图
        if p_lowfreq > 0 and lowfreq_amp_max > 0:
            mask_lf = (torch.rand((B, T, 1, 1, 1), device=device) < p_lowfreq).to(dtype)
            low_h, low_w = max(2, H // 16), max(2, W // 16)
            # 低频源：每帧/每通道独立，值域 [-1, 1]
            low = (torch.rand((B * T, C, low_h, low_w), device=device) * 2 - 1)
            low = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
            low = low.view(B, T, C, H, W).to(dtype)
            amp = torch.rand((B, T, 1, 1, 1), device=device, dtype=dtype) * lowfreq_amp_max * pos_ratio
            out = out + mask_lf * amp * low

        # ---------- 3) 块状均值化（模拟 JPEG / VAE 压缩的纹理粘连） ----------
        if p_block > 0 and len(block_sizes) > 0:
            mask_block = (torch.rand((B, T, 1, 1, 1), device=device) < p_block).to(dtype)
            # 每个 batch 选一个块大小（开销小）
            bsz = int(block_sizes[torch.randint(0, len(block_sizes), (1,)).item()])
            if H % bsz == 0 and W % bsz == 0:
                flat = out.reshape(B * T, C, H, W)
                pooled = F.avg_pool2d(flat, kernel_size=bsz, stride=bsz)
                upped = F.interpolate(pooled, size=(H, W), mode="nearest")
                upped = upped.view(B, T, C, H, W)
                # 混合系数也按位置缩放
                mix = 0.5 * pos_ratio  # 最多 50% 替换
                out = out * (1 - mask_block * mix) + upped * (mask_block * mix)

        # ---------- 4) Laplace 乘性扰动（重尾，替代高斯噪声） ----------
        if p_laplace > 0 and laplace_b_max > 0:
            mask_lap = (torch.rand((B, T, 1, 1, 1), device=device) < p_laplace).to(dtype)
            # Laplace(0, b): 由 Uniform 生成 u ∈ (-0.5, 0.5)，然后
            # x = -b * sign(u) * log(1 - 2|u|)
            u = torch.rand((B, T, C, H, W), device=device) - 0.5
            laplace = -torch.sign(u) * torch.log1p(-2.0 * torch.abs(u) + 1e-7)
            laplace = laplace.to(dtype)
            b_per_frame = laplace_b_max * pos_ratio  # [1, T, 1, 1, 1]
            out = out * (1 + mask_lap * b_per_frame * laplace)

        # ---------- 5) 空间模糊（弱） ----------
        if p_blur > 0 and blur_sigma_max > 0:
            mask_blur = (torch.rand((B, T, 1, 1, 1), device=device) < p_blur).to(dtype)
            sigma = float(torch.empty(1).uniform_(0.1, blur_sigma_max).item())
            ax = torch.arange(-1, 2, device=device, dtype=torch.float32)
            xx, yy = torch.meshgrid(ax, ax, indexing="ij")
            kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
            kernel = kernel / kernel.sum()
            kernel = kernel.to(dtype).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
            flat = out.reshape(B * T, C, H, W)
            blurred = F.conv2d(flat, kernel, padding=1, groups=C).view(B, T, C, H, W)
            # 按该帧采样到的 pos_ratio 加权混合
            mix = mask_blur * pos_ratio
            out = out * (1 - mix) + blurred * mix

        return out

    def compute_loss(self, processed_batch: Dict, timesteps: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Loss 计算：
          - 可选 self-forcing：以概率 self_forcing_prob 用模型 no_grad 输出替换中间历史帧；
          - 历史帧 latent 域退化（沿时间单调增强，含可选 VAE 往返）；
          - 加噪、模型前向，flow-match (noise - latent) 目标；
          - turn-weighted 平均 loss（alpha=0 即原始 loss）。
        """
        image_latents = processed_batch['image_latents']
        conditional_dict = processed_batch['conditional_dict']
        device = image_latents.device

        # 1. 获取基础参数
        batch_size, num_frames, channel, h, w = image_latents.shape

        # 2. 预先采样统一的 noise，方便 self-forcing rollout 与正式前向共用
        noise = torch.randn_like(image_latents)

        # 3. （可选）Self-forcing：用模型自己生成的 x0 替换中间历史帧（detach，无梯度）
        #    sf_latents 中被替换的帧 clean 不再是 GT；replace_mask 标记这些帧。
        #    被替换的帧仍参与"退化 + 加噪 + forward"，作为后续帧的上下文存在；
        #    但其自身 loss 必须被忽略（target 是 GT，与 clean=pred_x0 不一致）。
        sf_latents, replace_mask = self._self_forcing_rollout(
            image_latents=image_latents,
            conditional_dict=conditional_dict,
            timesteps=timesteps,
            noise=noise,
        )

        # 4. 帧位退化（latent 域，沿时间单调增强；含可选 VAE 往返）
        #    被替换的帧也参与退化，避免"被替换的帧不退化"成为模型识别 self-forcing 的偏置。
        degraded_latents = self._degrade_history_latents(sf_latents)

        # 5. 加噪（首帧也加噪，t 由 train_step 中的单调分布给出）
        #    被替换的帧同样加噪，保持与其他帧统一的 timestep 分布。
        noisy_latents_all = self.scheduler.add_noise(
            original_samples=degraded_latents.flatten(0, 1),
            noise=noise.flatten(0, 1),
            timestep=timesteps.flatten()
        ).view(batch_size, num_frames, channel, h, w)
        noisy_model_input = noisy_latents_all

        # 6. 模型前向
        model_output, _ = self.generator(
            noisy_model_input,
            timestep=timesteps,
            conditional_dict=conditional_dict,
        )

        # 7. Flow-match 目标 = noise - GT_latent；首帧（i=0）不计入 loss
        # 注意：target 始终用原始 GT（image_latents），即便历史被替换/退化，
        # 模型也要学着把目标拉回真值。
        target = noise[:, 1:, ...] - image_latents[:, 1:, ...]
        pred = model_output[:, 1:, ...]

        # 8. Turn-weighted MSE：每帧基础权重 w_i = 1 + alpha * p_i
        #    alpha=0 → 全 1，等价于原始 mean MSE。
        if num_frames > 1:
            pos = torch.linspace(0.0, 1.0, num_frames - 1, device=device)
        else:
            pos = torch.zeros((1,), device=device)
        base_weights = (1.0 + self.turn_weight_alpha * pos).view(1, -1).expand(batch_size, -1)  # [B, T-1]

        # 8.1 把被替换的帧权重置 0（不算 loss）
        # replace_mask[:, 1:] 对应 target/pred 的帧索引（已掐掉首帧）
        valid_mask = (~replace_mask[:, 1:]).to(base_weights.dtype)  # [B, T-1]
        eff_weights = base_weights * valid_mask  # [B, T-1]

        mse_per_frame = F.mse_loss(pred.float(), target.float(), reduction="none")  # [B, T-1, C, H, W]
        mse_per_frame = mse_per_frame.mean(dim=(2, 3, 4))  # [B, T-1]

        # 加权平均：(sum_i eff_w_i * mse_i) / sum_i eff_w_i
        # 用 sum_per_sample 防止某个样本所有帧都被替换导致除零（正常情况几乎不会）。
        wsum = eff_weights.sum(dim=1)  # [B]
        weighted_per_sample = (mse_per_frame * eff_weights).sum(dim=1) / wsum.clamp_min(1e-6)  # [B]
        # 仅对"还有有效帧"的样本求平均，避免被全替换样本拉偏
        sample_valid = (wsum > 0).to(weighted_per_sample.dtype)
        loss = (weighted_per_sample * sample_valid).sum() / sample_valid.sum().clamp_min(1.0)

        # 统计实际参与 loss 的帧数（用于日志监控）
        n_replaced = int(replace_mask[:, 1:].sum().item()) if num_frames > 1 else 0
        n_total_frames = batch_size * max(num_frames - 1, 0)

        # 9. 日志
        log_dict = {
            "loss": loss.detach().item(),
            "mean_timestep": timesteps[:, 1:].float().mean().item(),
            "first_frame_t": timesteps[:, 0].float().mean().item(),
            "last_frame_t": timesteps[:, -1].float().mean().item(),
            "sf_replace_ratio": (n_replaced / max(n_total_frames, 1)) if n_total_frames > 0 else 0.0,
        }

        return loss, log_dict

    def train_step(self) -> Dict:
        """
        Execute one training step using Teacher Forcing on the full sequence.
        """
        # 1. 获取 Batch
        batch = next(self.dataloader)
        
        # 2. 准备全序列数据
        processed_batch = self._prepare_multi_turn_batch(batch)
        
        # 3. 参数提取
        batch_size = processed_batch['image_latents'].shape[0]
        num_frames = processed_batch['image_latents'].shape[1]
        num_frame_per_block = self.config.model_kwargs.num_frame_per_block
        num_target_frames = num_frames - 1
        num_blocks = num_target_frames // num_frame_per_block

        # ------------------------------------------------------------------
        # 4. 构造每帧 timestep [B, T]
        # 设计（下限固定 + 上限随轮次提升，区间内独立 uniform 采样）：
        #   - u_lower 固定（每个样本一个，整段序列共享），作为所有帧 t 的下限；
        #     避免首帧 t 恒为极小值，让每帧都"见过一点噪声"。
        #   - 每帧上限 u_upper(p_i) 从 first_frame_noise_max_ratio 单调上升到 u_last_max：
        #         u_upper(p_i) = base_upper + (u_last_max - base_upper) * p_i
        #   - 每帧独立从 [u_lower, u_upper(p_i)] uniform 采样，**不做 sort**，
        #     保留"后帧期望更脏但仍有概率采到很干净"的分布多样性，
        #     更贴近真实推理（后帧也可能因为运气好而干净）。
        # ------------------------------------------------------------------

        # 4.1 末端采样上限 u_last_max ∈ [u_last_max_min_ratio, 1.0]，每个样本一个，经过 flow-shift
        # 抬高 u_last_max 的下界，确保末帧能采到足够大的 t（mean_t 不至于过低）。
        u_last_min_ratio = getattr(
            self.config.training, "u_last_max_min_ratio", 0.0
        )
        u_last_min_ratio = float(min(max(u_last_min_ratio, 0.0), 1.0))
        u_last_max = u_last_min_ratio + (1.0 - u_last_min_ratio) * torch.rand(
            (batch_size,), device=self.device
        )
        ts_shift = self.config.model_kwargs.get("timestep_shift", 1.0)
        if ts_shift != 1.0:
            u_last_max = (ts_shift * u_last_max) / (1 + (ts_shift - 1) * u_last_max)

        # 4.2 下限 u_lower ∈ [first_min_r, first_max_r]（每个样本一个，固定不随帧位置变化）
        first_frame_max_ratio = getattr(
            self.config.training, "first_frame_noise_max_ratio", 0.3
        )
        first_frame_min_ratio = getattr(
            self.config.training, "first_frame_noise_min_ratio", 0.0
        )
        u_lower = first_frame_min_ratio + (
            first_frame_max_ratio - first_frame_min_ratio
        ) * torch.rand((batch_size,), device=self.device)
        # u_lower 不能超过 u_last_max，否则区间非法
        u_lower = torch.minimum(u_lower, u_last_max)

        # 4.3 位置比例 p_i ∈ [0, 1]
        # monotonic_timestep=False 时所有帧共享同一全局上限 u_last_max（无位置偏置），
        # 适用于双向 self-attn（不存在"越晚越脏"的因果假设）。
        monotonic_t = bool(getattr(self.config.training, "monotonic_timestep", True))
        if monotonic_t and num_frames > 1:
            pos = torch.linspace(0.0, 1.0, num_frames, device=self.device)
        else:
            # 关闭位置偏置：所有帧 p_i = 1.0，即 u_upper(p_i) = u_last_max
            pos = torch.ones((num_frames,), device=self.device)
        pos = pos.unsqueeze(0)  # [1, T]

        # 4.4 每帧上限 u_upper(p_i)
        # monotonic 开启：base_upper → u_last_max 线性递增
        # monotonic 关闭：所有帧上限恒为 u_last_max（pos 全 1）
        base_upper = torch.maximum(
            u_lower + 1e-4,
            torch.full_like(u_lower, float(first_frame_max_ratio)),
        )
        base_upper = torch.minimum(base_upper, u_last_max)  # 不超过末端
        u_upper = base_upper.unsqueeze(1) + (u_last_max - base_upper).unsqueeze(1) * pos  # [B, T]

        # 4.5 在 [u_lower, u_upper(p_i)] 内 uniform 采样
        # 注意：每帧独立采样，不做 sort，保留"后帧期望更脏但仍有概率很干净"的分布语义。
        rand_u = torch.rand((batch_size, num_frames), device=self.device)
        u_grid = u_lower.unsqueeze(1) + (u_upper - u_lower.unsqueeze(1)) * rand_u  # [B, T]

        u_grid = u_grid.clamp(0.0, 1.0)

        # 4.7 映射到 [min_step, max_step]
        full_timesteps_f = self.min_step + u_grid * (self.max_step - self.min_step)
        full_timesteps = full_timesteps_f.long().clamp_(self.min_step, self.max_step - 1)

        # --- 梯度累计：仅在累计周期开始时清零梯度 ---
        is_accum_start = (self._accum_step_idx == 0)
        is_update_step = (self._accum_step_idx + 1 >= self.gradient_accumulation_steps)

        if is_accum_start:
            self.generator_optimizer.zero_grad(set_to_none=True)

        # 5. 计算 Loss
        loss, log_dict = self.compute_loss(processed_batch, timesteps=full_timesteps)

        # 为了让累计后的梯度数值等价于大 batch 的平均梯度，loss 除以累计步数
        loss_scaled = loss / self.gradient_accumulation_steps

        # 6. 反向传播（FSDP 下不使用 no_sync，以保证实现简单和正确性）
        loss_scaled.backward()

        log_dict["is_update_step"] = is_update_step
        log_dict["accum_idx"] = self._accum_step_idx

        if is_update_step:
            # 7. 梯度裁剪（累计完成后再裁剪，对应真实梯度范数）
            grad_norm = self.generator.clip_grad_norm_(self.max_grad_norm_generator)
            log_dict['grad_norm'] = grad_norm.detach()

            # 8. 优化器更新
            self.generator_optimizer.step()

            # 9. 更新 EMA
            if self.generator_ema is not None:
                self.generator_ema.update(self.generator)

            self._accum_step_idx = 0
        else:
            # 中间 micro-step 没有真正更新权重，占位一个 grad_norm 以便日志统一
            log_dict['grad_norm'] = torch.tensor(0.0, device=self.device)
            self._accum_step_idx += 1

        torch.cuda.empty_cache()

        return log_dict

    def save_checkpoint(self):
        """Save model checkpoint"""
        print("Start gathering distributed model states...")
        
        generator_state_dict = fsdp_state_dict(self.generator)
        
        if self.generator_ema is not None:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
                "step": self.step,
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "step": self.step,
            }

        if self.is_main_process:
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    def train(self):
        """Main training loop"""
        
        while self.step < self.config.training.max_steps:
            # Training step（一次 micro-step；梯度累计期间 self.step 不增加）
            log_dict = self.train_step()

            # 仅在真正完成一次优化器更新时，才视为前进了一个 global step
            if not log_dict.get("is_update_step", True):
                continue

            # --- 新增：聚合各卡上的指标 ---
            # 创建需要同步的 tensor 列表
            metrics = torch.tensor([
                log_dict["loss"], 
                log_dict["mean_timestep"]
            ], device=self.device)
            
            # 使用 all_reduce 求和
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= self.world_size  # 求平均
            
            # 注意：grad_norm 通常已经在 fsdp_wrap 后的 clip_grad_norm_ 中聚合过了，
            # 或者是所有进程一致的，不需要再次 all_reduce。
            
            # Increment step
            self.step += 1
            
            # Logging
            if self.is_main_process and self.step % self.config.training.log_interval == 0:
                # 提取聚合后的数据
                avg_loss = metrics[0].item()
                avg_timestep = metrics[1].item()
                current_grad_norm = log_dict["grad_norm"].item()
                
                # 使用 Tensorboard Writer 记录聚合后的平均值
                self.writer.add_scalar("train/loss", avg_loss, self.step)
                self.writer.add_scalar("train/grad_norm", current_grad_norm, self.step)
                self.writer.add_scalar("train/timestep_mean", avg_timestep, self.step)
                
                # 控制台打印
                print(f"Step {self.step}: loss={avg_loss:.4f}, "
                      f"grad_norm={current_grad_norm:.4f}, "
                      f"mean_t={avg_timestep:.1f}")
            
            # 以下部分保持不变
            if self.step % self.config.training.save_interval == 0:
                torch.cuda.empty_cache()
                self.save_checkpoint()
                torch.cuda.empty_cache()
            
            if self.step % getattr(self.config.training, "gc_interval", 1000) == 0:
                if dist.get_rank() == 0:
                    import logging
                    logging.info("Running garbage collection.")
                import gc
                gc.collect()
                torch.cuda.empty_cache()