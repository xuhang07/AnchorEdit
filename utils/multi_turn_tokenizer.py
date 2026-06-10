import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np


import torch
from typing import List, Dict, Optional

class UnifiedTokenizer:
    """
    统一的Tokenizer，支持多轮图像编辑和文本到图像任务。
    通过配置参数控制不同任务的行为：
    - independent_first_frame: True for I2I, False for T2I
    - max_turns: Number of generation turns
    """
    
    def __init__(self, text_encoder, vae, dtype, device='cuda', config=None):
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.config = config
        
        # 从配置中提取参数
        self.independent_first_frame = getattr(config.model_kwargs, "independent_first_frame", True) if config else True
        self.max_turns = getattr(config.data, "max_turns", 1) if config else 1
        
    def encode_texts(self, prompts: List[List[str]]) -> torch.Tensor:
        """
        并行编码所有 Prompt。
        Args:
            prompts: List[List[str]], 形状 [B, num_prompts]
        Returns:
            torch.Tensor: [B, num_prompts * seq_len, hidden_dim]
        """
        batch_size = len(prompts)
        num_prompts = len(prompts[0])
        
        # 将 [B, T] 展平为 [B*T] 以便并行编码
        flat_prompts = [p for sublist in prompts for p in sublist]
        
        # 编码所有文本 (假设 text_encoder 支持 batch 输入)
        # 返回通常是 {'prompt_embeds': [B*T, seq_len, hidden_dim]}
        text_dict = self.text_encoder(flat_prompts)
        
        if isinstance(text_dict, dict) and 'prompt_embeds' in text_dict:
            flat_embeds = text_dict['prompt_embeds']
        else:
            flat_embeds = text_dict
            
        # flat_embeds shape: [B*T, L, D]
        _, seq_len, hidden_dim = flat_embeds.shape
        
        # 重新组合并拼接: [B, T, L, D] -> [B, T*L, D]
        final_embeds = flat_embeds.view(batch_size, num_prompts, seq_len, hidden_dim)
        final_embeds = final_embeds.reshape(batch_size, num_prompts * seq_len, hidden_dim)
        
        return final_embeds

    def encode_images(self, images: torch.Tensor, max_vae_batch: int = 8) -> torch.Tensor:
        """
        Args:
            images: [B, T, C, H, W]
        Returns:
            [B, C_latent, T, H_latent, W_latent]
        """
        batch_size, num_images, c, h, w = images.shape
        
        # 1. 展平所有图片: [B*T, C, 1, H, W]
        # 注意：Wan VAE 通常需要 5D 输入 [N, C, T, H, W]，这里的 1 代表单帧视频
        flat_images = images.view(batch_size*num_images, c, 1, h, w)
        
        all_latents = []
        
        for i in range(0, flat_images.shape[0], max_vae_batch):
            chunk = flat_images[i : i + max_vae_batch].to(self.device, dtype=self.dtype)
            with torch.no_grad():
                # Wan VAE 返回通常是 [N, 16, 1, H/8, W/8]
                latent_chunk = self.vae.encode_to_latent(chunk)
                all_latents.append(latent_chunk.cpu())

        # 2. 合并结果: [B*T, 1, C_lat, H_lat, W_lat]
        latents = torch.cat(all_latents, dim=0)
        
        # 关键点：从编码后的结果中动态获取实际的 C, H, W
        _, t_lat, c_lat, h_lat, w_lat = latents.shape
        
        # 3. 恢复维度
        # 首先恢复 B 和 T: [B, T, C_lat, H_lat, W_lat]
        # 这里 t_lat 通常是 1 (因为我们是按单帧编码的)
        final_latents = latents.view(batch_size, num_images*t_lat, c_lat, h_lat, w_lat)
        
        return final_latents.to(self.device)

    def __call__(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        处理分桶后的 Batch，根据配置参数处理I2I或T2I任务。
        """
        # images 已经是 Tensor: [B, T, C, H, W]
        image_latents = self.encode_images(batch['images'])
        
        # prompts 是 List[List[str]]
        prompt_embeds = self.encode_texts(batch['prompts'])
        
        # 根据任务类型调整输出格式
        if self.independent_first_frame:
            # I2I任务：保留原始格式
            num_turns = batch['num_turns']
        else:
            # T2I任务：调整为单轮格式
            num_turns = 1
        
        return {
            'image_latents': image_latents, # [B, C, T, H, W]
            'prompt_embeds': prompt_embeds, # [B, T*L, D]
            'num_turns': num_turns
        }

# 向后兼容的别名
MultiTurnTokenizer = UnifiedTokenizer

# 使用示例：
# tokenizer = UnifiedTokenizer(text_encoder, vae, torch.bfloat16, config=config)
# processed_batch = tokenizer(batch_from_dataloader)


def prepare_unified_batch(
    batch: Dict[str, List],
    text_encoder,
    vae,
    device: str = 'cuda',
    config=None
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to process a unified batch for both I2I and T2I tasks.
    
    Args:
        batch: Raw batch from dataloader
        text_encoder: Text encoder model
        vae: VAE model
        device: Target device
        config: Configuration object
        
    Returns:
        Processed batch with encoded features
    """
    tokenizer = UnifiedTokenizer(text_encoder, vae, device, config=config)
    return tokenizer(batch)


class UnifiedCollator:
    """
    Unified collator for both multi-turn image editing and text-to-image batches.
    Handles padding and batching of variable-length sequences.
    """
    
    def __init__(self, text_encoder, vae, device='cuda', config=None):
        self.tokenizer = UnifiedTokenizer(text_encoder, vae, device, config=config)
        
    def __call__(self, batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into a batch.
        
        Args:
            batch_list: List of samples from dataset
            
        Returns:
            Batched and processed features
        """
        # Separate images and prompts
        all_images = []
        all_prompts = []
        
        for sample in batch_list:
            all_images.append(sample['images'])
            all_prompts.append(sample['prompts'])
        
        # Create batch dict
        batch = {
            'images': all_images,
            'prompts': all_prompts
        }
        
        # Process through tokenizer
        return self.tokenizer(batch)

# 向后兼容的别名
MultiTurnCollator = UnifiedCollator
prepare_multi_turn_batch = prepare_unified_batch