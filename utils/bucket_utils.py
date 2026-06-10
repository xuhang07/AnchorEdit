from dataclasses import dataclass, field
import json
import importlib.util
import inspect
from pathlib import Path
from typing import Type
def _generate_hw_buckets(base_height=256, base_width=256, step_width=16, step_height=16, max_ratio=4.0) -> list[tuple[int, int, int, int, int]]:
    """Generate dimension buckets based on aspect ratios"""
    buckets = []
    target_pixels = base_height * base_width

    height = target_pixels // step_width
    width = step_width

    while height >= step_height:
        if max(height, width) / min(height, width) <= max_ratio:
            ratio = height / width
            buckets.append((1, 1, 1, height, width))
        # Try to increase width or decrease height
        if height * (width + step_width) <= target_pixels:
            width += step_width
        else:
            height -= step_height

    return buckets


def generate_video_image_bucket(basesize=256, min_temporal=65, max_temporal=129, bs_img=8, bs_vid=1, bs_mimg=4, min_items=1, max_items=1):
    # (batch_size, num_items, num_frames, height, width)
    assert basesize in [
        256, 512, 768, 1024], f"[generate_video_image_bucket] wrong basesize {basesize}"
    bucket_list = []
    # base_bucket_list = [
    #     (1, 1, 1, 512, 128),  # 4:1
    #     (1, 1, 1, 128, 512),
    #     (1, 1, 1, 192, 352),  # 16:9
    #     (1, 1, 1, 352, 192),
    #     (1, 1, 1, 288, 224),  # 4:3
    #     (1, 1, 1, 224, 288),
    #     (1, 1, 1, 320, 208),  # 3:2
    #     (1, 1, 1, 208, 320),
    #     (1, 1, 1, 368, 176),  # 2:1
    #     (1, 1, 1, 176, 368),
    #     (1, 1, 1, 256, 256),  # 1:1
    # ]

    base_bucket_list = _generate_hw_buckets()
    # image
    for _bucket in base_bucket_list:
        bucket = list(_bucket)
        bucket[0] = bs_img
        bucket_list.append(bucket)
    # video
    for temporal in range(min_temporal, max_temporal+1, 8):
        for _bucket in base_bucket_list:
            bucket = list(_bucket)
            bs = (max_temporal + 1) // temporal * bs_vid
            bucket[0] = bs
            bucket[2] = temporal
            bucket_list.append(bucket)
    # multiple images
    for num_items in range(min_items, max_items+1):
        for _bucket in base_bucket_list:
            bucket = list(_bucket)
            bucket[0] = bs_mimg
            bucket[1] = num_items
            bucket_list.append(bucket)
    # spatial resize
    if basesize > 256:
        ratio = basesize // 256

        def resize(bucket, r):
            bucket[-2] *= r
            bucket[-1] *= r
            return bucket
        bucket_list = [resize(bucket, ratio) for bucket in bucket_list]
    return bucket_list


def generate_token_bucket(basesize=1024, vae_s=16, p_s=1, min_items=1, max_items=6, base_step = 512):
    """
    动态步长分桶生成函数
    num_steps_per_item: 每增加一个组件，增加多少个长度档位（控制桶的密度）
    """
    bucket_list = []
    align = vae_s * p_s
    # 单图基准 Token 数 (1024/16)^2 = 4096
    base_tokens = (basesize // align) ** 2
    

    for n in range(min_items, max_items + 1):
        # 这里的 step 随 n 增长：n=1 时 step=512, n=6 时 step=3072
        current_step = n * base_step
        max_tokens = n * base_tokens
        
        # 生成该组件数下的阶梯
        for t_len in range(current_step, max_tokens + current_step * 2, current_step):
            bucket_list.append((n, t_len))
            
    return bucket_list

def generate_simple_token_bucket(max_len=4096, max_items=6):
    """
    极简分桶：每个组件数只对应一个固定最大长度的桶。
    例如：(1, 4096), (2, 4096), ...
    """
    bucket_list = []
    for n in range(1, max_items + 1):
        # 这里的 max_len 应该是你模型能够处理的最大序列长度
        bucket_list.append((n, max_len))
    return bucket_list


def generate_simple_hw_buckets(base_height=256, base_width=256, step_width=16, step_height=16, max_ratio=4.0) -> list[tuple[int, int, int]]:
    """Generate dimension buckets based on aspect ratios"""
    buckets = []
    target_pixels = base_height * base_width

    height = target_pixels // step_width
    width = step_width

    while height >= step_height:
        if max(height, width) / min(height, width) <= max_ratio:
            ratio = height / width
            buckets.append((1, height, width))
        # Try to increase width or decrease height
        if height * (width + step_width) <= target_pixels:
            width += step_width
        else:
            height -= step_height

    return buckets

