import io
import json
import random
import re
from typing import Dict, Iterable, List, Tuple

import torch
import torchvision.transforms as transforms
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader

from utils.bucket_utils import generate_simple_hw_buckets


def process_sample_to_dict(sample, max_turns=None, require_exact_turns: bool = False):
    """
    require_exact_turns: 默认 False，保持原行为；为 True 时仅保留
    `num_turns == max_turns` 的样本（不足 max_turns 的样本被丢弃，
    超过 max_turns 的样本会被截断后等于 max_turns 也保留）。
    """
    if 'json' not in sample:
        return None

    try:
        # 兼容多种命名：N.jpg / N.jpeg / N.png / frame_N.jpg / 任何 *_N.<img_ext> 等
        image_pattern = re.compile(r"(?:^|[^\d])(\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
        image_keys: Dict[int, str] = {}

        for key in sample.keys():
            if key in ("json", "__key__", "__url__"):
                continue
            match = image_pattern.search(key)
            if match:
                idx = int(match.group(1))
                image_keys.setdefault(idx, key)

        image_indices = sorted(image_keys.keys())
        if len(image_indices) < 2:
            return None
        if image_indices != list(range(len(image_indices))):
            return None

        json_data = json.loads(sample['json'])
        instructions = json_data.get("instructions", None)
        if not isinstance(instructions, list) or len(instructions) == 0:
            return None

        # 多轮必须统一语言：整个样本随机选择一个语言
        lang = random.choice(["zh", "en"])
        prompts = []
        for ins in instructions:
            if not isinstance(ins, dict):
                return None
            text = ins.get(lang, None)
            if not isinstance(text, str):
                fallback = "en" if lang == "zh" else "zh"
                text = ins.get(fallback, None)
            if not isinstance(text, str):
                return None
            prompts.append(text.strip())

        num_turns = len(image_indices) - 1
        if len(prompts) != num_turns:
            return None

        if max_turns is not None:
            max_turns = int(max_turns)
            if max_turns <= 0:
                return None
            if num_turns > max_turns:
                num_turns = max_turns
                prompts = prompts[:num_turns]
                image_indices = image_indices[:num_turns + 1]

        # 严格模式：要求 num_turns 正好等于 max_turns，否则丢弃。
        # 默认关闭，仅在调用方显式开启时生效。
        if require_exact_turns:
            if max_turns is None or num_turns != int(max_turns):
                return None

        key_frames = []
        for idx in image_indices:
            img = Image.open(io.BytesIO(sample[image_keys[idx]])).convert('RGB')
            key_frames.append(img)

        if len(key_frames) != num_turns + 1:
            return None

        h, w = key_frames[0].height, key_frames[0].width

        return {
            "key_frames": key_frames,
            "prompts": prompts,
            "sample_id": sample.get('__key__'),
            "num_turns": num_turns,
            "orig_h": h,
            "orig_w": w,
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


def _build_hw_bucket_list(
    image_size: Tuple[int, int],
    step_width: int = 16,
    step_height: int = 16,
    max_ratio: float = 4.0,
) -> List[Tuple[int, int]]:
    base_h, base_w = int(image_size[0]), int(image_size[1])
    hw_buckets = generate_simple_hw_buckets(
        base_height=base_h,
        base_width=base_w,
        step_width=step_width,
        step_height=step_height,
        max_ratio=max_ratio,
    )
    # generate_simple_hw_buckets 返回 (1, h, w)
    return [(int(h), int(w)) for _, h, w in hw_buckets]


def _pick_hw_bucket(orig_h: int, orig_w: int, hw_buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    if len(hw_buckets) == 0:
        return orig_h, orig_w

    orig_ratio = float(orig_h) / float(max(orig_w, 1))

    def score(bucket_hw: Tuple[int, int]) -> Tuple[float, float]:
        bh, bw = bucket_hw
        bucket_ratio = float(bh) / float(max(bw, 1))
        ratio_err = abs(orig_ratio - bucket_ratio)
        area_err = abs((orig_h * orig_w) - (bh * bw))
        return ratio_err, area_err

    return min(hw_buckets, key=score)


def _materialize_sample(
    sample: Dict,
    transform,
    num_frame_per_block: int,
) -> Dict:
    key_frames = [transform(img) for img in sample["key_frames"]]

    images = [key_frames[0]]
    n = num_frame_per_block
    for turn_idx in range(sample["num_turns"]):
        start_f, end_f = key_frames[turn_idx], key_frames[turn_idx + 1]
        for k in range(1, n + 1):
            w = k / n
            images.append((1.0 - w) * start_f + w * end_f)

    return {
        "images": torch.stack(images),
        "prompts": sample["prompts"],
        "sample_id": sample["sample_id"],
        "num_turns": sample["num_turns"],
    }


def bucket_by_turn_and_hw(
    data_stream: Iterable[Dict],
    batch_size: int,
    num_frame_per_block: int,
    hw_buckets: List[Tuple[int, int]],
):
    """
    按 (num_turns, h, w) 联合分桶。
    当某个桶样本数达到 batch_size 时，构建并 yield 该桶 batch。
    """
    buckets: Dict[Tuple[int, int, int], List[Dict]] = {}

    def center_crop_then_resize(img: Image.Image, target_h: int, target_w: int) -> torch.Tensor:
        src_w, src_h = img.size
        src_ratio = src_h / max(src_w, 1)
        tgt_ratio = target_h / max(target_w, 1)

        if src_ratio > tgt_ratio:
            crop_w = src_w
            crop_h = int(round(crop_w * tgt_ratio))
        else:
            crop_h = src_h
            crop_w = int(round(crop_h / tgt_ratio))

        crop_h = max(1, min(crop_h, src_h))
        crop_w = max(1, min(crop_w, src_w))
        top = max(0, (src_h - crop_h) // 2)
        left = max(0, (src_w - crop_w) // 2)

        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        resized = cropped.resize((target_w, target_h), Image.BICUBIC)

        tensor = transforms.ToTensor()(resized)
        tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(tensor)
        return tensor

    for sample in data_stream:
        turns = int(sample["num_turns"])
        h, w = _pick_hw_bucket(int(sample["orig_h"]), int(sample["orig_w"]), hw_buckets)
        bucket_key = (turns, h, w)

        if bucket_key not in buckets:
            buckets[bucket_key] = []

        buckets[bucket_key].append(sample)

        if len(buckets[bucket_key]) >= batch_size:
            batch_raw = buckets[bucket_key][:batch_size]
            buckets[bucket_key] = buckets[bucket_key][batch_size:]

            transform = lambda img: center_crop_then_resize(img, h, w)

            batch = [
                _materialize_sample(item, transform, num_frame_per_block)
                for item in batch_raw
            ]

            yield {
                "images": torch.stack([item["images"] for item in batch]),
                "prompts": [item["prompts"] for item in batch],
                "sample_ids": [item["sample_id"] for item in batch],
                "num_turns": turns,
                "bucket_h": h,
                "bucket_w": w,
            }


def create_multi_turn_dataloader(
    data_path: str,
    batch_size: int,
    num_frame_per_block: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    image_size: tuple = (512, 512),
    max_turns: int = None,
    bucket_step_width: int = 16,
    bucket_step_height: int = 16,
    bucket_max_ratio: float = 4.0,
    require_exact_turns: bool = False,
):
    with open(data_path, 'r', encoding='utf-8') as f:
        tar_paths = json.load(f).get('resolved_files', [])

    hw_buckets = _build_hw_bucket_list(
        image_size=image_size,
        step_width=bucket_step_width,
        step_height=bucket_step_height,
        max_ratio=bucket_max_ratio,
    )

    dataset = wds.DataPipeline(
        wds.SimpleShardList(tar_paths),
        # 先按 node / worker 切 shard，避免多机多卡重复读同一份数据
        # wds.split_by_node,
        # wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(1000) if shuffle else (lambda x: x),
        wds.map(lambda s: process_sample_to_dict(s, max_turns=max_turns, require_exact_turns=require_exact_turns)),
        wds.select(lambda x: x is not None),
        lambda stream: bucket_by_turn_and_hw(
            stream,
            batch_size=batch_size,
            num_frame_per_block=num_frame_per_block,
            hw_buckets=hw_buckets,
        ),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader