#!/usr/bin/env python3
"""
Multi-turn I2V validation inference.

输入目录组织（与 single-turn 类似，但每个 uuid 下可能有多轮 GT）：
  <validation_root>/<uuid>/
      <uuid>.0.jpg            # 初始帧（必需）
      <uuid>.1.jpg            # 第 1 轮 GT（可选）
      <uuid>.2.jpg            # 第 2 轮 GT（可选）
      ...
      <uuid>.json             # 含 instructions 字段，每项为 {"zh": "...", "en": "...", "is_no_change": bool}

输出每个 uuid 一个子目录：
  source.png
  prompts.json
  pred_turn_01.png ... pred_turn_NN.png
  gt_turn_01.jpg  ... gt_turn_NN.jpg     # 仅当对应 GT 存在时保存
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import yaml
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from inference_multi_turn_i2v import (
    MultiTurnI2VInferencePipeline,
    cfg_get,
    save_tensor_as_png,
)
from utils.misc import set_seed
from utils.multi_turn_dataset import _build_hw_bucket_list, _pick_hw_bucket


def load_config(config_path: str) -> OmegaConf:
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    config.i2v = False
    return config


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def read_prompts_from_json(json_path: Path, lang: str) -> List[str]:
    """
    读取 instructions 列表，按 lang 选择字符串；如该语言缺失则回退到另一语言。
    兼容旧格式（conversations[0].value 字符串）。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"json 不是 dict: {json_path}")

    # 新格式：instructions 列表
    instructions = obj.get("instructions", None)
    if isinstance(instructions, list) and len(instructions) > 0:
        prompts: List[str] = []
        for ins in instructions:
            if not isinstance(ins, dict):
                raise ValueError(f"instructions 项不是 dict: {json_path}")
            text = ins.get(lang, None)
            if not isinstance(text, str):
                fallback = "en" if lang == "zh" else "zh"
                text = ins.get(fallback, None)
            if not isinstance(text, str):
                raise ValueError(f"instructions 项缺少 {lang}/{fallback} 文本: {json_path}")
            prompts.append(text.strip())
        return prompts

    # 兼容旧格式
    convs = obj.get("conversations", [])
    for turn in convs:
        if isinstance(turn, dict) and turn.get("from") == "human":
            value = turn.get("value", "")
            value = value.replace("<image>\n", "").replace("<image>", "").strip()
            if value:
                return [value]
    raise ValueError(f"json 中未找到 instructions 或 conversations/human: {json_path}")


def load_image_as_model_input(image_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    src_w, src_h = image.size
    tgt_ratio = target_h / max(target_w, 1)
    src_ratio = src_h / max(src_w, 1)

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

    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    resized = cropped.resize((target_w, target_h), Image.BICUBIC)

    tensor = pil_to_tensor(resized).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor


def _find_indexed_image(sub: Path, uuid: str, idx: int) -> Optional[Path]:
    """在 <sub>/<uuid>.<idx>.<ext> 中按常见扩展名探测一张图片。"""
    for ext in ("jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG", "WEBP"):
        p = sub / f"{uuid}.{idx}.{ext}"
        if p.exists():
            return p
    return None


def collect_samples(validation_root: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    扫描 <validation_root>/<uuid>/，收集：
      - source: <uuid>.0.{jpg|png|jpeg|webp}
      - gts: 连续的 <uuid>.{1..N}.{jpg|png|jpeg|webp}
      - meta: <uuid>.json
    缺少 source 或 meta 直接跳过。
    """
    if not validation_root.exists():
        raise FileNotFoundError(f"validation root 不存在: {validation_root}")

    samples: List[Dict[str, Any]] = []
    for sub in sorted(validation_root.iterdir()):
        if not sub.is_dir():
            continue

        uuid = sub.name
        src = _find_indexed_image(sub, uuid, 0)
        meta = sub / f"{uuid}.json"
        if src is None or not meta.exists():
            continue

        # 收集多轮 GT：从 1 起递增直到不存在
        gts: List[Path] = []
        idx = 1
        while True:
            gt_path = _find_indexed_image(sub, uuid, idx)
            if gt_path is None:
                break
            gts.append(gt_path)
            idx += 1

        samples.append(
            {
                "uuid": uuid,
                "source_path": src,
                "gt_paths": gts,
                "json_path": meta,
            }
        )

        if limit is not None and len(samples) >= limit:
            break

    return samples


def init_distributed_env(device_arg: str) -> Tuple[bool, int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return True, rank, world_size, local_rank, device

    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if device_arg.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA 设备，但当前不可用")
        device = torch.device(device_arg)
    return False, 0, 1, 0, device


def barrier_if_needed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.barrier()


def merge_result_shards(output_dir: Path, world_size: int) -> None:
    shard_paths = [output_dir / f"results_rank{rank:02d}.jsonl" for rank in range(world_size)]
    merged: List[Dict[str, Any]] = []
    for shard_path in shard_paths:
        if not shard_path.exists():
            continue
        with open(shard_path, "r", encoding="utf-8") as sf:
            for line in sf:
                line = line.strip()
                if not line:
                    continue
                merged.append(json.loads(line))
    merged.sort(key=lambda x: int(x.get("index", 0)))
    merged_path = output_dir / "results.jsonl"
    with open(merged_path, "w", encoding="utf-8") as mf:
        for record in merged:
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[Done] Merged results saved to: {merged_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WAN multi-turn edit inference on validation_samples folders")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--validation_root", type=str,
        default="/path/to/validation_samples",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_turn", type=int, default=None,
                        help="单条样本最多推理多少轮（None 表示由 instructions 决定）")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"],
                        help="选择 instructions 中的语言（缺失时自动回退另一语言）")

    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--timeshift", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")

    args = parser.parse_args()

    is_distributed, rank, world_size, local_rank, device = init_distributed_env(args.device)
    set_seed(args.seed + rank)

    config = load_config(args.config)

    num_inference_steps = int(args.num_inference_steps) if args.num_inference_steps is not None else int(
        cfg_get(config, "num_inference_steps", 50)
    )
    timeshift = float(args.timeshift) if args.timeshift is not None else float(
        cfg_get(config, "timeshift", cfg_get(config, "timestep_shift", cfg_get(config, "model_kwargs.timestep_shift", 1.0)))
    )
    guidance_scale = float(args.guidance_scale) if args.guidance_scale is not None else float(
        cfg_get(config, "training.guidance_scale", 1.0)
    )
    negative_prompt = args.negative_prompt if args.negative_prompt is not None else str(
        cfg_get(config, "training.negative_prompt", "")
    )

    raw_image_size = cfg_get(config, "data.image_size", None)
    if rank == 0:
        print(f"[ConfigDebug] config_path={args.config}")
        print(f"[ConfigDebug] raw data.image_size={raw_image_size}, type={type(raw_image_size)}")

    if isinstance(raw_image_size, (ListConfig, list, tuple)) and len(raw_image_size) == 2:
        image_size = (int(raw_image_size[0]), int(raw_image_size[1]))
    else:
        raise ValueError(
            f"invalid data.image_size in config: path={args.config}, value={raw_image_size}, type={type(raw_image_size)}. "
            "expected [H, W], e.g. [1024, 1024]"
        )

    bucket_step_width = int(cfg_get(config, "data.bucket_step_width", 16))
    bucket_step_height = int(cfg_get(config, "data.bucket_step_height", 16))
    bucket_max_ratio = float(cfg_get(config, "data.bucket_max_ratio", 4.0))
    hw_buckets = _build_hw_bucket_list(
        image_size=image_size,
        step_width=bucket_step_width,
        step_height=bucket_step_height,
        max_ratio=bucket_max_ratio,
    )

    if rank == 0:
        print(f"[ConfigDebug] parsed image_size={image_size}")
        print(
            f"[BucketConfig] image_size={image_size}, step=({bucket_step_height},{bucket_step_width}), "
            f"max_ratio={bucket_max_ratio}, num_buckets={len(hw_buckets)}"
        )

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier_if_needed(is_distributed)

    pipeline = MultiTurnI2VInferencePipeline(
        config=config,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        timeshift=timeshift,
    )
    pipeline.load_checkpoint(args.checkpoint, use_ema=args.use_ema)

    samples = collect_samples(Path(args.validation_root), limit=args.max_samples)
    if rank == 0:
        print(f"[Data] loaded {len(samples)} samples from {args.validation_root}")
        if len(samples) == 0:
            # 给出诊断信息
            root = Path(args.validation_root)
            if not root.exists():
                print(f"[Data][Debug] validation_root 不存在: {root}")
            else:
                subs = [p for p in root.iterdir() if p.is_dir()]
                print(f"[Data][Debug] validation_root 下子目录数={len(subs)}")
                for sub in subs[:3]:
                    files = sorted(os.listdir(sub))[:10]
                    print(f"[Data][Debug]   {sub.name}/ 样例文件={files}")
        else:
            head = samples[:3]
            for s in head:
                print(
                    f"[Data][Debug] uuid={s['uuid']} source={s['source_path'].name} "
                    f"num_gt={len(s['gt_paths'])} json={s['json_path'].name}"
                )

    assigned_indices = list(range(rank, len(samples), world_size))
    local_items = [(sample_idx, samples[sample_idx]) for sample_idx in assigned_indices]

    shard_results_path = output_dir / f"results_rank{rank:02d}.jsonl"
    with open(shard_results_path, "w", encoding="utf-8") as fout:
        for global_idx, sample in tqdm(local_items, desc=f"Multi-turn WAN rank{rank}", disable=(rank != 0)):
            uuid = sample["uuid"]
            safe_uuid = sanitize_name(uuid) or f"sample_{global_idx:04d}"

            try:
                prompts_all = read_prompts_from_json(sample["json_path"], lang=args.lang)
            except Exception as e:
                print(f"[Skip][rank{rank}] read_prompts failed for {sample['json_path']}: {e}")
                continue

            if len(prompts_all) == 0:
                print(f"[Skip][rank{rank}] empty instructions: {sample['json_path']}")
                continue

            # 决定本次实际推理的轮数：受 --max_turn 限制
            num_turns = len(prompts_all)
            if args.max_turn is not None:
                num_turns = min(num_turns, int(args.max_turn))
            if num_turns <= 0:
                continue
            used_prompts = prompts_all[:num_turns]

            # 选择 hw bucket（与 single-turn 一致：按源图原始长宽匹配最近桶）
            with Image.open(sample["source_path"]) as src_img:
                orig_w, orig_h = src_img.size
            target_h, target_w = _pick_hw_bucket(orig_h, orig_w, hw_buckets)

            initial_frame = load_image_as_model_input(
                sample["source_path"], target_h=target_h, target_w=target_w
            )

            # 多轮推理：内部正确处理 KV cache（每轮 reset + prefill history）
            pred_frames = pipeline.infer_multi_turn(
                initial_frame=initial_frame,
                prompts=used_prompts,
                seed=args.seed + global_idx * 1000,
                max_turn=num_turns,
            )
            if len(pred_frames) == 0:
                print(f"[Skip][rank{rank}] no prediction generated for {uuid}")
                continue

            sample_dir = output_dir / safe_uuid
            sample_dir.mkdir(parents=True, exist_ok=True)

            # 保存 source
            source_tensor = (initial_frame * 0.5 + 0.5).clamp(0, 1)
            source_out = sample_dir / "source.png"
            save_tensor_as_png(source_tensor, str(source_out))

            # 保存 prompts.json
            prompts_out = sample_dir / "prompts.json"
            with open(prompts_out, "w", encoding="utf-8") as pf:
                json.dump(
                    {"lang": args.lang, "instructions": used_prompts},
                    pf, ensure_ascii=False, indent=2,
                )

            # 逐轮保存 pred 与 gt（gt 仅在原文件存在时保存）
            pred_paths: List[str] = []
            gt_paths: List[str] = []
            gt_paths_src = sample["gt_paths"]
            for turn_idx in range(num_turns):
                pred_tensor = pred_frames[turn_idx][0].clamp(0, 1)
                if pred_tensor.shape[-2] != target_h or pred_tensor.shape[-1] != target_w:
                    raise RuntimeError(
                        f"model output size mismatch at turn {turn_idx + 1}: "
                        f"pred=({pred_tensor.shape[-2]},{pred_tensor.shape[-1]}) "
                        f"bucket=({target_h},{target_w}) uuid={uuid}"
                    )

                pred_out = sample_dir / f"pred_turn_{turn_idx + 1:02d}.png"
                save_tensor_as_png(pred_tensor, str(pred_out))
                pred_paths.append(str(pred_out))

                if turn_idx < len(gt_paths_src):
                    gt_src = gt_paths_src[turn_idx]
                    gt_tensor = load_image_as_model_input(
                        gt_src, target_h=target_h, target_w=target_w
                    )
                    gt_save = (gt_tensor * 0.5 + 0.5).clamp(0, 1)
                    gt_out = sample_dir / f"gt_turn_{turn_idx + 1:02d}.png"
                    save_tensor_as_png(gt_save, str(gt_out))
                    gt_paths.append(str(gt_out))

            print(
                f"[Size][rank{rank}] {uuid} orig=({orig_h},{orig_w}) bucket=({target_h},{target_w}) "
                f"turns={num_turns} preds={len(pred_paths)} gts={len(gt_paths)}"
            )

            record: Dict[str, Any] = {
                "index": global_idx,
                "uuid": uuid,
                "sample_dir": str(sample_dir),
                "source_image": str(source_out),
                "prompts_file": str(prompts_out),
                "prompts": used_prompts,
                "lang": args.lang,
                "num_turns_total": len(prompts_all),
                "num_turns_inferred": num_turns,
                "pred_images": pred_paths,
                "gt_images": gt_paths,
                "rank": rank,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    barrier_if_needed(is_distributed)
    if rank == 0:
        merge_result_shards(output_dir=output_dir, world_size=world_size)
    barrier_if_needed(is_distributed)

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()