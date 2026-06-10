#!/usr/bin/env python3
"""
从一个包含 tar 列表的 json 中（{"resolved_files": [tar_path, ...]}），
取最后 N 个 tar，每个 tar 随机选 1 个 sample 解出来到 validation_root：

  <output_root>/<uuid>/
      <uuid>.0.jpg
      <uuid>.1.jpg
      ...
      <uuid>.<num_turns>.jpg
      <uuid>.json

文件命名遵从 webdataset 解出来的字段名规范：__key__ + 字段后缀。
本脚本不依赖 webdataset，直接用 tarfile 顺序扫描，保证与训练数据格式一致。
"""

import argparse
import json
import os
import random
import re
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _parse_member_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    将 tar 内文件名拆为 (sample_key, field)。

    与 webdataset 保持一致：以 basename（去掉 tar 内目录前缀）的第一个 '.' 之前为 key，
    之后为 field。例如 'a/b/uuid.0.jpg' -> ('uuid', '0.jpg')。
    """
    if not name:
        return None, None
    base = os.path.basename(name)
    if not base or base.startswith("."):
        return None, None
    if "." not in base:
        return None, None
    key, field = base.split(".", 1)
    if not key or not field:
        return None, None
    return key, field


def _is_image_field(field: str) -> bool:
    f = field.lower()
    # field 形如 '0.png' / '1.jpg' / 'frame_0.webp'，按整段或后缀匹配
    return any(f.endswith(ext) or f == ext.lstrip(".") for ext in IMAGE_EXTS)


def _is_json_field(field: str) -> bool:
    f = field.lower()
    # field 可能就是 'json'（webdataset 风格，第一个 '.' 之后即 field），也可能 'meta.json'
    return f == "json" or f.endswith(".json")


def _scan_tar_samples(tar_path: Path, debug: bool = False) -> Dict[str, Dict[str, str]]:
    """
    扫描一个 tar，按 sample key 聚合成员。返回 {key: {field: member_name}}。
    """
    samples: Dict[str, Dict[str, str]] = {}
    member_count = 0
    debug_examples: List[str] = []
    with tarfile.open(tar_path, "r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            member_count += 1
            if debug and len(debug_examples) < 5:
                debug_examples.append(member.name)
            key, field = _parse_member_name(member.name)
            if key is None or field is None:
                continue
            samples.setdefault(key, {})[field] = member.name

    if debug:
        print(f"  [Debug] tar={tar_path.name} total_members={member_count} "
              f"parsed_samples={len(samples)} example_names={debug_examples}")
        # 再展示一个 sample 的字段
        if samples:
            any_key = next(iter(samples))
            print(f"  [Debug] sample[{any_key}] fields={list(samples[any_key].keys())[:10]}")
    return samples


def _find_image_indices(fields: Dict[str, str]) -> List[int]:
    """
    从字段名中提取数字索引（兼容 '0.jpg' / 'frame_0.jpg' / 'image_0.png' 等）。
    返回排序后的连续索引；若不连续从 0 起，返回空列表表示该 sample 不可用。
    """
    pattern = re.compile(r"(?:^|[^\d])(\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
    idx_to_field: Dict[int, str] = {}
    for field in fields.keys():
        if not _is_image_field(field):
            continue
        m = pattern.search(field)
        if not m:
            continue
        idx = int(m.group(1))
        idx_to_field.setdefault(idx, field)

    if not idx_to_field:
        return []

    indices = sorted(idx_to_field.keys())
    if indices != list(range(len(indices))):
        return []
    return indices


def _select_valid_sample(
    sample_index: Dict[str, Dict[str, str]],
    rng: random.Random,
    max_try: int = 20,
    debug: bool = False,
) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    在 sample_index 中随机抽样若干次，挑一个同时具备：
      - json 字段
      - 至少 2 张连续编号的图片（含初始帧 + 至少 1 轮 GT）
    """
    keys = list(sample_index.keys())
    if not keys:
        return None
    rng.shuffle(keys)
    candidates = keys[:max_try] if len(keys) >= max_try else keys
    last_reason = "no_candidates"
    for key in candidates:
        fields = sample_index[key]
        json_field = None
        for f in fields:
            if _is_json_field(f):
                json_field = f
                break
        if json_field is None:
            last_reason = f"no_json (fields_sample={list(fields.keys())[:6]})"
            continue
        indices = _find_image_indices(fields)
        if len(indices) < 2:
            last_reason = (
                f"too_few_images (indices={indices}, "
                f"fields_sample={list(fields.keys())[:6]})"
            )
            continue
        return key, fields
    if debug:
        print(f"  [Debug] _select_valid_sample failed: tried={len(candidates)} last_reason={last_reason}")
    return None


def _extract_sample_to_dir(
    tar_path: Path,
    sample_key: str,
    sample_fields: Dict[str, str],
    output_dir: Path,
) -> List[str]:
    """
    把指定 sample 的所有成员提取到 output_dir，并以 '<sample_key>.<field>' 命名。
    返回写入的文件名列表。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    member_to_field = {member: field for field, member in sample_fields.items()}

    written: List[str] = []
    with tarfile.open(tar_path, "r:") as tf:
        for member in tf:
            if not member.isfile():
                continue
            if member.name not in member_to_field:
                continue
            field = member_to_field[member.name]
            out_name = f"{sample_key}.{field}"
            out_path = output_dir / out_name
            f = tf.extractfile(member)
            if f is None:
                continue
            data = f.read()
            with open(out_path, "wb") as wf:
                wf.write(data)
            written.append(out_name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validation_root by sampling from tar shards")
    parser.add_argument("--data_path", type=str, required=True,
                        help="包含 resolved_files 列表的 json 路径")
    parser.add_argument("--output_root", type=str, required=True,
                        help="输出 validation_root 目录")
    parser.add_argument("--num_tars", type=int, default=50,
                        help="从最后多少个 tar 中各取 1 个样本")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="若目标 uuid 目录已存在则覆盖（默认跳过）")
    parser.add_argument("--debug", action="store_true",
                        help="打印 tar 成员名示例与筛选失败原因")
    parser.add_argument("--max_try", type=int, default=0,
                        help="每个 tar 随机抽样尝试次数上限；0 表示遍历所有候选")

    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.data_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    tar_paths_all = meta.get("resolved_files", [])
    if not isinstance(tar_paths_all, list) or len(tar_paths_all) == 0:
        print(f"[Error] resolved_files 为空或缺失: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    tar_paths = tar_paths_all[-int(args.num_tars):]
    print(f"[Info] total tars={len(tar_paths_all)}, using last {len(tar_paths)} tars")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0

    for i, tp in enumerate(tar_paths):
        tar_path = Path(tp)
        if not tar_path.exists():
            print(f"[Skip][{i + 1}/{len(tar_paths)}] tar 不存在: {tar_path}")
            skipped += 1
            continue
        try:
            sample_index = _scan_tar_samples(tar_path, debug=args.debug)
        except Exception as e:
            print(f"[Fail][{i + 1}/{len(tar_paths)}] 扫描失败 {tar_path}: {e}")
            failed += 1
            continue
        if not sample_index:
            print(f"[Skip][{i + 1}/{len(tar_paths)}] 空 tar: {tar_path}")
            skipped += 1
            continue

        max_try = args.max_try if args.max_try and args.max_try > 0 else len(sample_index)
        picked = _select_valid_sample(sample_index, rng, max_try=max_try, debug=args.debug)
        if picked is None:
            print(f"[Skip][{i + 1}/{len(tar_paths)}] 无可用 sample: {tar_path}")
            skipped += 1
            continue

        sample_key, sample_fields = picked
        sample_dir = output_root / sample_key
        if sample_dir.exists() and not args.overwrite:
            print(f"[Skip][{i + 1}/{len(tar_paths)}] 已存在 {sample_dir}")
            skipped += 1
            continue

        try:
            written = _extract_sample_to_dir(
                tar_path=tar_path,
                sample_key=sample_key,
                sample_fields=sample_fields,
                output_dir=sample_dir,
            )
        except Exception as e:
            print(f"[Fail][{i + 1}/{len(tar_paths)}] 提取失败 {tar_path} key={sample_key}: {e}")
            failed += 1
            continue

        print(f"[OK][{i + 1}/{len(tar_paths)}] tar={tar_path.name} key={sample_key} files={len(written)}")
        success += 1

    print(f"[Done] success={success} skipped={skipped} failed={failed} output_root={output_root}")


if __name__ == "__main__":
    main()