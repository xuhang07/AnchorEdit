#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import json
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return default


def extract_edit_prompts(meta: Dict[str, Any], required_turns: int) -> Optional[List[str]]:
    rounds = meta.get("rounds", [])
    if not isinstance(rounds, list):
        return None

    turn_to_prompt: Dict[int, str] = {}
    for item in rounds:
        if not isinstance(item, dict):
            continue
        try:
            turn = int(item.get("turn", -1))
        except Exception:
            continue
        prompt_obj = item.get("prompt", {})
        if not isinstance(prompt_obj, dict):
            continue
        edit_prompt = str(prompt_obj.get("edit_prompt", "")).strip()
        if edit_prompt:
            turn_to_prompt[turn] = edit_prompt

    prompts: List[str] = []
    for t in range(1, required_turns + 1):
        p = turn_to_prompt.get(t, "")
        if not p:
            return None
        prompts.append(p)
    return prompts


def collect_basic_source(meta: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    source = meta.get("source", {})
    if not isinstance(source, dict):
        return out

    for k in ("caption", "image_path", "tar_path", "tar_member"):
        v = source.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def validate_sample(sample_dir: Path, required_turns: int) -> Optional[Dict[str, Any]]:
    sample_json = sample_dir / "sample.json"
    if not sample_json.exists():
        return None

    meta = load_json(sample_json, default={})
    prompts = extract_edit_prompts(meta, required_turns=required_turns)
    if prompts is None:
        return None

    images: List[Path] = []
    for i in range(required_turns + 1):
        p = sample_dir / f"turn_{i}.jpg"
        if not p.exists():
            return None
        images.append(p)

    return {
        "old_sample_id": sample_dir.name,
        "images": images,
        "edit_prompt": prompts,
        "source": collect_basic_source(meta),
    }


def build_output_json(new_id: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "sample_id": new_id,
        "old_sample_id": sample["old_sample_id"],
        "num_turns": len(sample["edit_prompt"]),
        "edit_prompt": sample["edit_prompt"],
    }
    source = sample.get("source", {})
    if isinstance(source, dict) and source:
        payload["source"] = source
    return payload


def write_one_tar(tar_path: Path, samples: List[Dict[str, Any]]) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    now = int(time.time())

    with tarfile.open(tar_path, "w") as tar:
        for sample in samples:
            new_id = uuid.uuid4().hex

            for idx, img_path in enumerate(sample["images"]):
                arcname = f"{new_id}.{idx}.jpg"
                tar.add(str(img_path), arcname=arcname, recursive=False)

            data = json.dumps(
                build_output_json(new_id, sample),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            info = tarfile.TarInfo(name=f"{new_id}.json")
            info.size = len(data)
            info.mtime = now
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))


def discover_samples(samples_root: Path) -> List[Path]:
    if not samples_root.exists():
        return []
    return sorted([p for p in samples_root.iterdir() if p.is_dir()], key=lambda x: x.name)


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    out: List[List[Any]] = []
    for i in range(0, len(items), size):
        out.append(items[i:i + size])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 root 下 sample 打包成 tar，每10个sample一个tar，重置uuid并输出精简json。"
    )
    parser.add_argument("--samples_root", type=str, required=True, help="输入sample根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="tar输出目录")
    parser.add_argument("--samples_per_tar", type=int, default=10, help="每个tar包含多少sample，默认10")
    parser.add_argument("--required_turns", type=int, default=5, help="必须具备的编辑轮数，默认5")
    parser.add_argument("--tar_prefix", type=str, default="samples", help="输出tar文件名前缀")
    args = parser.parse_args()

    if args.samples_per_tar < 1:
        raise ValueError("--samples_per_tar 必须 >= 1")
    if args.required_turns < 1:
        raise ValueError("--required_turns 必须 >= 1")

    samples_root = Path(args.samples_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dirs = discover_samples(samples_root)
    valid_samples: List[Dict[str, Any]] = []
    dropped = 0

    for sample_dir in all_dirs:
        item = validate_sample(sample_dir, required_turns=args.required_turns)
        if item is None:
            dropped += 1
            continue
        valid_samples.append(item)

    chunks = chunk_list(valid_samples, args.samples_per_tar)
    for idx, chunk in enumerate(chunks):
        tar_path = output_dir / f"{args.tar_prefix}_{idx:06d}.tar"
        write_one_tar(tar_path, chunk)

    print(
        f"[done] total_dirs={len(all_dirs)}, valid={len(valid_samples)}, "
        f"dropped={dropped}, tar_files={len(chunks)}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()