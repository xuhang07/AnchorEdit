#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import yaml
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from utils.misc import set_seed
from utils.multi_turn_dataset import create_multi_turn_dataloader
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


def load_config(config_path: str) -> OmegaConf:
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    config.i2v = False
    return config


def cfg_get(cfg: Any, key_path: str, default: Any) -> Any:
    current = cfg
    for key in key_path.split("."):
        if isinstance(current, DictConfig):
            if key not in current:
                return default
            current = current[key]
        elif isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        else:
            return default
    return current


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def save_tensor_as_png(tensor_chw: torch.Tensor, output_path: str) -> None:
    image = to_pil_image(tensor_chw.clamp(0, 1).cpu())
    image.save(output_path)


def normalize_generator_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = [
        "_orig_mod.",
        "_checkpoint_wrapped_module.",
        "module.",
    ]
    normalized = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        normalized[nk] = v
    return normalized


class MultiTurnI2VInferencePipeline(torch.nn.Module):
    def __init__(
        self,
        config: OmegaConf,
        device: torch.device,
        num_inference_steps: int,
        guidance_scale: float,
        negative_prompt: str,
        timeshift: float,
    ):
        super().__init__()
        self.config = config
        self.device = device

        mixed_precision = bool(cfg_get(config, "training.mixed_precision", True))
        self.generator_dtype = torch.bfloat16 if mixed_precision else torch.float32

        model_kwargs = cfg_get(config, "model_kwargs", {})
        if isinstance(model_kwargs, DictConfig):
            model_kwargs = OmegaConf.to_container(model_kwargs, resolve=True)

        self.model_name = model_kwargs["model_name"]
        self.num_frame_per_block = int(model_kwargs.get("num_frame_per_block", 1))
        self.num_train_timesteps = int(cfg_get(config, "training.num_train_timesteps", 1000))

        self.generator = WanDiffusionWrapper(**model_kwargs).to(device=device, dtype=self.generator_dtype).eval()
        self.text_encoder = WanTextEncoder(self.model_name).to(device=device).eval()
        self.vae = WanVAEWrapper(self.model_name).to(device=device).eval()

        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.num_inference_steps = int(num_inference_steps)
        self.timeshift = float(timeshift)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.shift = self.timeshift
        self.scheduler.num_train_timesteps = self.num_train_timesteps
        self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)

        self.vae_dtype = next(self.vae.model.parameters()).dtype

    def load_checkpoint(self, checkpoint_path: str, use_ema: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if use_ema and "generator_ema" in checkpoint:
                state_dict = checkpoint["generator_ema"]
            elif "generator" in checkpoint:
                state_dict = checkpoint["generator"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        state_dict = normalize_generator_state_dict_keys(state_dict)
        missing, unexpected = self.generator.load_state_dict(state_dict, strict=False)
        print(f"[Checkpoint] loaded from {checkpoint_path}")
        print(f"[Checkpoint] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    @torch.no_grad()
    def encode_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        text_dict = self.text_encoder([str(p) for p in prompts])
        flat_embeds = text_dict["prompt_embeds"].to(device=self.device, dtype=self.generator_dtype)
        num_prompts, seq_len, hidden_dim = flat_embeds.shape
        merged = flat_embeds.view(1, num_prompts * seq_len, hidden_dim)
        return {"prompt_embeds": merged}

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        return self.encode_prompts([prompt])

    def _initialize_kv_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seq_length: int,
        min_frames: int = 0,
    ) -> List[Dict[str, torch.Tensor]]:
        num_transformer_blocks = len(self.generator.model.blocks)
        num_heads = int(self.generator.model.num_heads)
        head_dim = int(self.generator.model.dim // self.generator.model.num_heads)
        local_attn_size = int(getattr(self.generator.model, "local_attn_size", -1))
        if local_attn_size != -1:
            kv_cache_size = local_attn_size * frame_seq_length
        else:
            kv_cache_size = 32760
        # 多轮推理时按需扩容：保证至少能容纳 min_frames 帧
        if min_frames > 0:
            need_size = int(min_frames) * frame_seq_length
            if need_size > kv_cache_size:
                kv_cache_size = need_size

        kv_cache: List[Dict[str, torch.Tensor]] = []
        for _ in range(num_transformer_blocks):
            kv_cache.append(
                {
                    "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "global_end_index": torch.zeros([1], dtype=torch.long, device=device),
                    "local_end_index": torch.zeros([1], dtype=torch.long, device=device),
                }
            )
        return kv_cache

    @staticmethod
    def _reset_kv_cache_indices(kv_cache: List[Dict[str, torch.Tensor]]) -> None:
        for layer_cache in kv_cache:
            layer_cache["global_end_index"].zero_()
            layer_cache["local_end_index"].zero_()

    @torch.no_grad()
    def _prefill_history_to_cache(
        self,
        history_latents: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        kv_cache: List[Dict[str, torch.Tensor]],
        frame_seq_length: int,
    ) -> None:
        bsz, history_len = history_latents.shape[:2]
        if history_len <= 0:
            return

        history_timestep = torch.zeros((bsz, 1), device=self.device, dtype=torch.float32)
        for frame_idx in range(history_len):
            current_latent = history_latents[:, frame_idx:frame_idx + 1]
            current_start = frame_idx * frame_seq_length
            # 与 KV cache 已写入长度对齐，避免 current_start 超过缓存可表达范围导致外推
            cache_aligned_start = int(kv_cache[0]["global_end_index"].item())
            self.generator(
                noisy_image_or_video=current_latent,
                conditional_dict=conditional_dict,
                timestep=history_timestep,
                kv_cache=kv_cache,
                current_start=cache_aligned_start,
                cache_start=cache_aligned_start,
            )

    @torch.no_grad()
    def infer_multi_turn(
        self,
        initial_frame: torch.Tensor,
        prompts: List[str],
        seed: int,
        max_turn: Optional[int] = None,
    ) -> List[torch.Tensor]:
        initial_video = initial_frame.unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=self.vae_dtype)
        history_latents = self.vae.encode_to_latent(initial_video).to(device=self.device, dtype=self.generator_dtype)

        prompt_list = [str(p) for p in prompts]
        total_turns = len(prompt_list)
        if max_turn is not None:
            total_turns = min(total_turns, int(max_turn))

        pred_frames: List[torch.Tensor] = []
        if total_turns <= 0:
            return pred_frames

        use_cfg = self.guidance_scale > 1.0
        use_kv_cache = bool(getattr(self.generator, "is_causal", False))
        timesteps = self.scheduler.timesteps.to(self.device)

        bsz, _, c, h, w = history_latents.shape
        patch_size = getattr(self.generator.model, "patch_size", (1, 2, 2))
        frame_seq_length = (h // int(patch_size[1])) * (w // int(patch_size[2]))

        kv_cache_cond = self._initialize_kv_cache(
            batch_size=bsz,
            dtype=self.generator_dtype,
            device=self.device,
            frame_seq_length=frame_seq_length,
            min_frames=int(total_turns) + 1,
        ) if use_kv_cache else None

        kv_cache_uncond = self._initialize_kv_cache(
            batch_size=bsz,
            dtype=self.generator_dtype,
            device=self.device,
            frame_seq_length=frame_seq_length,
            min_frames=int(total_turns) + 1,
        ) if (use_kv_cache and use_cfg) else None

        for turn_idx in range(total_turns):
            prompt_context = [str(p) for p in prompt_list[turn_idx : turn_idx + 1]]
            cond = self.encode_prompts(prompt_context)
            uncond = self.encode_prompts([self.negative_prompt] * len(prompt_context)) if use_cfg else None

            if use_kv_cache and kv_cache_cond is not None:
                self._reset_kv_cache_indices(kv_cache_cond)
                self._prefill_history_to_cache(
                    history_latents=history_latents,
                    conditional_dict=cond,
                    kv_cache=kv_cache_cond,
                    frame_seq_length=frame_seq_length,
                )

                if use_cfg and uncond is not None and kv_cache_uncond is not None:
                    self._reset_kv_cache_indices(kv_cache_uncond)
                    self._prefill_history_to_cache(
                        history_latents=history_latents,
                        conditional_dict=uncond,
                        kv_cache=kv_cache_uncond,
                        frame_seq_length=frame_seq_length,
                    )

            noise_gen = torch.Generator(device=self.device)
            noise_gen.manual_seed(seed + turn_idx)
            current_noisy = torch.randn(
                (bsz, 1, c, h, w),
                device=self.device,
                dtype=self.generator_dtype,
                generator=noise_gen,
            )

            for t in timesteps:
                t_val = float(t.item())
                t_current = torch.full(
                    (bsz, 1),
                    t_val,
                    device=self.device,
                    dtype=torch.float32,
                )

                if use_kv_cache and kv_cache_cond is not None:
                    current_start = history_latents.shape[1] * frame_seq_length
                    flow_cond, _ = self.generator(
                        noisy_image_or_video=current_noisy,
                        conditional_dict=cond,
                        timestep=t_current,
                        kv_cache=kv_cache_cond,
                        current_start=current_start,
                        cache_start=current_start,
                    )
                    flow_last = flow_cond
                else:
                    t_history = torch.zeros(
                        (bsz, history_latents.shape[1]),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    t_full = torch.cat([t_history, t_current], dim=1)
                    model_input = torch.cat([history_latents, current_noisy], dim=1)

                    flow_cond, _ = self.generator(
                        noisy_image_or_video=model_input,
                        conditional_dict=cond,
                        timestep=t_full,
                    )
                    flow_last = flow_cond[:, -1:]

                if use_cfg and uncond is not None:
                    if use_kv_cache and kv_cache_uncond is not None:
                        flow_uncond, _ = self.generator(
                            noisy_image_or_video=current_noisy,
                            conditional_dict=uncond,
                            timestep=t_current,
                            kv_cache=kv_cache_uncond,
                            current_start=current_start,
                            cache_start=current_start,
                        )
                        flow_last_uncond = flow_uncond
                    else:
                        flow_uncond, _ = self.generator(
                            noisy_image_or_video=model_input,
                            conditional_dict=uncond,
                            timestep=t_full,
                        )
                        flow_last_uncond = flow_uncond[:, -1:]
                    flow_last = flow_last_uncond + self.guidance_scale * (flow_last - flow_last_uncond)

                current_noisy = self.scheduler.step(
                    model_output=flow_last.flatten(0, 1),
                    timestep=t_current.flatten(0, 1),
                    sample=current_noisy.flatten(0, 1),
                ).unflatten(0, (bsz, 1))
                current_noisy = current_noisy.to(dtype=self.generator_dtype)

            pred_latent = current_noisy
            pred_pixel = self.vae.decode_to_pixel(pred_latent.to(dtype=self.vae_dtype))
            pred_pixel = (pred_pixel * 0.5 + 0.5).clamp(0, 1)
            pred_frames.append(pred_pixel[:, 0].detach().cpu())

            history_latents = torch.cat([history_latents, pred_latent], dim=1)

        return pred_frames


def build_or_load_cache(
    config: OmegaConf,
    cache_path: str,
    max_samples: int,
    refresh_cache: bool,
    max_turn: Optional[int] = None,
) -> List[Dict[str, Any]]:
    cache_file = Path(cache_path)
    if cache_file.exists() and not refresh_cache:
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        if len(cached) == 0:
            return cached
        first_sample = cached[0]
        if isinstance(first_sample, dict) and isinstance(first_sample.get("prompts", None), list):
            print(f"[Cache] loaded {len(cached)} samples from {cache_path}")
            return cached[:max_samples]
        print("[Cache] detected legacy single-turn cache format, rebuilding for multi-turn...")
        print("[Cache] tip: you can also use --refresh_cache to force rebuild.")

    print("[Cache] building cache from dataset...")
    loader_max_turns = max_turn if max_turn is not None else int(cfg_get(config, "data.max_turns", 1))
    dataloader = create_multi_turn_dataloader(
        data_path=cfg_get(config, "data.data_path", ""),
        batch_size=1,
        num_workers=0,
        num_frame_per_block=int(cfg_get(config, "model_kwargs.num_frame_per_block", 1)),
        max_turns=loader_max_turns,
        shuffle=False,
        image_size=tuple(cfg_get(config, "data.image_size", [480, 480])),
    )

    cached_samples: List[Dict[str, Any]] = []
    num_frame_per_block = int(cfg_get(config, "model_kwargs.num_frame_per_block", 1))

    for batch in tqdm(dataloader, desc="Caching samples"):
        if not batch:
            continue

        images = batch["images"]  # [1, 1 + turn*num_frame_per_block, 3, H, W], normalized to [-1, 1]
        prompts = batch["prompts"]
        sample_ids = batch["sample_ids"]

        prompt_list = prompts[0] if isinstance(prompts[0], list) else [prompts[0]]
        prompt_list = [str(p) for p in prompt_list]
        if max_turn is not None:
            prompt_list = prompt_list[: int(max_turn)]

        real_turns = len(prompt_list)
        gt_turn_frames = []
        for turn_idx in range(real_turns):
            frame_index = (turn_idx + 1) * num_frame_per_block
            if frame_index < images.shape[1]:
                gt_turn_frames.append(images[0, frame_index].cpu().to(torch.float16))

        sample_id = str(sample_ids[0])

        cached_samples.append(
            {
                "sample_id": sample_id,
                "prompt": str(prompt_list[0]) if len(prompt_list) > 0 else "",
                "prompts": prompt_list,
                "num_turns": int(real_turns),
                "initial_frame": images[0, 0].cpu().to(torch.float16),
                "gt_turn_frames": gt_turn_frames,
            }
        )

        if len(cached_samples) >= max_samples:
            break

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached_samples, cache_file)
    print(f"[Cache] saved {len(cached_samples)} samples to {cache_path}")
    return cached_samples


def init_distributed_env() -> tuple[bool, int, int, int, torch.device]:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    parser = argparse.ArgumentParser(description="Multi-turn I2V inference for first N cached samples")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--cache_path", type=str, default="cache/multi_turn_i2v_first50.pt")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--timeshift", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--maxturn", type=int, default=None, help="Maximum turns to infer per sample")
    args = parser.parse_args()

    is_distributed, rank, world_size, local_rank, device = init_distributed_env()
    set_seed(args.seed + rank)

    config = load_config(args.config)
    output_dir = Path(args.output_dir)

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier_if_needed(is_distributed)

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

    pipeline = MultiTurnI2VInferencePipeline(
        config=config,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        timeshift=timeshift,
    )
    pipeline.load_checkpoint(args.checkpoint, use_ema=args.use_ema)

    if is_distributed:
        if rank == 0:
            _ = build_or_load_cache(
                config=config,
                cache_path=args.cache_path,
                max_samples=args.max_samples,
                refresh_cache=args.refresh_cache,
                max_turn=args.maxturn,
            )
        barrier_if_needed(is_distributed)
        cached_samples = build_or_load_cache(
            config=config,
            cache_path=args.cache_path,
            max_samples=args.max_samples,
            refresh_cache=False,
            max_turn=args.maxturn,
        )
    else:
        cached_samples = build_or_load_cache(
            config=config,
            cache_path=args.cache_path,
            max_samples=args.max_samples,
            refresh_cache=args.refresh_cache,
            max_turn=args.maxturn,
        )

    assigned_indices = list(range(rank, len(cached_samples), world_size))
    local_items = [(sample_idx, cached_samples[sample_idx]) for sample_idx in assigned_indices]

    shard_jsonl = output_dir / f"results_rank{rank:02d}.jsonl"
    with open(shard_jsonl, "w", encoding="utf-8") as fout:
        for idx, sample in tqdm(local_items, desc=f"Inference rank{rank}", disable=(rank != 0)):
            raw_sample_id = str(sample["sample_id"])
            sample_id = sanitize_name(raw_sample_id)
            if not sample_id:
                sample_id = f"sample_{idx:04d}"

            initial_frame = sample["initial_frame"].to(dtype=torch.float32)

            prompt_list = sample.get("prompts", None)
            if prompt_list is None:
                legacy_prompt = str(sample.get("prompt", ""))
                prompt_list = [legacy_prompt] if legacy_prompt else []
            prompt_list = [str(p) for p in prompt_list]

            real_turns = int(sample.get("num_turns", len(prompt_list)))
            real_turns = min(real_turns, len(prompt_list))
            infer_turns = real_turns
            if args.maxturn is not None:
                infer_turns = min(infer_turns, int(args.maxturn))
            used_prompts = prompt_list[:infer_turns]

            pred_frames = pipeline.infer_multi_turn(
                initial_frame=initial_frame,
                prompts=used_prompts,
                seed=args.seed + idx * 1000,
                max_turn=infer_turns,
            )

            gt_turn_frames = sample.get("gt_turn_frames", [])
            gt_turn_frames = [x.to(dtype=torch.float32) for x in gt_turn_frames[:infer_turns]]

            src = ((initial_frame * 0.5) + 0.5).clamp(0, 1)

            sample_dir = output_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            src_path = sample_dir / "source.png"
            prompts_path = sample_dir / "prompts.json"
            save_tensor_as_png(src, str(src_path))
            with open(prompts_path, "w", encoding="utf-8") as pf:
                json.dump(used_prompts, pf, ensure_ascii=False, indent=2)

            pred_paths: List[str] = []
            gt_paths: List[str] = []
            for turn_idx in range(infer_turns):
                if turn_idx < len(pred_frames):
                    pred_img = pred_frames[turn_idx][0]
                    pred_path = sample_dir / f"pred_turn_{turn_idx + 1:02d}.png"
                    save_tensor_as_png(pred_img, str(pred_path))
                    pred_paths.append(str(pred_path))

                if turn_idx < len(gt_turn_frames):
                    gt_img = ((gt_turn_frames[turn_idx] * 0.5) + 0.5).clamp(0, 1)
                    gt_path = sample_dir / f"gt_turn_{turn_idx + 1:02d}.png"
                    save_tensor_as_png(gt_img, str(gt_path))
                    gt_paths.append(str(gt_path))

            record = {
                "index": idx,
                "sample_id": raw_sample_id,
                "sample_dir": str(sample_dir),
                "source_image": str(src_path),
                "prompts_file": str(prompts_path),
                "prompts": used_prompts,
                "num_turns_real": real_turns,
                "num_turns_inferred": infer_turns,
                "pred_images": pred_paths,
                "gt_images": gt_paths,
                "rank": rank,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    barrier_if_needed(is_distributed)
    if rank == 0:
        merge_result_shards(output_dir=output_dir, world_size=world_size)

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()