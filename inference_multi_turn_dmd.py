#!/usr/bin/env python3
"""
Multi-turn DMD 4-step distilled model inference.

与 trainer/multi_turn_dmd.py 中 `_self_rollout_history` / `_run_generator_at_k`
流程严格对齐：
  - 用 config.denoising_step_list（默认 [1000, 750, 500, 250]）做少步推理
  - 每个去噪步：generator -> pred_x0 -> add_noise(next_ts) 迭代
  - 最后一步直接拿 pred_x0 当 clean_t，再以 t=context_noise(=0) 写回 KV cache
  - generator 默认不开 CFG（DMD 蒸馏后的 generator 已经把 teacher CFG 蒸到权重里）

输入 / 输出目录组织与 inference_multi_turn_i2v_validation.py 一致：
  <validation_root>/<uuid>/{<uuid>.{0..N}.jpg, <uuid>.json}
  <output_dir>/<uuid>/{source.png, prompts.json, pred_turn_*.png, gt_turn_*.png}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from inference_multi_turn_i2v import (
    cfg_get,
    normalize_generator_state_dict_keys,
    save_tensor_as_png,
)
from inference_multi_turn_i2v_validation import (
    barrier_if_needed,
    collect_samples,
    init_distributed_env,
    load_image_as_model_input,
    merge_result_shards,
    read_prompts_from_json,
    sanitize_name,
)
from utils.misc import set_seed
from utils.multi_turn_dataset import _build_hw_bucket_list, _pick_hw_bucket
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


class MultiTurnDMDInferencePipeline(torch.nn.Module):
    """4-step DMD inference. 与 trainer.multi_turn_dmd._self_rollout_history 完全对齐。"""

    def __init__(
        self,
        config: OmegaConf,
        device: torch.device,
        denoising_step_list: List[int],
        guidance_scale: float,
        negative_prompt: str,
        timeshift: float,
        context_noise: int = 0,
    ):
        super().__init__()
        self.config = config
        self.device = device

        mixed_precision = bool(cfg_get(config, "mixed_precision", cfg_get(config, "training.mixed_precision", True)))
        self.generator_dtype = torch.bfloat16 if mixed_precision else torch.float32

        model_kwargs = cfg_get(config, "model_kwargs", {})
        if isinstance(model_kwargs, DictConfig):
            model_kwargs = OmegaConf.to_container(model_kwargs, resolve=True)

        self.model_name = model_kwargs["model_name"]
        self.num_frame_per_block = int(model_kwargs.get("num_frame_per_block", 1))
        self.num_train_timesteps = int(cfg_get(config, "num_train_timestep", 1000))

        self.generator = WanDiffusionWrapper(**model_kwargs).to(
            device=device, dtype=self.generator_dtype
        ).eval()
        self.text_encoder = WanTextEncoder(self.model_name).to(device=device).eval()
        self.vae = WanVAEWrapper(self.model_name).to(device=device).eval()

        self.guidance_scale = float(guidance_scale)
        self.negative_prompt = str(negative_prompt or "")
        self.context_noise = int(context_noise)
        self.timeshift = float(timeshift)

        # 关键：用 denoising_step_list 而非 num_inference_steps
        self.denoising_step_list = torch.tensor(
            list(denoising_step_list), dtype=torch.long, device=device
        )

        # scheduler 仅用于 add_noise（中间步骤插值）
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.shift = self.timeshift
        self.scheduler.num_train_timesteps = self.num_train_timesteps
        # 训练时 scheduler 也按 num_train_timesteps 跑（用于 add_noise 索引）
        self.scheduler.set_timesteps(num_inference_steps=self.num_train_timesteps, training=True)

        self.vae_dtype = next(self.vae.model.parameters()).dtype

    def load_checkpoint(self, checkpoint_path: str, use_ema: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if use_ema and "generator_ema" in checkpoint and checkpoint["generator_ema"] is not None:
                state_dict = checkpoint["generator_ema"]
                source = "generator_ema"
            elif "generator" in checkpoint:
                state_dict = checkpoint["generator"]
                source = "generator"
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
                source = "model"
            else:
                state_dict = checkpoint
                source = "<root>"
        else:
            state_dict = checkpoint
            source = "<raw>"

        state_dict = normalize_generator_state_dict_keys(state_dict)
        missing, unexpected = self.generator.load_state_dict(state_dict, strict=False)
        print(f"[Checkpoint] loaded from {checkpoint_path} (key='{source}')")
        print(f"[Checkpoint] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    @torch.no_grad()
    def _encode_prompts_for_k(self, prompts_per_turn: List[List[str]], k: int) -> torch.Tensor:
        """编码 batch 中前 k 个 prompt -> [B, k*L, D]，与训练 _encode_prompts_for_k 一致。"""
        B = len(prompts_per_turn)
        flat: List[str] = []
        for b in range(B):
            ps = prompts_per_turn[b][:k]
            if len(ps) < k:
                ps = ps + [ps[-1]] * (k - len(ps))
            flat.extend(ps)
        td = self.text_encoder(flat)
        flat_embeds = (td["prompt_embeds"] if isinstance(td, dict) else td).to(
            device=self.device, dtype=self.generator_dtype
        )
        _, seq_len, hidden_dim = flat_embeds.shape
        return flat_embeds.view(B, k, seq_len, hidden_dim).reshape(B, k * seq_len, hidden_dim)

    @torch.no_grad()
    def _encode_single_prompt_at(
        self,
        prompts_per_turn: List[List[str]],
        turn_index: int,
    ) -> torch.Tensor:
        """编码 batch 中每个样本的「第 turn_index 个 prompt（0-based）」一条 -> [B, L, D]。

        与训练 trainer/multi_turn_dmd.py::_encode_single_prompt_at 完全对齐：
        模型在 cross_attn_current_prompt_only=True 下，每帧只看属于自己那一条 prompt，
        训练 rollout 与推理 rollout 必须保持「单条当前 prompt」的一致编码。
        """
        B = len(prompts_per_turn)
        flat: List[str] = []
        for sample in prompts_per_turn:
            if turn_index < len(sample):
                flat.append(sample[turn_index])
            elif len(sample) > 0:
                flat.append(sample[-1])
            else:
                flat.append("")
        td = self.text_encoder(flat)
        flat_embeds = (td["prompt_embeds"] if isinstance(td, dict) else td).to(
            device=self.device, dtype=self.generator_dtype
        )
        return flat_embeds  # [B, L, D]，视为 num_prompts=1

    @torch.no_grad()
    def _encode_single_uncond(self, batch_size: int) -> torch.Tensor:
        flat = [self.negative_prompt] * batch_size
        td = self.text_encoder(flat)
        return (td["prompt_embeds"] if isinstance(td, dict) else td).to(
            device=self.device, dtype=self.generator_dtype
        )

    @torch.no_grad()
    def _encode_uncond_for_k(self, batch_size: int, k: int) -> torch.Tensor:
        flat = [self.negative_prompt] * (batch_size * k)
        td = self.text_encoder(flat)
        flat_embeds = (td["prompt_embeds"] if isinstance(td, dict) else td).to(
            device=self.device, dtype=self.generator_dtype
        )
        _, seq_len, hidden_dim = flat_embeds.shape
        return flat_embeds.view(batch_size, k, seq_len, hidden_dim).reshape(
            batch_size, k * seq_len, hidden_dim
        )

    def _frame_seq_length(self, h_lat: int, w_lat: int) -> int:
        try:
            patch_size = getattr(self.generator.model, "patch_size", (1, 2, 2))
        except Exception:
            patch_size = (1, 2, 2)
        return (h_lat // int(patch_size[1])) * (w_lat // int(patch_size[2]))

    def _init_kv_cache(
        self, batch_size: int, h_lat: int, w_lat: int,
        min_frames: int = 0,
    ) -> List[Dict[str, torch.Tensor]]:
        wrapper_model = self.generator.model
        num_blocks = len(wrapper_model.blocks)
        num_heads = int(wrapper_model.num_heads)
        head_dim = int(wrapper_model.dim // wrapper_model.num_heads)
        local_attn_size = int(getattr(wrapper_model, "local_attn_size", -1))
        fsl = self._frame_seq_length(h_lat, w_lat)
        if local_attn_size != -1:
            base_size = local_attn_size * fsl
        else:
            base_size = 32760
        # 多轮推理时按需扩容：保证至少能容纳 min_frames 帧（首帧+每轮 1 帧）
        if min_frames > 0:
            need_size = int(min_frames) * fsl
            if need_size > base_size:
                base_size = need_size
        kv_cache_size = base_size

        kv_cache: List[Dict[str, torch.Tensor]] = []
        for _ in range(num_blocks):
            kv_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, num_heads, head_dim],
                        dtype=self.generator_dtype, device=self.device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, num_heads, head_dim],
                        dtype=self.generator_dtype, device=self.device,
                    ),
                    "global_end_index": torch.zeros([1], dtype=torch.long, device=self.device),
                    "local_end_index": torch.zeros([1], dtype=torch.long, device=self.device),
                }
            )
        return kv_cache

    @torch.no_grad()
    def infer_multi_turn(
        self,
        initial_frame: torch.Tensor,
        prompts: List[str],
        seed: int,
        max_turn: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        与训练 _self_rollout_history 完全对齐的多轮 4 步推理：
          1. VAE encode 首帧 -> history_latent[:, 0:1]
          2. 用对齐到 turn-1 的 prompt prefill 首帧到 KV cache（t=0）
          3. 对每个 turn t∈[1..N]：
             a. 编码 prompts[0..t]
             b. 采 noise，跑 denoising_step_list 少步去噪 -> clean_t
             c. 把 clean_t 以 t=context_noise(=0) 写回 KV cache
             d. VAE decode clean_t -> pixel
        """
        # 编码首帧 latent
        initial_video = initial_frame.unsqueeze(0).unsqueeze(2).to(
            device=self.device, dtype=self.vae_dtype
        )
        history_latents = self.vae.encode_to_latent(initial_video).to(
            device=self.device, dtype=self.generator_dtype
        )  # [1, 1, C, H, W]

        prompt_list = [str(p) for p in prompts]
        total_turns = len(prompt_list)
        if max_turn is not None:
            total_turns = min(total_turns, int(max_turn))
        if total_turns <= 0:
            return []

        # 与训练保持一致：每个 batch 只有 1 条样本
        prompts_per_turn = [prompt_list[:total_turns]]

        B, _, C, H, W = history_latents.shape
        fsl = self._frame_seq_length(H, W)

        # 初始化 KV cache（cond 一份；如果开 generator CFG 再加一份 uncond）
        # 至少能放下 首帧(prefill) + 每轮 1 帧 = total_turns + 1 帧
        use_cfg = self.guidance_scale > 1.0
        min_frames = int(total_turns) + 1
        kv_cache = self._init_kv_cache(B, H, W, min_frames=min_frames)
        kv_cache_uncond = (
            self._init_kv_cache(B, H, W, min_frames=min_frames) if use_cfg else None
        )

        # === 首帧 prefill (t=0)，与 _self_rollout_history 一致 ===
        # 与 pipeline/multi_turn_inference.py 一致：首帧 prefill 用第 1 轮 prompt
        # （即 prompts[0]）作为 cross-attn 上下文，单条 prompt。
        first_prompt = self._encode_single_prompt_at(prompts_per_turn, turn_index=0)
        cond_first = {"prompt_embeds": first_prompt}
        first_ts = torch.zeros([B, 1], device=self.device, dtype=torch.long)
        self.generator(
            noisy_image_or_video=history_latents[:, 0:1],
            conditional_dict=cond_first,
            timestep=first_ts,
            kv_cache=kv_cache,
            current_start=0,
            cache_start=0,
        )
        if use_cfg:
            uncond_first = {"prompt_embeds": self._encode_single_uncond(B)}
            self.generator(
                noisy_image_or_video=history_latents[:, 0:1],
                conditional_dict=uncond_first,
                timestep=first_ts,
                kv_cache=kv_cache_uncond,
                current_start=0,
                cache_start=0,
            )

        pred_pixels: List[torch.Tensor] = []

        for turn_idx in range(1, total_turns + 1):
            # turn_idx == 1 对应 prompts[0]，与训练 _self_rollout_history 帧索引对齐：
            # frame index = turn_idx，cross-attn prompt = prompts[turn_idx - 1]，单条。
            cur_prompt = self._encode_single_prompt_at(
                prompts_per_turn, turn_index=turn_idx - 1
            )
            cond_t = {"prompt_embeds": cur_prompt}
            uncond_t = (
                {"prompt_embeds": self._encode_single_uncond(B)} if use_cfg else None
            )
            current_start = turn_idx * fsl

            # 初始化 noise
            noise_gen = torch.Generator(device=self.device)
            noise_gen.manual_seed(seed + turn_idx * 7919)
            noisy_t = torch.randn(
                [B, 1, C, H, W],
                device=self.device,
                dtype=self.generator_dtype,
                generator=noise_gen,
            )

            # === 少步去噪（与训练 _self_rollout_history 内层循环完全一致） ===
            clean_t = noisy_t  # placeholder
            for step_idx, ts_val in enumerate(self.denoising_step_list):
                ts_tensor = torch.full(
                    [B, 1], int(ts_val.item()), device=self.device, dtype=torch.long
                )

                _, pred_x0 = self.generator(
                    noisy_image_or_video=noisy_t,
                    conditional_dict=cond_t,
                    timestep=ts_tensor,
                    kv_cache=kv_cache,
                    current_start=current_start,
                    cache_start=current_start,
                )

                if use_cfg and uncond_t is not None and kv_cache_uncond is not None:
                    _, pred_x0_uncond = self.generator(
                        noisy_image_or_video=noisy_t,
                        conditional_dict=uncond_t,
                        timestep=ts_tensor,
                        kv_cache=kv_cache_uncond,
                        current_start=current_start,
                        cache_start=current_start,
                    )
                    pred_x0 = pred_x0_uncond + self.guidance_scale * (pred_x0 - pred_x0_uncond)

                if step_idx < len(self.denoising_step_list) - 1:
                    next_ts = self.denoising_step_list[step_idx + 1]
                    noisy_t = self.scheduler.add_noise(
                        pred_x0.flatten(0, 1),
                        torch.randn_like(pred_x0.flatten(0, 1)),
                        next_ts * torch.ones([B], device=self.device, dtype=torch.long),
                    ).unflatten(0, (B, 1)).to(dtype=self.generator_dtype)
                else:
                    clean_t = pred_x0  # [B, 1, C, H, W]

            # === 把 clean_t 写回 KV cache (t=context_noise)，下一轮就能看到这帧 ===
            ctx_ts = torch.full(
                [B, 1], self.context_noise, device=self.device, dtype=torch.long
            )
            self.generator(
                noisy_image_or_video=clean_t,
                conditional_dict=cond_t,
                timestep=ctx_ts,
                kv_cache=kv_cache,
                current_start=current_start,
                cache_start=current_start,
            )
            if use_cfg and uncond_t is not None and kv_cache_uncond is not None:
                self.generator(
                    noisy_image_or_video=clean_t,
                    conditional_dict=uncond_t,
                    timestep=ctx_ts,
                    kv_cache=kv_cache_uncond,
                    current_start=current_start,
                    cache_start=current_start,
                )

            # 解码到像素
            pred_pixel = self.vae.decode_to_pixel(clean_t.to(dtype=self.vae_dtype))
            pred_pixel = (pred_pixel * 0.5 + 0.5).clamp(0, 1)
            pred_pixels.append(pred_pixel[:, 0].detach().cpu())

            # 把 clean_t 加入 history（仅用于 record，不影响 KV cache）
            history_latents = torch.cat([history_latents, clean_t], dim=1)

        return pred_pixels


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn DMD 4-step distilled inference")
    parser.add_argument("--config", type=str, required=True,
                        help="DMD distill config (e.g. multi_turn_dmd_distill_config.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="DMD ckpt model.pt (含 generator / generator_ema)")
    parser.add_argument("--validation_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_turn", type=int, default=None)
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"])

    # 默认从 config 读，命令行可覆盖
    parser.add_argument("--denoising_step_list", type=str, default=None,
                        help="逗号分隔，e.g. '1000,750,500,250'。默认从 config 读")
    parser.add_argument("--timeshift", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="generator CFG，DMD 蒸馏后默认 1.0（不开 CFG）")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--context_noise", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true", help="加载 generator_ema 权重")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    is_distributed, rank, world_size, _, device = init_distributed_env(args.device)
    set_seed(args.seed + rank)
    config = load_config(args.config)

    # 解析 denoising_step_list
    if args.denoising_step_list is not None:
        dsl = [int(x.strip()) for x in args.denoising_step_list.split(",") if x.strip()]
    else:
        dsl_cfg = cfg_get(config, "denoising_step_list", [1000, 750, 500, 250])
        dsl = [int(x) for x in (dsl_cfg if isinstance(dsl_cfg, (list, ListConfig)) else [1000, 750, 500, 250])]

    timeshift = float(args.timeshift) if args.timeshift is not None else float(
        cfg_get(config, "timestep_shift", cfg_get(config, "model_kwargs.timestep_shift", 1.0))
    )
    negative_prompt = args.negative_prompt if args.negative_prompt is not None else str(
        cfg_get(config, "negative_prompt", "")
    )
    context_noise = int(args.context_noise) if args.context_noise is not None else int(
        cfg_get(config, "context_noise", 0)
    )

    # 解析 image_size / hw_buckets
    raw_image_size = cfg_get(config, "data.image_size", None)
    if not (isinstance(raw_image_size, (ListConfig, list, tuple)) and len(raw_image_size) == 2):
        raise ValueError(f"invalid data.image_size: {raw_image_size}")
    image_size = (int(raw_image_size[0]), int(raw_image_size[1]))

    bucket_step_width = int(cfg_get(config, "data.bucket_step_width", 64))
    bucket_step_height = int(cfg_get(config, "data.bucket_step_height", 64))
    bucket_max_ratio = float(cfg_get(config, "data.bucket_max_ratio", 4.0))
    hw_buckets = _build_hw_bucket_list(
        image_size=image_size,
        step_width=bucket_step_width,
        step_height=bucket_step_height,
        max_ratio=bucket_max_ratio,
    )

    if rank == 0:
        print(f"[DMD] denoising_step_list={dsl}, timeshift={timeshift}, "
              f"guidance={args.guidance_scale}, context_noise={context_noise}, "
              f"image_size={image_size}, num_buckets={len(hw_buckets)}")

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier_if_needed(is_distributed)

    pipeline = MultiTurnDMDInferencePipeline(
        config=config,
        device=device,
        denoising_step_list=dsl,
        guidance_scale=args.guidance_scale,
        negative_prompt=negative_prompt,
        timeshift=timeshift,
        context_noise=context_noise,
    )
    pipeline.load_checkpoint(args.checkpoint, use_ema=args.use_ema)

    samples = collect_samples(Path(args.validation_root), limit=args.max_samples)
    if rank == 0:
        print(f"[Data] loaded {len(samples)} samples")

    assigned = list(range(rank, len(samples), world_size))
    local_items = [(i, samples[i]) for i in assigned]
    shard_path = output_dir / f"results_rank{rank:02d}.jsonl"

    with open(shard_path, "w", encoding="utf-8") as fout:
        for global_idx, sample in tqdm(local_items, desc=f"DMD rank{rank}", disable=(rank != 0)):
            uuid = sample["uuid"]
            safe_uuid = sanitize_name(uuid) or f"sample_{global_idx:04d}"
            try:
                prompts_all = read_prompts_from_json(sample["json_path"], lang=args.lang)
            except Exception as e:
                print(f"[Skip][rank{rank}] {sample['json_path']}: {e}")
                continue
            if len(prompts_all) == 0:
                continue

            num_turns = len(prompts_all)
            if args.max_turn is not None:
                num_turns = min(num_turns, int(args.max_turn))
            if num_turns <= 0:
                continue
            used_prompts = prompts_all[:num_turns]

            with Image.open(sample["source_path"]) as src_img:
                orig_w, orig_h = src_img.size
            target_h, target_w = _pick_hw_bucket(orig_h, orig_w, hw_buckets)
            initial_frame = load_image_as_model_input(
                sample["source_path"], target_h=target_h, target_w=target_w
            )

            pred_frames = pipeline.infer_multi_turn(
                initial_frame=initial_frame,
                prompts=used_prompts,
                seed=args.seed + global_idx * 1000,
                max_turn=num_turns,
            )
            if len(pred_frames) == 0:
                continue

            sample_dir = output_dir / safe_uuid
            sample_dir.mkdir(parents=True, exist_ok=True)
            source_out = sample_dir / "source.png"
            save_tensor_as_png((initial_frame * 0.5 + 0.5).clamp(0, 1), str(source_out))

            prompts_out = sample_dir / "prompts.json"
            with open(prompts_out, "w", encoding="utf-8") as pf:
                json.dump(
                    {"lang": args.lang, "instructions": used_prompts},
                    pf, ensure_ascii=False, indent=2,
                )

            pred_paths: List[str] = []
            gt_paths: List[str] = []
            for turn_idx in range(num_turns):
                pred_tensor = pred_frames[turn_idx][0].clamp(0, 1)
                pred_out = sample_dir / f"pred_turn_{turn_idx + 1:02d}.png"
                save_tensor_as_png(pred_tensor, str(pred_out))
                pred_paths.append(str(pred_out))

                if turn_idx < len(sample["gt_paths"]):
                    gt_src = sample["gt_paths"][turn_idx]
                    gt_tensor = load_image_as_model_input(gt_src, target_h=target_h, target_w=target_w)
                    gt_save = (gt_tensor * 0.5 + 0.5).clamp(0, 1)
                    gt_out = sample_dir / f"gt_turn_{turn_idx + 1:02d}.png"
                    save_tensor_as_png(gt_save, str(gt_out))
                    gt_paths.append(str(gt_out))

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
                "denoising_step_list": dsl,
                "timeshift": timeshift,
                "guidance_scale": args.guidance_scale,
                "use_ema": bool(args.use_ema),
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