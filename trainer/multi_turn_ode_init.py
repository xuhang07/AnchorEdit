"""
Multi-Turn Causal I2V ODE-Init Trainer.

目标：在走 DMD 蒸馏之前，先用 ODE-regression loss 预训练一个"会走少步路径"的
generator，作为后续 [`MultiTurnDMDTrainer`](trainer/multi_turn_dmd.py:1) 阶段
generator / fake_score 的初始化（即 CausVid 论文里 Sec 4.3 所说的 ODE init）。

为什么需要 ODE init？
  - 原始多轮 causal I2V 模型是按 1000 步连续 timestep 训练的；直接拿去做 DMD
    蒸馏时 generator 单步/少步路径完全没见过，容易 KL 爆炸、训练发散。
  - 先用 `denoising_step_list`（e.g. [1000, 750, 500, 250]）作为离散 timestep
    集合，让模型学会在这些整数 timestep 上做 x0 预测；之后接入 DMD 就稳定得多。

与 [`MultiTurnDMDTrainer`](trainer/multi_turn_dmd.py:1) 的差异：
  - 只训 generator，不需要 real_score / fake_score / CFG。
  - 每个 micro-step：
      1) 取多轮 batch -> [B, T, C, H, W]，T = 1 + N_turns
      2) 同步采样目标 turn k ∈ [1, T-1]
      3) no_grad self-rollout 出 history_pred[:, 1:k]，并把 [0:k] prefill 到
         generator 的 KV cache（与真实 AR 推理路径一致）
      4) 在第 k 帧随机采一个 denoising step idx s，做 backward simulation
         拿到 noisy_input_k，单步 forward generator -> pred_x0_k（带梯度）
      5) loss = MSE(pred_x0_k, GT clean_k)，仅在 timestep > 0 的帧上计算

与 [`ODERegression`](model/ode_regression.py:9) 的差异：
  - 原版 ODE regression 需要一份「预先存好的 ODE 轨迹」LMDB 数据；多轮场景里
    生成这种配对数据开销巨大，这里改成「在线用 GT clean_k 现场 backward
    simulation 构造 noisy_k」，训练信号完全等价：两者都是让 generator 在
    denoising_step_list 的若干 timestep 上还原出 clean_k。
"""

import gc
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from utils.misc import cycle
from utils.distributed import EMA_FSDP, fsdp_state_dict, fsdp_wrap, launch_distributed_job
from utils.misc import set_seed
from utils.multi_turn_dataset import create_multi_turn_dataloader
from utils.multi_turn_tokenizer import MultiTurnTokenizer
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def _strip_ckpt_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """剥离 FSDP / compile / DDP 包装产生的 key 前缀。"""
    prefixes = ["_orig_mod.", "_checkpoint_wrapped_module.", "module."]
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        out[nk] = v
    return out


def _load_generator_ckpt(
    module: WanDiffusionWrapper,
    ckpt_path: str,
    tag: str = "module",
    rank0_only: bool = False,
    prefer_ema: bool = False,
) -> None:
    """加载多轮 causal 模型 ckpt 到 WanDiffusionWrapper。

    Ckpt 解析优先级：
        - `prefer_ema=True` 时，优先取 `generator_ema`，找不到再回落到 `generator` / `model`
          (上游 [`save_checkpoint()`](trainer/multi_turn_i2v.py:748) 同时保存 EMA 与实时权重，
           推理脚本 [`inference_multi_turn_i2v.py`](inference_multi_turn_i2v.py:1) 也用 EMA)。
        - `prefer_ema=False` 时，按原顺序：generator -> model -> raw。

    Args:
        rank0_only: 仅 rank0 加载 ckpt，其他 rank 由 FSDP `sync_module_states` 广播。
        prefer_ema: 是否优先使用 EMA 权重。EMA 在多轮训练中已被证明
            是「实际推理路径」，下游蒸馏（DMD / ODE init）以此为起点更稳。
    """
    if rank0_only and dist.is_initialized() and dist.get_rank() != 0:
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = None
    used_key = None
    if isinstance(ckpt, dict):
        # 多轮训练 ckpt 结构：{"generator": ..., "generator_ema": ..., "step": ...}
        # ema 优先级仅在 prefer_ema=True 时启用
        candidate_keys = (
            ["generator_ema", "generator", "model"]
            if prefer_ema else
            ["generator", "model"]
        )
        for k in candidate_keys:
            if k in ckpt and ckpt[k] is not None:
                sd = ckpt[k]
                used_key = k
                break
        if sd is None:
            sd = ckpt
            used_key = "raw_dict"
    else:
        sd = ckpt
        used_key = "raw_tensor"

    sd = _strip_ckpt_prefixes(sd)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    print(f"[Checkpoint][{tag}] loaded from {ckpt_path} (key={used_key}, prefer_ema={prefer_ema}), "
          f"missing={len(missing)}, unexpected={len(unexpected)}")

    del ckpt
    del sd
    gc.collect()


class MultiTurnODEInitTrainer:
    """多轮 causal AR ODE-init 预训练器（为 DMD 蒸馏做准备）。"""

    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # ----- 分布式 -----
        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = (global_rank == 0)
        self.disable_wandb = bool(getattr(config, "disable_wandb", True))
        self.enable_tensorboard = bool(getattr(config, "enable_tensorboard", True))

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)

        # ----- Tensorboard -----
        if self.is_main_process and self.enable_tensorboard:
            os.makedirs(config.logdir, exist_ok=True)
            os.makedirs(config.tensorboard_logdir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=config.tensorboard_logdir)
            cfg_dict = OmegaConf.to_container(config, resolve=True)
            self.writer.add_text("config", json.dumps(cfg_dict, indent=2, default=str), 0)
        else:
            self.writer = None

        self.output_path = config.logdir

        # ----- 超参 -----
        self.num_train_timestep = int(config.num_train_timestep)
        self.timestep_shift = float(getattr(config, "timestep_shift", 1.0))
        self.context_noise = int(getattr(config, "context_noise", 0))
        self.gradient_accumulation_steps = max(
            1, int(getattr(config, "gradient_accumulation_steps", 1))
        )
        self._accum_step_idx = 0

        # ----- 少步 schedule -----
        self.denoising_step_list = torch.tensor(
            list(config.denoising_step_list), dtype=torch.long, device=self.device
        )
        # 若每帧独立采 idx，[B, T] timesteps 分布更均匀；但 self-rollout 期间所有
        # 位置共用同一份 denoising_step_list，这里只在 k 位置用它采一个 idx。

        # ----- 模型 / 优化器 / 数据 -----
        self._setup_models()
        self._setup_optimizer()
        self._setup_dataloader()

        self.num_frame_per_block = int(getattr(config, "num_frame_per_block", 1))
        self.max_grad_norm_generator = float(getattr(config, "max_grad_norm_generator", 10.0))

        # ----- EMA -----
        ema_weight = float(getattr(config, "ema_weight", 0.0) or 0.0)
        self.generator_ema = None
        if ema_weight > 0.0:
            if self.is_main_process:
                print(f"[EMA] enable with decay={ema_weight}")
            self.generator_ema = EMA_FSDP(self.generator, decay=ema_weight)

        self.previous_time = None

        if self.is_main_process:
            print(f"[Init] denoising_step_list={self.denoising_step_list.tolist()}, "
                  f"grad_accum={self.gradient_accumulation_steps}, "
                  f"world_size={self.world_size}")

    # ------------------------------------------------------------------
    # 模型 / 优化器 / 数据
    # ------------------------------------------------------------------
    def _setup_models(self) -> None:
        cfg = self.config
        model_kwargs = OmegaConf.to_container(cfg.model_kwargs, resolve=True)

        # 只有 generator（要训练 + 少步推理）
        self.generator = WanDiffusionWrapper(**model_kwargs)
        self.generator.model.requires_grad_(True)
        if bool(getattr(cfg, "gradient_checkpointing", False)):
            self.generator.model.gradient_checkpointing = True

        # text encoder / vae
        self.text_encoder = WanTextEncoder(model_kwargs["model_name"])
        self.text_encoder.requires_grad_(False)
        self.vae = WanVAEWrapper(model_kwargs["model_name"])
        self.vae.requires_grad_(False)

        # scheduler 与 DMD 训练对齐到 1000 点网格
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.set_timesteps(num_inference_steps=self.num_train_timestep, training=True)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

        # ---- 加载 ckpt ----
        # prefer_ema=True 时优先取 ckpt["generator_ema"]，与推理路径
        # ([`inference_multi_turn_i2v.py`](inference_multi_turn_i2v.py:1)) 完全一致。
        # `generator_ckpt`（resume 训练用）默认仍取实时权重，因为 resume 期望
        # 接续训练状态而非取一个被平滑过的快照。
        ckpt_cfg = getattr(cfg, "checkpoint", None)
        rank0_only_load = bool(getattr(ckpt_cfg, "rank0_only_load", True)) if ckpt_cfg is not None else True
        prefer_ema = bool(getattr(ckpt_cfg, "prefer_ema", True)) if ckpt_cfg is not None else True
        pretrained_pt = getattr(ckpt_cfg, "pretrained_pt", None) if ckpt_cfg is not None else None
        if pretrained_pt:
            _load_generator_ckpt(
                self.generator, pretrained_pt,
                tag="generator(init)", rank0_only=rank0_only_load,
                prefer_ema=prefer_ema,
            )
        if ckpt_cfg is not None:
            gen_ckpt = getattr(ckpt_cfg, "generator_ckpt", None)
            if gen_ckpt:
                _load_generator_ckpt(
                    self.generator, gen_ckpt,
                    tag="generator(override)", rank0_only=rank0_only_load,
                    prefer_ema=False,  # resume 强制走实时权重
                )

        if dist.is_initialized():
            dist.barrier()
        gc.collect()

        # ---- FSDP wrap ----
        gen_offload = bool(getattr(cfg, "generator_cpu_offload", False))
        sync_states = rank0_only_load

        self.generator = fsdp_wrap(
            self.generator,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.generator_fsdp_wrap_strategy,
            cpu_offload=gen_offload,
            sync_module_states=sync_states,
        )
        self.text_encoder = fsdp_wrap(
            self.text_encoder,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.text_encoder_fsdp_wrap_strategy,
            cpu_offload=bool(getattr(cfg, "text_encoder_cpu_offload", True)),
        )

        if self.is_main_process:
            print(f"[Init][Offload] generator={gen_offload}, "
                  f"text_encoder={bool(getattr(cfg, 'text_encoder_cpu_offload', True))}",
                  flush=True)

        # VAE
        self.vae_cpu_offload = bool(getattr(cfg, "vae_cpu_offload", False))
        vae_dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32
        if self.vae_cpu_offload:
            self.vae = self.vae.to(device="cpu", dtype=vae_dtype)
            if self.is_main_process:
                print("[Init][Offload] vae=True (kept on CPU, moved to GPU on demand)", flush=True)
        else:
            self.vae = self.vae.to(device=self.device, dtype=vae_dtype)

        self.tokenizer = MultiTurnTokenizer(
            text_encoder=self.text_encoder,
            vae=self.vae,
            dtype=self.dtype,
            device=self.device,
            config=self.config,
        )

    def _setup_optimizer(self) -> None:
        cfg = self.config
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.generator.parameters() if p.requires_grad],
            lr=float(cfg.lr),
            betas=(float(getattr(cfg, "beta1", 0.0)), float(getattr(cfg, "beta2", 0.999))),
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )

    def _setup_dataloader(self) -> None:
        cfg = self.config
        require_exact = bool(getattr(cfg.data, "require_exact_turns", True))
        dataloader = create_multi_turn_dataloader(
            data_path=cfg.data.data_path,
            batch_size=int(cfg.batch_size),
            num_workers=int(getattr(cfg, "num_workers", 1)),
            num_frame_per_block=int(getattr(cfg, "num_frame_per_block", 1)),
            max_turns=getattr(cfg.data, "max_turns", None),
            shuffle=True,
            image_size=tuple(cfg.data.image_size),
            bucket_step_width=int(getattr(cfg.data, "bucket_step_width", 64)),
            bucket_step_height=int(getattr(cfg.data, "bucket_step_height", 64)),
            bucket_max_ratio=float(getattr(cfg.data, "bucket_max_ratio", 4.0)),
            require_exact_turns=require_exact,
        )
        self.dataloader = cycle(dataloader)

    # ------------------------------------------------------------------
    # 数据预处理
    # ------------------------------------------------------------------
    def _prepare_batch(self, batch: Dict) -> Dict:
        if self.vae_cpu_offload:
            self.vae = self.vae.to(self.device)
        try:
            processed = self.tokenizer(batch)
        finally:
            if self.vae_cpu_offload:
                self.vae = self.vae.to("cpu")
                torch.cuda.empty_cache()
        image_latents = processed["image_latents"]
        prompt_embeds_full = processed["prompt_embeds"]
        prompts_per_turn = batch["prompts"]
        return {
            "image_latents": image_latents,
            "prompt_embeds_full": prompt_embeds_full,
            "prompts_per_turn": prompts_per_turn,
        }

    @torch.no_grad()
    def _encode_single_prompt_at(
        self,
        prompts_per_turn: List[List[str]],
        turn_index: int,
    ) -> torch.Tensor:
        """编码每个样本的第 turn_index 个 prompt -> [B, L, D]。

        与 [`_encode_single_prompt_at()`](trainer/multi_turn_dmd.py:457) 保持一致：
        cross_attn_current_prompt_only=True 下，rollout / 单步 forward 每帧只看
        自己那一条 prompt。
        """
        batch_size = len(prompts_per_turn)
        flat: List[str] = []
        for sample in prompts_per_turn:
            if turn_index < len(sample):
                flat.append(sample[turn_index])
            elif len(sample) > 0:
                flat.append(sample[-1])
            else:
                flat.append("")
        td = self.text_encoder(flat)
        if isinstance(td, dict):
            flat_embeds = td["prompt_embeds"]
        else:
            flat_embeds = td
        return flat_embeds.to(device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # KV cache 工具（与 DMD trainer 复用同套逻辑）
    # ------------------------------------------------------------------
    def _frame_seq_length(self, h_lat: int, w_lat: int) -> int:
        try:
            patch_size = getattr(self.generator.model, "patch_size", (1, 2, 2))
        except Exception:
            patch_size = (1, 2, 2)
        return (h_lat // int(patch_size[1])) * (w_lat // int(patch_size[2]))

    def _init_kv_cache(self, batch_size: int, h_lat: int, w_lat: int) -> List[Dict[str, torch.Tensor]]:
        wrapper_model = self.generator.model
        num_blocks = len(wrapper_model.blocks)
        num_heads = int(wrapper_model.num_heads)
        head_dim = int(wrapper_model.dim // wrapper_model.num_heads)
        local_attn_size = int(getattr(wrapper_model, "local_attn_size", -1))
        frame_seq_length = self._frame_seq_length(h_lat, w_lat)
        kv_cache_size = (
            local_attn_size * frame_seq_length if local_attn_size != -1 else 32760
        )
        kv_cache: List[Dict[str, torch.Tensor]] = []
        for _ in range(num_blocks):
            kv_cache.append({
                "k": torch.zeros(
                    [batch_size, kv_cache_size, num_heads, head_dim],
                    dtype=self.dtype, device=self.device),
                "v": torch.zeros(
                    [batch_size, kv_cache_size, num_heads, head_dim],
                    dtype=self.dtype, device=self.device),
                "global_end_index": torch.zeros([1], dtype=torch.long, device=self.device),
                "local_end_index": torch.zeros([1], dtype=torch.long, device=self.device),
            })
        return kv_cache

    # ------------------------------------------------------------------
    # Self-Rollout：用 generator 自回归生成 history（no_grad）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _self_rollout_history(
        self,
        image_latents: torch.Tensor,
        prompts_per_turn: List[List[str]],
        target_turn_k: int,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """用 generator 自回归生成 turn 1..k-1 的 latent（模拟真实推理路径）。

        与 [`_self_rollout_history()`](trainer/multi_turn_dmd.py:565) 语义一致。
        """
        B, T, C, H, W = image_latents.shape
        fsl = self._frame_seq_length(H, W)
        kv_cache = self._init_kv_cache(B, H, W)

        # 首帧 GT prefill
        first_frame = image_latents[:, 0:1].to(dtype=self.dtype)
        first_prompt = self._encode_single_prompt_at(prompts_per_turn, turn_index=0)
        cond_first = {"prompt_embeds": first_prompt}
        self.generator(
            noisy_image_or_video=first_frame,
            conditional_dict=cond_first,
            timestep=torch.zeros([B, 1], device=self.device, dtype=torch.long),
            kv_cache=kv_cache,
            current_start=0,
            cache_start=0,
        )

        history_frames = [first_frame.squeeze(1)]

        for t_idx in range(1, target_turn_k):
            cur_prompt = self._encode_single_prompt_at(prompts_per_turn, turn_index=t_idx - 1)
            cond_t = {"prompt_embeds": cur_prompt}
            current_start = t_idx * fsl

            noisy_t = torch.randn([B, 1, C, H, W], device=self.device, dtype=self.dtype)

            for step_idx, ts_val in enumerate(self.denoising_step_list):
                ts_tensor = torch.full([B, 1], int(ts_val.item()), device=self.device, dtype=torch.long)
                _, pred_x0 = self.generator(
                    noisy_image_or_video=noisy_t,
                    conditional_dict=cond_t,
                    timestep=ts_tensor,
                    kv_cache=kv_cache,
                    current_start=current_start,
                    cache_start=current_start,
                )
                if step_idx < len(self.denoising_step_list) - 1:
                    next_ts = self.denoising_step_list[step_idx + 1]
                    noisy_t = self.scheduler.add_noise(
                        pred_x0.flatten(0, 1),
                        torch.randn_like(pred_x0.flatten(0, 1)),
                        next_ts * torch.ones([B], device=self.device, dtype=torch.long),
                    ).unflatten(0, (B, 1))
                else:
                    clean_t = pred_x0

            ctx_ts = torch.full([B, 1], self.context_noise, device=self.device, dtype=torch.long)
            self.generator(
                noisy_image_or_video=clean_t,
                conditional_dict=cond_t,
                timestep=ctx_ts,
                kv_cache=kv_cache,
                current_start=current_start,
                cache_start=current_start,
            )

            history_frames.append(clean_t.squeeze(1))

        history_pred = torch.stack(history_frames, dim=1)
        return history_pred, kv_cache

    # ------------------------------------------------------------------
    # Generator 单步 forward（带梯度） + ODE regression loss
    # ------------------------------------------------------------------
    def _generator_step_at_k(
        self,
        image_latents: torch.Tensor,
        prompts_per_turn: List[List[str]],
        target_turn_k: int,
        kv_cache: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict]:
        """在第 k turn 位置做 backward simulation + 单步 forward（带梯度）。

        返回 (loss, log_dict)；loss = MSE(pred_x0_k, GT clean_k)，仅在
        denoised_from > 0 时计入（与 [`generator_loss()`](model/ode_regression.py:98)
        中 `mask = timestep != 0` 一致）。
        """
        B, T, C, H, W = image_latents.shape
        fsl = self._frame_seq_length(H, W)
        current_start = target_turn_k * fsl

        cur_prompt = self._encode_single_prompt_at(prompts_per_turn, turn_index=target_turn_k - 1)
        cond_k = {"prompt_embeds": cur_prompt}

        # GT clean latent（目标）
        clean_k = image_latents[:, target_turn_k:target_turn_k + 1].to(dtype=self.dtype)

        # 跨 rank 同步采样 denoising step idx
        step_idx_t = torch.randint(0, len(self.denoising_step_list), (1,), device=self.device)
        if dist.is_initialized():
            dist.broadcast(step_idx_t, src=0)
        step_idx = int(step_idx_t.item())

        ts_val = self.denoising_step_list[step_idx]
        denoised_from = int(ts_val.item())

        # Backward simulation：给 clean_k 加噪到 denoised_from
        noise_k = torch.randn_like(clean_k)
        if denoised_from > 0:
            ts_tensor = torch.full([B], denoised_from, device=self.device, dtype=torch.long)
            noisy_k = self.scheduler.add_noise(
                clean_k.flatten(0, 1),
                noise_k.flatten(0, 1),
                ts_tensor,
            ).unflatten(0, (B, 1))
        else:
            noisy_k = clean_k

        # 单步 forward（带梯度）
        ts_input = torch.full([B, 1], denoised_from, device=self.device, dtype=torch.long)
        _, pred_x0_k = self.generator(
            noisy_image_or_video=noisy_k,
            conditional_dict=cond_k,
            timestep=ts_input,
            kv_cache=kv_cache,
            current_start=current_start,
            cache_start=current_start,
        )

        # ODE regression loss：MSE 到 GT clean_k
        if denoised_from > 0:
            loss = F.mse_loss(pred_x0_k, clean_k.detach(), reduction="mean")
        else:
            # t=0 时 pred = noisy = clean，loss 恒 0，跳过
            loss = torch.zeros([], device=self.device, dtype=self.dtype, requires_grad=True)

        log_dict = {
            "ode_loss": loss.detach().item(),
            "denoised_from": denoised_from,
            "target_turn_k": target_turn_k,
        }
        return loss, log_dict

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------
    def _sample_target_turn_k(self, num_turns_in_batch: int) -> int:
        """跨 rank 同步采样 k ∈ [1, T-1]（与 DMD trainer 一致）。"""
        T = int(num_turns_in_batch)
        if T <= 1:
            return 1
        k_t = torch.randint(1, T, (1,), device=self.device, dtype=torch.long)
        if dist.is_initialized():
            dist.broadcast(k_t, src=0)
        return int(k_t.item())

    def generator_step(
        self,
        image_latents: torch.Tensor,
        prompts_per_turn: List[List[str]],
        target_turn_k: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """一次 generator ODE-regression 训练 step。"""
        B, T, C, H, W = image_latents.shape
        if T <= 1:
            zero = torch.zeros([], device=self.device, dtype=self.dtype, requires_grad=True)
            return zero, {"ode_loss": 0.0, "skipped": 1, "target_turn_k": 0}
        target_turn_k = int(max(1, min(target_turn_k, T - 1)))

        # Step 1: self-rollout history
        history_pred, kv_cache = self._self_rollout_history(
            image_latents=image_latents,
            prompts_per_turn=prompts_per_turn,
            target_turn_k=target_turn_k,
        )

        # 防御
        if history_pred.dim() == 4:
            history_pred = history_pred.unsqueeze(0)

        # Step 2: generator 单步 forward at k + ODE loss
        loss, log_dict = self._generator_step_at_k(
            image_latents=image_latents,
            prompts_per_turn=prompts_per_turn,
            target_turn_k=target_turn_k,
            kv_cache=kv_cache,
        )

        del kv_cache
        return loss, log_dict

    def fwd_bwd_one_step(self, prepared_batch: Dict) -> Dict:
        """Forward + Backward 一次。"""
        image_latents = prepared_batch["image_latents"]
        prompts_per_turn = prepared_batch["prompts_per_turn"]
        T = image_latents.shape[1]

        target_turn_k = self._sample_target_turn_k(T)

        self.generator.train()
        loss, log_dict = self.generator_step(
            image_latents=image_latents,
            prompts_per_turn=prompts_per_turn,
            target_turn_k=target_turn_k,
        )
        loss_scaled = loss / self.gradient_accumulation_steps
        loss_scaled.backward()
        log_dict["ode_loss_raw"] = loss.detach().item()
        return log_dict

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(self) -> None:
        if self.is_main_process:
            print(f"[Save] gathering distributed model states at step {self.step} ...")

        gen_sd = fsdp_state_dict(self.generator)
        state = {"generator": gen_sd, "step": self.step}
        if self.generator_ema is not None:
            state["generator_ema"] = self.generator_ema.state_dict()

        if self.is_main_process:
            ckpt_dir = os.path.join(self.output_path, f"checkpoint_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "model.pt")
            torch.save(state, ckpt_path)
            print(f"[Save] saved to {ckpt_path}")

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def train_one_step(self) -> None:
        """完整一步：支持梯度累计；累计最后一个 micro-step 后再 step 优化器。"""
        cfg = self.config

        # 本次是否为累计周期的最后一步
        is_last_micro = (self._accum_step_idx == self.gradient_accumulation_steps - 1)

        batch = next(self.dataloader)
        prepared = self._prepare_batch(batch)

        log_dict = self.fwd_bwd_one_step(prepared)

        self._accum_step_idx += 1

        if is_last_micro:
            grad_norm = self.generator.clip_grad_norm_(self.max_grad_norm_generator)
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad(set_to_none=True)
            if self.generator_ema is not None:
                self.generator_ema.update(self.generator)
            self._accum_step_idx = 0
        else:
            grad_norm = None

        # logging
        if self.is_main_process and self.writer is not None and (self.step % int(getattr(cfg, "log_iters", 1)) == 0):
            for k, v in log_dict.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"train/{k}", float(v), self.step)
            if grad_norm is not None:
                try:
                    self.writer.add_scalar("train/generator_grad_norm", float(grad_norm), self.step)
                except Exception:
                    pass

        if self.is_main_process and (self.step % int(getattr(cfg, "log_iters", 1)) == 0):
            msg = f"[Step {self.step}] " + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in log_dict.items()
            )
            print(msg, flush=True)

        if self.step > 0 and self.step % int(getattr(cfg, "gc_interval", 100)) == 0:
            gc.collect()
            torch.cuda.empty_cache()

    def train(self) -> None:
        max_steps = int(getattr(self.config, "max_steps", 100000))
        save_iters = int(getattr(self.config, "save_iters", 500))
        no_save = bool(getattr(self.config, "no_save", False))

        while self.step < max_steps:
            self.train_one_step()

            if (not no_save) and self.step > 0 and self.step % save_iters == 0:
                self.save_checkpoint()
                if dist.is_initialized():
                    dist.barrier()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is not None and self.writer is not None:
                    try:
                        self.writer.add_scalar(
                            "time/per_iter_sec", current_time - self.previous_time, self.step
                        )
                    except Exception:
                        pass
                self.previous_time = current_time

            self.step += 1