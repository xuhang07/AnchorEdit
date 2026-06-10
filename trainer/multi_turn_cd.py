"""
Multi-Turn Causal I2V Consistency Distillation Trainer.

动机：
  - ODE-regression init（[`trainer/multi_turn_ode_init.py`](trainer/multi_turn_ode_init.py:1)）
    让 student 在 4 个离散 timestep 上做 x0 回归；但 4 步 teacher 轨迹本身质量
    很差，不足以作为监督信号。
  - 经典 ODE init（[`scripts/generate_ode_pairs.py`](scripts/generate_ode_pairs.py:1)）
    需要离线用 teacher 跑 48 步采样存 LMDB，多轮 + 1024x1024 场景存储 / 算力都吃不消。
  - Consistency Distillation (CD) 直接在**密集** timestep 网格（默认 N=48）上
    蒸馏：teacher 每步只前进一小步（1/N），误差极小；student 自一致性 loss
    让 (x_t, t) 的预测与 (x_{t-dt}, t-dt) 的预测对齐，最终得到一个在任意
    timestep 都能直接预测 x0 的模型——再用任意少步 solver 推理即可。

参考：
  - 单轮实现：[`consistency_model.py`](consistency_model.py:1)
  - 多轮 causal KV cache 路径：[`trainer/multi_turn_dmd.py`](trainer/multi_turn_dmd.py:1)

每个 micro-step 流程：
  1. 取多轮 batch -> [B, T, C, H, W]
  2. 同步采目标 turn k ∈ [1, T-1]
  3. no_grad self-rollout history[1..k-1]，prefill KV cache 到 k-1
  4. 同步采 timestep_idx ∈ [0, N-1)，得到 (t, t_next)
  5. 对 GT clean_k 加噪到 t -> latent_t
  6. Teacher 单步 ODE（带 CFG）：latent_t_next = latent_t - dt * v_pred_teacher
  7. Student forward (带梯度)：_, cm_pred_t = generator(latent_t, t, kv_cache)
  8. EMA student forward (no_grad)：_, cm_pred_t_next = generator_ema(latent_t_next, t_next, kv_cache')
  9. loss = MSE(cm_pred_t, cm_pred_t_next.detach())
  10. backward + update generator + update EMA
"""

import gc
import json
import os
import time
from copy import deepcopy
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
from utils.scheduler import FlowMatchScheduler
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
    prefer_ema: bool = True,
) -> None:
    """加载多轮 causal ckpt。prefer_ema=True 时优先取 generator_ema 权重。"""
    if rank0_only and dist.is_initialized() and dist.get_rank() != 0:
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = None
    used_key = None
    if isinstance(ckpt, dict):
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


class MultiTurnCDTrainer:
    """多轮 causal AR Consistency Distillation 训练器。"""

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
        self.enable_tensorboard = bool(getattr(config, "enable_tensorboard", True))

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)

        if self.is_main_process and self.enable_tensorboard:
            os.makedirs(config.logdir, exist_ok=True)
            os.makedirs(config.tensorboard_logdir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=config.tensorboard_logdir)
            cfg_dict = OmegaConf.to_container(config, resolve=True)
            self.writer.add_text("config", json.dumps(cfg_dict, indent=2, default=str), 0)
        else:
            self.writer = None

        self.output_path = config.logdir

        # ----- CD 超参 -----
        self.discrete_cd_N = int(getattr(config, "discrete_cd_N", 48))
        self.teacher_guidance_scale = float(getattr(config, "teacher_guidance_scale", 6.0))
        self.context_noise = int(getattr(config, "context_noise", 0))
        self.gradient_accumulation_steps = max(
            1, int(getattr(config, "gradient_accumulation_steps", 1))
        )
        self._accum_step_idx = 0

        # student 少步推理用 schedule（只用来构造 KV cache history rollout；训练 loss
        # 不依赖它，而是跑在 discrete_cd_N 的密集网格上）
        self.denoising_step_list = torch.tensor(
            list(getattr(config, "denoising_step_list", [1000, 750, 500, 250])),
            dtype=torch.long, device=self.device,
        )

        # ----- 模型 / 优化器 / 数据 -----
        self._setup_models()
        self._setup_optimizer()
        self._setup_dataloader()

        self.num_frame_per_block = int(getattr(config, "num_frame_per_block", 1))
        self.max_grad_norm_generator = float(getattr(config, "max_grad_norm_generator", 10.0))

        # ----- EMA （CD 必备）-----
        ema_decay = float(getattr(config, "ema_decay", 0.999))
        if self.is_main_process:
            print(f"[EMA] CD requires EMA, decay={ema_decay}")
        self.generator_ema_state = EMA_FSDP(self.generator, decay=ema_decay)

        self.previous_time = None

        if self.is_main_process:
            print(f"[Init] discrete_cd_N={self.discrete_cd_N}, "
                  f"teacher_cfg={self.teacher_guidance_scale}, "
                  f"grad_accum={self.gradient_accumulation_steps}, "
                  f"world_size={self.world_size}")

    # ------------------------------------------------------------------
    # 模型 / 优化器 / 数据
    # ------------------------------------------------------------------
    def _setup_models(self) -> None:
        cfg = self.config
        model_kwargs = OmegaConf.to_container(cfg.model_kwargs, resolve=True)

        # generator（训练目标）
        self.generator = WanDiffusionWrapper(**model_kwargs)
        self.generator.model.requires_grad_(True)
        if bool(getattr(cfg, "gradient_checkpointing", False)):
            self.generator.model.gradient_checkpointing = True

        # generator_ema（student 的 EMA 快照，用于计算 cm_pred_t_next 目标）
        self.generator_ema = WanDiffusionWrapper(**model_kwargs)
        self.generator_ema.model.requires_grad_(False)
        self.generator_ema.model.eval()

        # teacher（冻结，走 1/N ODE 步）
        self.teacher = WanDiffusionWrapper(**model_kwargs)
        self.teacher.model.requires_grad_(False)
        self.teacher.model.eval()

        # text encoder / vae
        self.text_encoder = WanTextEncoder(model_kwargs["model_name"])
        self.text_encoder.requires_grad_(False)
        self.vae = WanVAEWrapper(model_kwargs["model_name"])
        self.vae.requires_grad_(False)

        # ----- CD scheduler：shift=5 的密集 N 网格（独立于 generator 自带 scheduler）-----
        cd_shift = float(getattr(cfg, "cd_scheduler_shift", 5.0))
        self.cd_scheduler = FlowMatchScheduler(shift=cd_shift, sigma_min=0.0, extra_one_step=True)
        self.cd_scheduler.set_timesteps(num_inference_steps=self.discrete_cd_N, denoising_strength=1.0)
        self.cd_scheduler.sigmas = self.cd_scheduler.sigmas.to(self.device)
        self.cd_scheduler.timesteps = self.cd_scheduler.timesteps.to(self.device)

        # student 推理用的 1000 点 scheduler（与 DMD trainer / inference 对齐）
        self.inference_scheduler = self.generator.get_scheduler()
        self.inference_scheduler.set_timesteps(num_inference_steps=1000, training=True)
        self.inference_scheduler.timesteps = self.inference_scheduler.timesteps.to(self.device)

        # ---- 加载 ckpt ----
        ckpt_cfg = getattr(cfg, "checkpoint", None)
        rank0_only_load = bool(getattr(ckpt_cfg, "rank0_only_load", True)) if ckpt_cfg is not None else True
        prefer_ema = bool(getattr(ckpt_cfg, "prefer_ema", True)) if ckpt_cfg is not None else True
        pretrained_pt = getattr(ckpt_cfg, "pretrained_pt", None) if ckpt_cfg is not None else None

        if pretrained_pt:
            # 三份模型共用同一份预训练权重（teacher 冻结、student/EMA 训练起点）
            _load_generator_ckpt(self.generator, pretrained_pt, tag="generator(init)",
                                 rank0_only=rank0_only_load, prefer_ema=prefer_ema)
            _load_generator_ckpt(self.generator_ema, pretrained_pt, tag="generator_ema(init)",
                                 rank0_only=rank0_only_load, prefer_ema=prefer_ema)
            _load_generator_ckpt(self.teacher, pretrained_pt, tag="teacher(frozen)",
                                 rank0_only=rank0_only_load, prefer_ema=prefer_ema)

        if ckpt_cfg is not None:
            gen_ckpt = getattr(ckpt_cfg, "generator_ckpt", None)
            if gen_ckpt:
                _load_generator_ckpt(self.generator, gen_ckpt, tag="generator(override)",
                                     rank0_only=rank0_only_load, prefer_ema=False)

        if dist.is_initialized():
            dist.barrier()
        gc.collect()

        # ---- FSDP wrap ----
        gen_offload = bool(getattr(cfg, "generator_cpu_offload", False))
        ema_offload = bool(getattr(cfg, "generator_ema_cpu_offload", True))
        teacher_offload = bool(getattr(cfg, "teacher_cpu_offload", True))
        sync_states = rank0_only_load

        self.generator = fsdp_wrap(
            self.generator,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.generator_fsdp_wrap_strategy,
            cpu_offload=gen_offload,
            sync_module_states=sync_states,
        )
        self.generator_ema = fsdp_wrap(
            self.generator_ema,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.generator_fsdp_wrap_strategy,
            cpu_offload=ema_offload,
            sync_module_states=sync_states,
        )
        self.teacher = fsdp_wrap(
            self.teacher,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.generator_fsdp_wrap_strategy,
            cpu_offload=teacher_offload,
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
            print(f"[Init][Offload] generator={gen_offload}, ema={ema_offload}, teacher={teacher_offload}", flush=True)

        # VAE
        self.vae_cpu_offload = bool(getattr(cfg, "vae_cpu_offload", False))
        vae_dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32
        if self.vae_cpu_offload:
            self.vae = self.vae.to(device="cpu", dtype=vae_dtype)
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
    # 文本编码 / batch
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
        return {
            "image_latents": processed["image_latents"],
            "prompts_per_turn": batch["prompts"],
        }

    @torch.no_grad()
    def _encode_single_prompt_at(self, prompts_per_turn: List[List[str]], turn_index: int) -> torch.Tensor:
        flat: List[str] = []
        for sample in prompts_per_turn:
            if turn_index < len(sample):
                flat.append(sample[turn_index])
            elif len(sample) > 0:
                flat.append(sample[-1])
            else:
                flat.append("")
        td = self.text_encoder(flat)
        flat_embeds = td["prompt_embeds"] if isinstance(td, dict) else td
        return flat_embeds.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _encode_single_uncond(self, batch_size: int) -> torch.Tensor:
        neg = str(getattr(self.config, "negative_prompt", ""))
        flat = [neg] * batch_size
        td = self.text_encoder(flat)
        flat_embeds = td["prompt_embeds"] if isinstance(td, dict) else td
        return flat_embeds.to(device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------
    def _frame_seq_length(self, h_lat: int, w_lat: int) -> int:
        try:
            patch_size = getattr(self.generator.model, "patch_size", (1, 2, 2))
        except Exception:
            patch_size = (1, 2, 2)
        return (h_lat // int(patch_size[1])) * (w_lat // int(patch_size[2]))

    def _init_kv_cache(self, wrapper_module: WanDiffusionWrapper, batch_size: int, h_lat: int, w_lat: int) -> List[Dict[str, torch.Tensor]]:
        model = wrapper_module.model
        num_blocks = len(model.blocks)
        num_heads = int(model.num_heads)
        head_dim = int(model.dim // model.num_heads)
        local_attn_size = int(getattr(model, "local_attn_size", -1))
        frame_seq_length = self._frame_seq_length(h_lat, w_lat)
        kv_cache_size = local_attn_size * frame_seq_length if local_attn_size != -1 else 32760
        kv_cache: List[Dict[str, torch.Tensor]] = []
        for _ in range(num_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim],
                                 dtype=self.dtype, device=self.device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim],
                                 dtype=self.dtype, device=self.device),
                "global_end_index": torch.zeros([1], dtype=torch.long, device=self.device),
                "local_end_index": torch.zeros([1], dtype=torch.long, device=self.device),
            })
        return kv_cache

    # ------------------------------------------------------------------
    # Self-Rollout history（用 GT history + teacher forcing 风格 prefill，
    # 比 4 步 student rollout 稳定得多；不依赖 student 已经学会少步）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _prefill_history_cache(
        self,
        wrapper: WanDiffusionWrapper,
        image_latents: torch.Tensor,
        prompts_per_turn: List[List[str]],
        target_turn_k: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """把 GT clean latent[:, 0:k] 以 t=context_noise prefill 到 wrapper 的 KV cache。

        区别于 DMD/ODE-init 里用 student 自 rollout history：CD 初期 student 还
        没学会少步，用 GT history 更稳；等 student 收敛了，再切回 student rollout
        也 OK（通过 config.cd_history_mode 控制）。
        """
        B, T, C, H, W = image_latents.shape
        fsl = self._frame_seq_length(H, W)
        kv_cache = self._init_kv_cache(wrapper, B, H, W)

        for t_idx in range(target_turn_k):
            frame = image_latents[:, t_idx:t_idx + 1].to(dtype=self.dtype)
            cur_prompt = self._encode_single_prompt_at(
                prompts_per_turn, turn_index=max(0, t_idx - 1) if t_idx > 0 else 0
            )
            # 首帧对应 prompts[0]（第 1 轮要执行的指令），第 i (i>=1) 帧也对应 prompts[i-1]
            # —— 与 inference / DMD trainer 逻辑保持一致
            if t_idx == 0:
                cur_prompt = self._encode_single_prompt_at(prompts_per_turn, turn_index=0)
            else:
                cur_prompt = self._encode_single_prompt_at(prompts_per_turn, turn_index=t_idx - 1)
            cond = {"prompt_embeds": cur_prompt}
            ts = torch.zeros([B, 1], device=self.device, dtype=torch.long) if t_idx == 0 \
                else torch.full([B, 1], self.context_noise, device=self.device, dtype=torch.long)
            wrapper(
                noisy_image_or_video=frame,
                conditional_dict=cond,
                timestep=ts,
                kv_cache=kv_cache,
                current_start=t_idx * fsl,
                cache_start=t_idx * fsl,
            )
        return kv_cache

    # ------------------------------------------------------------------
    # CD 单步训练
    # ------------------------------------------------------------------
    def _sample_target_turn_k(self, T: int) -> int:
        if T <= 1:
            return 1
        k_t = torch.randint(1, T, (1,), device=self.device, dtype=torch.long)
        if dist.is_initialized():
            dist.broadcast(k_t, src=0)
        return int(k_t.item())

    def _sample_cd_timestep_idx(self) -> int:
        """采样 [0, N-1)，rank0 broadcast。"""
        idx = torch.randint(0, self.discrete_cd_N - 1, (1,), device=self.device, dtype=torch.long)
        if dist.is_initialized():
            dist.broadcast(idx, src=0)
        return int(idx.item())

    def cd_step(
        self,
        image_latents: torch.Tensor,
        prompts_per_turn: List[List[str]],
        target_turn_k: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """Consistency distillation 一步。"""
        B, T, C, H, W = image_latents.shape
        if T <= 1:
            zero = torch.zeros([], device=self.device, dtype=self.dtype, requires_grad=True)
            return zero, {"cd_loss": 0.0, "skipped": 1}
        target_turn_k = int(max(1, min(target_turn_k, T - 1)))
        fsl = self._frame_seq_length(H, W)
        current_start = target_turn_k * fsl

        # ---- Step 1: 把 GT history 分别 prefill 进 teacher / student / ema 三份 KV cache ----
        # 三份 cache 独立；每个 wrapper forward 时的 KV 必须是它自己的。
        teacher_kv = self._prefill_history_cache(self.teacher, image_latents, prompts_per_turn, target_turn_k)
        gen_kv = self._prefill_history_cache(self.generator, image_latents, prompts_per_turn, target_turn_k)
        ema_kv = self._prefill_history_cache(self.generator_ema, image_latents, prompts_per_turn, target_turn_k)

        # ---- Step 2: 编码 prompt（目标 turn 和 negative）----
        cur_prompt = self._encode_single_prompt_at(prompts_per_turn, turn_index=target_turn_k - 1)
        cond = {"prompt_embeds": cur_prompt}
        uncond_prompt = self._encode_single_uncond(B)
        uncond = {"prompt_embeds": uncond_prompt}

        # ---- Step 3: 采 (t, t_next) ----
        ts_idx = self._sample_cd_timestep_idx()
        t_val = float(self.cd_scheduler.timesteps[ts_idx].item())
        t_next_val = float(self.cd_scheduler.timesteps[ts_idx + 1].item())
        t_tensor = torch.full([B, 1], t_val, device=self.device, dtype=self.dtype)
        t_next_tensor = torch.full([B, 1], t_next_val, device=self.device, dtype=self.dtype)

        # ---- Step 4: 对 GT clean_k 加噪到 t ----
        clean_k = image_latents[:, target_turn_k:target_turn_k + 1].to(dtype=self.dtype)  # [B, 1, C, H, W]
        noise_k = torch.randn_like(clean_k)
        t_for_add = torch.tensor([t_val], device=self.device)
        latent_t = self.cd_scheduler.add_noise(
            clean_k.flatten(0, 1), noise_k.flatten(0, 1), t_for_add
        ).unflatten(0, (B, 1)).to(dtype=self.dtype)

        # ---- Step 5: Teacher 单步 ODE（带 CFG）-> latent_t_next ----
        with torch.no_grad():
            v_cond, _ = self.teacher(
                noisy_image_or_video=latent_t,
                conditional_dict=cond,
                timestep=t_tensor,
                kv_cache=teacher_kv,
                current_start=current_start,
                cache_start=current_start,
            )
            v_uncond, _ = self.teacher(
                noisy_image_or_video=latent_t,
                conditional_dict=uncond,
                timestep=t_tensor,
                kv_cache=teacher_kv,
                current_start=current_start,
                cache_start=current_start,
            )
            v_pred = v_uncond + self.teacher_guidance_scale * (v_cond - v_uncond)
            dt = (t_tensor - t_next_tensor).reshape(B, 1, 1, 1, 1).to(v_pred.dtype) / 1000.0
            latent_t_next = latent_t - dt * v_pred

        # ---- Step 6: Student forward at (latent_t, t)，带梯度 ----
        _, cm_pred_t = self.generator(
            noisy_image_or_video=latent_t,
            conditional_dict=cond,
            timestep=t_tensor,
            kv_cache=gen_kv,
            current_start=current_start,
            cache_start=current_start,
        )

        # ---- Step 7: EMA student forward at (latent_t_next, t_next)，no_grad ----
        # 先把最新 EMA 权重 copy 到 generator_ema（FSDP 内部处理 shard）
        with torch.no_grad():
            self.generator_ema_state.copy_to(self.generator_ema)
            _, cm_pred_t_next = self.generator_ema(
                noisy_image_or_video=latent_t_next,
                conditional_dict=cond,
                timestep=t_next_tensor,
                kv_cache=ema_kv,
                current_start=current_start,
                cache_start=current_start,
            )

        # ---- Step 8: CD loss ----
        loss = F.mse_loss(cm_pred_t, cm_pred_t_next.detach(), reduction="mean")

        log_dict = {
            "cd_loss": loss.detach().item(),
            "t": t_val,
            "t_next": t_next_val,
            "ts_idx": ts_idx,
            "target_turn_k": target_turn_k,
        }

        del teacher_kv, gen_kv, ema_kv
        return loss, log_dict

    def fwd_bwd_one_step(self, prepared_batch: Dict) -> Dict:
        image_latents = prepared_batch["image_latents"]
        prompts_per_turn = prepared_batch["prompts_per_turn"]
        T = image_latents.shape[1]
        target_turn_k = self._sample_target_turn_k(T)

        self.generator.train()
        loss, log_dict = self.cd_step(image_latents, prompts_per_turn, target_turn_k)
        (loss / self.gradient_accumulation_steps).backward()
        log_dict["cd_loss_raw"] = loss.detach().item()
        return log_dict

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(self) -> None:
        if self.is_main_process:
            print(f"[Save] gathering distributed model states at step {self.step} ...")

        gen_sd = fsdp_state_dict(self.generator)
        state = {
            "generator": gen_sd,
            "generator_ema": self.generator_ema_state.state_dict(),
            "step": self.step,
        }

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
        cfg = self.config
        is_last_micro = (self._accum_step_idx == self.gradient_accumulation_steps - 1)

        batch = next(self.dataloader)
        prepared = self._prepare_batch(batch)
        log_dict = self.fwd_bwd_one_step(prepared)

        self._accum_step_idx += 1

        if is_last_micro:
            grad_norm = self.generator.clip_grad_norm_(self.max_grad_norm_generator)
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad(set_to_none=True)
            # 更新 EMA（只在真正完成一次优化器 step 后）
            self.generator_ema_state.update(self.generator)
            self._accum_step_idx = 0
        else:
            grad_norm = None

        log_iters = int(getattr(cfg, "log_iters", 1))
        if self.is_main_process and self.writer is not None and (self.step % log_iters == 0):
            for k, v in log_dict.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"train/{k}", float(v), self.step)
            if grad_norm is not None:
                try:
                    self.writer.add_scalar("train/generator_grad_norm", float(grad_norm), self.step)
                except Exception:
                    pass

        if self.is_main_process and (self.step % log_iters == 0):
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