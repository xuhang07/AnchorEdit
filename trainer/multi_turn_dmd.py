"""
Multi-Turn Causal I2V DMD Distillation Trainer.

把已经训练好的多轮 causal 编辑模型蒸馏成少步（默认 4 步）模型。

核心设计：
  - generator / real_score / fake_score 都是 causal AR 多轮模型，三者共享同一份
    pretrained ckpt 初始化（real_score 冻结，generator/fake_score 训练）。
  - 每个 micro-step：
      1. 取一条多轮 batch -> [B, T, C, H, W]，T = 1 + N_turns
      2. 同步采样目标 turn k ∈ [1, T-1]
      3. no_grad self-rollout 出 history_pred[:, 1:k]，并把 [0:k] 都写入
         generator 的 KV cache（与真实 AR 推理路径一致）
      4. 在 k 位置采样一个 denoising step idx s，做 backward simulation 拿到
         noisy_input_k，单步 forward generator -> pred_x0_k（带梯度）
      5. real / fake score 不带 KV cache，整段 causal forward；只在第 k 帧
         加噪 + 计算 DMD grad（前面帧 t=0，后面帧 t=0 仅占位）

参考：
  - DMD baseline: model/dmd.py、trainer/distillation.py
  - 多轮训练: trainer/multi_turn_i2v.py
  - causal AR 推理: pipeline/multi_turn_inference.py、inference_multi_turn_i2v.py
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
from utils.loss import get_denoising_loss
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
    """加载多轮 causal 模型 ckpt 到 WanDiffusionWrapper（generator / real / fake 共用）。

    Ckpt 解析优先级：
        - `prefer_ema=True` 时优先取 `generator_ema`，找不到再回落到 `generator` / `model`。
          多轮训练 ckpt 结构（见 [`save_checkpoint()`](trainer/multi_turn_i2v.py:748)）同时
          保存 EMA 与实时权重；推理 ([`inference_multi_turn_i2v.py`](inference_multi_turn_i2v.py:1))
          用的是 EMA，蒸馏 teacher / 初始 generator 也以 EMA 为起点更一致、更稳。
        - `prefer_ema=False` 时按原顺序：generator -> model -> raw。

    Args:
        rank0_only: 默认 False（保持原行为，所有 rank 都加载）。设为 True 时，
            仅 rank0 真正读取 ckpt 并 load_state_dict；其它 rank 跳过，等待
            FSDP wrap 阶段通过 `sync_module_states=True` 从 rank0 broadcast。
            这是为了避免每个 rank 都在 CPU 上完整解压一份 14B ckpt 导致
            集群 cgroup 触发 OOMKilled (exitCode 137)。
        prefer_ema: 是否优先使用 EMA 权重。DMD 蒸馏建议 True：三份模型
            （generator / real_score / fake_score）都以 EMA 为起点，KL gradient
            更贴近真实推理分布，也更稳定。
    """
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

    # 显式释放 CPU 内存：避免连续加载 generator/real_score/fake_score 三份 ckpt
    # 时 Python 对象延迟回收导致 RSS 峰值 = 3×ckpt_size。
    del ckpt
    del sd
    gc.collect()


class MultiTurnDMDTrainer:
    """
    多轮 causal AR DMD 蒸馏训练器。
    """

    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # ----- 分布式初始化 -----
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

        # ----- 蒸馏超参 -----
        self.num_train_timestep = int(config.num_train_timestep)
        self.min_step = int(getattr(config, "min_step_ratio", 0.02) * self.num_train_timestep)
        self.max_step = int(getattr(config, "max_step_ratio", 0.98) * self.num_train_timestep)
        self.real_guidance_scale = float(getattr(config, "real_guidance_scale",
                                                 getattr(config, "guidance_scale", 6.0)))
        self.fake_guidance_scale = float(getattr(config, "fake_guidance_scale", 0.0))
        self.timestep_shift = float(getattr(config, "timestep_shift", 1.0))
        self.context_noise = int(getattr(config, "context_noise", 0))
        self.dfake_gen_update_ratio = int(getattr(config, "dfake_gen_update_ratio", 5))
        self.gradient_accumulation_steps = max(
            1, int(getattr(config, "gradient_accumulation_steps", 1))
        )
        self._accum_step_idx = 0

        # ----- 纯蒸馏 / GT-free 控制 -----
        # self_rollout_prob: 第 k 帧 backward simulation 输入的来源概率
        #   = 1.0  完全 self-rollout（从 pure noise 多步去噪到第 s 步），不依赖 GT
        #   = 0.0  完全用 GT 第 k 帧 + add_noise（旧行为，可能引入偏色）
        #   ∈ (0,1) 按概率混合（每个 micro-step 独立采样，跨 rank 同步）
        # 默认 1.0：用户明确要纯蒸馏
        self.self_rollout_prob = float(getattr(config, "self_rollout_prob", 1.0))
        # use_gt_future_frames: real/fake score 整段 forward 时，第 k+1..T-1 帧
        #   是否仍用 GT 占位。True 旧行为；False 用零张量占位（避免后续 GT 帧
        #   通过 causal attention 影响第 k 帧的 score 估计）。
        #   注意：因为模型是 causal，理论上后续帧不会影响 k；但 cross-attn mask
        #   的构造依赖 num_prompts 维度，置零更干净。默认 False（纯蒸馏）。
        self.use_gt_future_frames = bool(getattr(config, "use_gt_future_frames", False))

        # ----- 少步 schedule -----
        self.denoising_step_list = torch.tensor(
            list(config.denoising_step_list), dtype=torch.long, device=self.device
        )
        if bool(getattr(config, "warp_denoising_step", False)):
            timesteps = torch.cat(
                (self._build_scheduler_timesteps().cpu(), torch.tensor([0], dtype=torch.float32))
            )
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list.cpu()].long().to(self.device)

        # ----- 构建模型 -----
        self._setup_models()
        self._setup_optimizer()
        self._setup_dataloader()

        # 取一些常用形状
        self.num_frame_per_block = int(getattr(config, "num_frame_per_block", 1))

        self.max_grad_norm_generator = float(getattr(config, "max_grad_norm_generator", 10.0))
        self.max_grad_norm_critic = float(getattr(config, "max_grad_norm_critic", 10.0))

        # ----- EMA -----
        ema_weight = float(getattr(config, "ema_weight", 0.0) or 0.0)
        self.generator_ema = None
        if ema_weight > 0.0:
            if self.is_main_process:
                print(f"[EMA] enable with decay={ema_weight}")
            self.generator_ema = EMA_FSDP(self.generator, decay=ema_weight)

        # ----- denoising loss for fake_score（flow / x0 / etc.）-----
        self.denoising_loss_func = get_denoising_loss(
            getattr(config, "denoising_loss_type", "flow")
        )()

        self.previous_time = None

        if self.is_main_process:
            print(f"[Init] denoising_step_list={self.denoising_step_list.tolist()}, "
                  f"dfake_gen_update_ratio={self.dfake_gen_update_ratio}, "
                  f"grad_accum={self.gradient_accumulation_steps}, "
                  f"world_size={self.world_size}")
            print(f"[Init][PureDistill] self_rollout_prob={self.self_rollout_prob}, "
                  f"use_gt_future_frames={self.use_gt_future_frames}")

    # ------------------------------------------------------------------
    # 模型 / 优化器 / 数据
    # ------------------------------------------------------------------
    def _build_scheduler_timesteps(self) -> torch.Tensor:
        """临时建一个 wrapper 以拿到 1000 训练 timesteps 序列（仅用于 warp）。"""
        tmp = WanDiffusionWrapper(**self.config.model_kwargs).get_scheduler()
        return tmp.timesteps

    def _setup_models(self) -> None:
        cfg = self.config
        model_kwargs = OmegaConf.to_container(cfg.model_kwargs, resolve=True)

        # generator（要训练 + 少步推理）
        self.generator = WanDiffusionWrapper(**model_kwargs)
        self.generator.model.requires_grad_(True)
        # gradient checkpointing：直接设属性（CausalWanModel 默认就是 True；HF 版本差异下
        # enable_gradient_checkpointing() 可能失败，故直接绕开）
        if bool(getattr(cfg, "gradient_checkpointing", False)):
            self.generator.model.gradient_checkpointing = True

        # real_score（teacher，冻结）
        self.real_score = WanDiffusionWrapper(**model_kwargs)
        self.real_score.model.requires_grad_(False)
        self.real_score.model.eval()

        # fake_score（critic，训练）
        self.fake_score = WanDiffusionWrapper(**model_kwargs)
        self.fake_score.model.requires_grad_(True)
        if bool(getattr(cfg, "gradient_checkpointing", False)):
            self.fake_score.model.gradient_checkpointing = True

        # text encoder / vae
        self.text_encoder = WanTextEncoder(model_kwargs["model_name"])
        self.text_encoder.requires_grad_(False)
        self.vae = WanVAEWrapper(model_kwargs["model_name"])
        self.vae.requires_grad_(False)

        # scheduler 共享一份（实际上每个 wrapper 自带，但训练逻辑用 generator 这份）
        self.scheduler = self.generator.get_scheduler()
        # FlowMatchScheduler.add_noise 内部用 argmin(|self.timesteps - t|) 找 sigma 网格点。
        # 默认 set_timesteps(num_inference_steps=100) 只有 100 个 sigma 点，
        # 而 denoising_step_list 来自 [0, 1000) 区间，会把 1000/750/500/250 投到 100 点
        # 网格上引入显著偏差。这里强制对齐到 1000 点网格（与 inference_multi_turn_dmd.py
        # 中 `set_timesteps(num_inference_steps=num_train_timesteps, training=True)` 完全一致），
        # 让训练 / 推理 add_noise 使用的 sigma 完全相同。
        self.scheduler.set_timesteps(num_inference_steps=self.num_train_timestep, training=True)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

        # ---- 加载 ckpt：三个模型都用同一份多轮 causal pretrained ----
        # 内存优化要点：
        # 1) [`_load_generator_ckpt()`](trainer/multi_turn_dmd.py:64) 内部加载完
        #    立刻 `del ckpt; gc.collect()`，避免 Python 引用计数延迟回收，
        #    使三次连续加载的 RAM 峰值 ~= 1×ckpt_size 而非 3×ckpt_size。
        # 2) 可选 `rank0_only_load`：仅 rank0 加载 ckpt，其它 rank 在 wrap
        #    时通过 FSDP `sync_module_states=True` 接收 broadcast。这能进一步
        #    把单节点峰值 RAM 从 8×ckpt_size 降到 1×ckpt_size（缓解集群
        #    cgroup OOMKilled / exitCode 137）。
        #    注意：开启后，非 rank0 的模型必须仍是合法可初始化对象（即使
        #    权重是随机初始化的），FSDP 会用 rank0 的真实权重覆盖之。
        ckpt_cfg = getattr(cfg, "checkpoint", None)
        rank0_only_load = bool(getattr(ckpt_cfg, "rank0_only_load", True)) if ckpt_cfg is not None else True
        # prefer_ema=True 时，三份模型都从 ckpt["generator_ema"] 初始化，
        # 与推理路径完全一致；resume 用的 generator_ckpt / fake_score_ckpt
        # 仍走实时权重（见下方 override 分支）。
        prefer_ema = bool(getattr(ckpt_cfg, "prefer_ema", True)) if ckpt_cfg is not None else True

        pretrained_pt = getattr(ckpt_cfg, "pretrained_pt", None) if ckpt_cfg is not None else None
        if pretrained_pt:
            _load_generator_ckpt(self.generator, pretrained_pt, tag="generator(init)",
                                 rank0_only=rank0_only_load, prefer_ema=prefer_ema)
            _load_generator_ckpt(self.real_score, pretrained_pt, tag="real_score(teacher)",
                                 rank0_only=rank0_only_load, prefer_ema=prefer_ema)
            _load_generator_ckpt(self.fake_score, pretrained_pt, tag="fake_score(init)",
                                 rank0_only=rank0_only_load, prefer_ema=prefer_ema)

        # 单独覆盖（resume 场景下仍取实时权重，保证恢复精确训练状态）
        if ckpt_cfg is not None:
            gen_ckpt = getattr(ckpt_cfg, "generator_ckpt", None)
            if gen_ckpt:
                _load_generator_ckpt(self.generator, gen_ckpt, tag="generator(override)",
                                     rank0_only=rank0_only_load, prefer_ema=False)
            fake_ckpt = getattr(ckpt_cfg, "fake_score_ckpt", None)
            if fake_ckpt:
                _load_generator_ckpt(self.fake_score, fake_ckpt, tag="fake_score(override)",
                                     rank0_only=rank0_only_load, prefer_ema=False)

        # 等待 rank0 加载完毕，避免其它 rank 提前进入 FSDP wrap
        if dist.is_initialized():
            dist.barrier()
        gc.collect()

        # ---- FSDP wrap ----
        # 显存优化：通过 cpu_offload 把参数常驻 CPU，仅在 forward / backward
        # 时 all-gather 到 GPU。real_score 是冻结 teacher，offload 几乎无吞吐
        # 损失（仅多一次 H2D，且 no_grad 不需要 backward），强烈推荐打开。
        # generator / fake_score 是训练模型，offload 会显著降速但能省 50%+
        # 参数显存，按需打开。
        gen_offload = bool(getattr(cfg, "generator_cpu_offload", False))
        real_offload = bool(getattr(cfg, "real_score_cpu_offload", True))
        fake_offload = bool(getattr(cfg, "fake_score_cpu_offload", False))

        # sync_module_states 仅在 rank0_only_load 时打开，让 FSDP 自动从 rank0
        # broadcast 权重到全 group。文本编码器较小、且没在 ckpt 里，无需广播。
        sync_states = rank0_only_load

        self.generator = fsdp_wrap(
            self.generator,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.generator_fsdp_wrap_strategy,
            cpu_offload=gen_offload,
            sync_module_states=sync_states,
        )
        self.real_score = fsdp_wrap(
            self.real_score,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.real_score_fsdp_wrap_strategy,
            cpu_offload=real_offload,
            sync_module_states=sync_states,
        )
        self.fake_score = fsdp_wrap(
            self.fake_score,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.fake_score_fsdp_wrap_strategy,
            cpu_offload=fake_offload,
            sync_module_states=sync_states,
        )
        self.text_encoder = fsdp_wrap(
            self.text_encoder,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.text_encoder_fsdp_wrap_strategy,
            cpu_offload=bool(getattr(cfg, "text_encoder_cpu_offload", True)),
            # text_encoder 权重不来自 ckpt（HuggingFace 自动下载），所有 rank 已
            # 加载相同权重，无需 sync_module_states。
        )

        if self.is_main_process:
            print(
                f"[Init][Offload] generator={gen_offload}, real_score={real_offload}, "
                f"fake_score={fake_offload}, "
                f"text_encoder={bool(getattr(cfg, 'text_encoder_cpu_offload', True))}",
                flush=True,
            )

        # VAE：默认放 GPU；可通过 vae_cpu_offload=True 改为常驻 CPU，
        # 仅在 _prepare_batch 编码时临时搬到 GPU（编码量很小，开销可忽略）。
        self.vae_cpu_offload = bool(getattr(cfg, "vae_cpu_offload", False))
        vae_dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32
        if self.vae_cpu_offload:
            self.vae = self.vae.to(device="cpu", dtype=vae_dtype)
            if self.is_main_process:
                print("[Init][Offload] vae=True (kept on CPU, moved to GPU on demand)", flush=True)
        else:
            self.vae = self.vae.to(device=self.device, dtype=vae_dtype)

        # tokenizer（图像 -> latent + 文本编码）
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
        self.critic_optimizer = torch.optim.AdamW(
            [p for p in self.fake_score.parameters() if p.requires_grad],
            lr=float(getattr(cfg, "lr_critic", cfg.lr)),
            betas=(float(getattr(cfg, "beta1_critic", 0.0)), float(getattr(cfg, "beta2_critic", 0.999))),
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )

    def _setup_dataloader(self) -> None:
        cfg = self.config
        # DMD 蒸馏要求所有 batch 的 num_turns 严格等于 max_turns，确保多 rank
        # FSDP 训练时不会因为 T 不齐而出现集合通信对齐问题（参见
        # _sample_target_turn_k 注释）。可在 config 中通过 data.require_exact_turns
        # 显式覆盖，默认 True。
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
    # 数据预处理 / 文本编码
    # ------------------------------------------------------------------
    def _prepare_batch(self, batch: Dict) -> Dict:
        """
        利用现有 MultiTurnTokenizer 把 batch 中的图像/文本编码到 latent / embeddings。
        Returns:
          - image_latents: [B, T, C, H, W]，T = 1 + N_turns
          - prompt_embeds_full: [B, T*L, D]  对应所有 turn 的 prompt 拼接（不一定用得到）
          - prompts_per_turn: List[List[str]]，原始 prompt（按 turn 维度），shape [B][N_turns]
        """
        # 若开启了 vae_cpu_offload，编码前临时把 VAE 搬到 GPU，编码后搬回 CPU
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
    def _encode_prompts_for_k(
        self,
        prompts_per_turn: List[List[str]],
        k: int,
    ) -> torch.Tensor:
        """
        编码到第 k turn 为止的 prompt（即 prompts[0..k-1] 拼接），shape [B, k*L, D]。
        与 utils/multi_turn_tokenizer.py 中 encode_texts 逻辑一致。
        """
        batch_size = len(prompts_per_turn)
        truncated = []
        for sample in prompts_per_turn:
            ps = list(sample)[:k]
            while len(ps) < k:
                ps.append("")
            truncated.append(ps)

        flat = [p for sub in truncated for p in sub]
        td = self.text_encoder(flat)
        if isinstance(td, dict):
            flat_embeds = td["prompt_embeds"]
        else:
            flat_embeds = td
        flat_embeds = flat_embeds.to(device=self.device, dtype=self.dtype)
        _, seq_len, hidden_dim = flat_embeds.shape
        out = flat_embeds.view(batch_size, k, seq_len, hidden_dim).reshape(
            batch_size, k * seq_len, hidden_dim
        )
        return out

    @torch.no_grad()
    def _encode_single_prompt_at(
        self,
        prompts_per_turn: List[List[str]],
        turn_index: int,
    ) -> torch.Tensor:
        """
        编码 batch 中每个样本的「第 turn_index 个 prompt（0-based）」一条 -> [B, L, D]。

        与推理路径（pipeline/multi_turn_inference.py、inference_multi_turn_dmd.py）保持一致：
        模型在 cross_attn_current_prompt_only=True 下，KV cache prefill / rollout 期间
        每一帧只看属于自己那一条 prompt，因此训练自回归 rollout 也必须每帧只传单条 prompt，
        否则 generator 在推理时（单条 prompt）行为与训练时（累计 prompt）不一致，导致蒸馏崩塌。
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
        flat_embeds = flat_embeds.to(device=self.device, dtype=self.dtype)
        # [B, L, D] -> 视为 num_prompts=1 的拼接结果
        return flat_embeds

    @torch.no_grad()
    def _encode_single_uncond(self, batch_size: int) -> torch.Tensor:
        """编码 negative prompt 一条 -> [B, L, D]。与 _encode_single_prompt_at 配套。"""
        neg = str(getattr(self.config, "negative_prompt", ""))
        flat = [neg] * batch_size
        td = self.text_encoder(flat)
        if isinstance(td, dict):
            flat_embeds = td["prompt_embeds"]
        else:
            flat_embeds = td
        return flat_embeds.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _encode_uncond_for_k(self, batch_size: int, k: int) -> torch.Tensor:
        """编码 negative prompt，对齐到 k 条 -> [B, k*L, D]。"""
        neg = str(getattr(self.config, "negative_prompt", ""))
        flat = [neg] * (batch_size * k)
        td = self.text_encoder(flat)
        if isinstance(td, dict):
            flat_embeds = td["prompt_embeds"]
        else:
            flat_embeds = td
        flat_embeds = flat_embeds.to(device=self.device, dtype=self.dtype)
        _, seq_len, hidden_dim = flat_embeds.shape
        out = flat_embeds.view(batch_size, k, seq_len, hidden_dim).reshape(
            batch_size, k * seq_len, hidden_dim
        )
        return out

    # ------------------------------------------------------------------
    # KV cache 工具
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
            kv_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, num_heads, head_dim],
                        dtype=self.dtype, device=self.device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, num_heads, head_dim],
                        dtype=self.dtype, device=self.device,
                    ),
                    "global_end_index": torch.zeros([1], dtype=torch.long, device=self.device),
                    "local_end_index": torch.zeros([1], dtype=torch.long, device=self.device),
                }
            )
        return kv_cache

    @staticmethod
    def _reset_kv_cache_indices(kv_cache: List[Dict[str, torch.Tensor]]) -> None:
        for layer in kv_cache:
            layer["global_end_index"].zero_()
            layer["local_end_index"].zero_()

    @staticmethod
    def _rewind_kv_cache_to(
        kv_cache: List[Dict[str, torch.Tensor]],
        rewind_to: int,
    ) -> None:
        """
        把 KV cache 的写入指针回退到 `rewind_to`（global_end_index 单位 = token 数）。

        语义：CausalWanModel 写入新帧时按 `local_end_index += (current_end - global_end_index)`
        计算位置；只要把两个 end_index 都置为 `rewind_to`，下一次以 `current_start=rewind_to`
        进行 forward 时就会从这个位置开始覆盖写入，等价于"丢弃"了之前写在该位置的 K/V。

        注意：仅在没有触发 local-attn rolling（cache 没溢出）的情况下严格成立。
        本训练脚本中 local_attn_size = -1（全局 attn）+ 多轮编辑帧数远小于 32760，
        永远不会走 rolling 分支，因此安全。
        """
        for layer in kv_cache:
            layer["global_end_index"].fill_(int(rewind_to))
            layer["local_end_index"].fill_(int(rewind_to))

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
        """
        用 generator 自回归生成 turn 1..k-1 的 latent（模拟真实推理路径）。

        流程：
          1. 初始化 KV cache
          2. prefill 首帧 GT（t=0）
          3. 对 turn 1..k-1：
             a. 随机 noise -> 跑 denoising_step_list 少步去噪 -> clean_t
             b. 把 clean_t 以 t=0 写回 KV cache
          4. 返回 history_pred [B, k, C, H, W]（含首帧 GT）和 KV cache

        Args:
            image_latents: [B, T, C, H, W]，GT latent（仅用首帧）
            prompts_per_turn: [B][N_turns]，原始 prompt
            target_turn_k: 目标 turn 索引（1-based in frame dim）
        Returns:
            history_pred: [B, k, C, H, W]，前 k 帧（含首帧 GT + k-1 帧 self-rollout）
            kv_cache: 已经 prefill 到第 k-1 帧的 KV cache
        """
        B, T, C, H, W = image_latents.shape
        fsl = self._frame_seq_length(H, W)

        # 初始化 KV cache
        kv_cache = self._init_kv_cache(B, H, W)

        # 首帧 GT prefill
        first_frame = image_latents[:, 0:1].to(dtype=self.dtype)  # [B, 1, C, H, W]
        # 编码首帧对应的 prompt：与 pipeline/multi_turn_inference.py 一致，用 prompts[0]
        # （即第 1 轮要执行的指令）作为首帧的 cross-attn 上下文，单条 prompt。
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

        history_frames = [first_frame.squeeze(1)]  # list of [B, C, H, W]

        # 逐 turn self-rollout
        for t_idx in range(1, target_turn_k):
            # 第 t_idx 帧（1-based turn）对应 prompts[t_idx - 1]（0-based）
            # 与推理 multi_turn_inference / inference_multi_turn_dmd 严格一致：
            # 每轮 cross-attn 只看自己那一条 prompt（cross_attn_current_prompt_only=True）。
            cur_prompt = self._encode_single_prompt_at(prompts_per_turn, turn_index=t_idx - 1)
            cond_t = {"prompt_embeds": cur_prompt}
            current_start = t_idx * fsl

            # 初始化 noise
            noisy_t = torch.randn([B, 1, C, H, W], device=self.device, dtype=self.dtype)

            # 少步去噪循环
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
                    clean_t = pred_x0  # [B, 1, C, H, W]

            # 把 clean_t 以 t=context_noise 写回 KV cache（与推理一致）
            ctx_ts = torch.full([B, 1], self.context_noise, device=self.device, dtype=torch.long)
            self.generator(
                noisy_image_or_video=clean_t,
                conditional_dict=cond_t,
                timestep=ctx_ts,
                kv_cache=kv_cache,
                current_start=current_start,
                cache_start=current_start,
            )

            history_frames.append(clean_t.squeeze(1))  # [B, C, H, W]

        # stack -> [B, k, C, H, W]
        history_pred = torch.stack(history_frames, dim=1)
        return history_pred, kv_cache

    # ------------------------------------------------------------------
    # Generator 单步 forward（带梯度）
    # ------------------------------------------------------------------
    def _run_generator_at_k(
        self,
        image_latents: torch.Tensor,
        prompts_per_turn: List[List[str]],
        target_turn_k: int,
        kv_cache: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, int, int]:
        """
        在第 k turn 位置构造 noisy_input_k，并单步 forward（带梯度）得到 pred_x0_k。

        两种 noisy_k 构造模式（由 self.self_rollout_prob 控制，跨 rank 同步采样）:

          (A) self-rollout（纯蒸馏，不依赖 GT 第 k 帧）:
              从 pure noise 出发，用 generator 跑 denoising_step_list[0..s-1] 的前
              s 步（no_grad），得到第 s 步对应 timestep 的 noisy_k。这等价于推理
              路径走到一半的中间态，完全不接触 GT 第 k 帧，从而解耦 GT 依赖，
              避免 GT 分布不一致导致的偏色。

          (B) GT backward simulation（旧行为）:
              取 GT 第 k 帧 + scheduler.add_noise(timestep=denoising_step_list[s])
              得到 noisy_k。实现简单但会把 GT 颜色分布强行带进来。

        选择好 noisy_k 后，用单步带梯度 forward 得到 pred_x0_k，作为 DMD 的
        "generator 估计 clean"。

        Args:
            image_latents: [B, T, C, H, W]，GT latent（只在 mode B 或回退时用到）
            prompts_per_turn: [B][N_turns]
            target_turn_k: 目标 turn 索引（frame dim，1-based）
            kv_cache: 已经 prefill 到第 k-1 帧的 KV cache（不含第 k 帧）
        Returns:
            pred_x0_k: [B, 1, C, H, W]，generator 对第 k 帧的预测（带梯度）
            denoised_from: int，输入 timestep（即 denoising_step_list[step_idx]）
            denoised_to: int，下一个 timestep（仅用于日志）
        """
        B, T, C, H, W = image_latents.shape
        fsl = self._frame_seq_length(H, W)
        current_start = target_turn_k * fsl

        # 编码第 k turn 对应的 prompt（单条，与推理一致）
        cur_prompt = self._encode_single_prompt_at(prompts_per_turn, turn_index=target_turn_k - 1)
        cond_k = {"prompt_embeds": cur_prompt}

        # 跨 rank 同步采样 step_idx ∈ [0, len(denoising_step_list))
        step_idx = torch.randint(0, len(self.denoising_step_list), (1,), device=self.device)
        if dist.is_initialized():
            dist.broadcast(step_idx, src=0)
        step_idx = int(step_idx.item())

        ts_val = self.denoising_step_list[step_idx]
        denoised_from = int(ts_val.item())
        denoised_to = (
            int(self.denoising_step_list[step_idx + 1].item())
            if step_idx < len(self.denoising_step_list) - 1
            else 0
        )

        # 跨 rank 同步采样"是否走 self-rollout"（伯努利）
        use_self_rollout = True
        if self.self_rollout_prob < 1.0:
            flag = torch.zeros([1], device=self.device, dtype=torch.float32)
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                flag[0] = 1.0 if (torch.rand(1).item() < self.self_rollout_prob) else 0.0
            if dist.is_initialized():
                dist.broadcast(flag, src=0)
            use_self_rollout = bool(flag.item() > 0.5)

        # ---- 构造 noisy_k ----
        if use_self_rollout:
            # (A) 从 pure noise 出发，无梯度跑前 step_idx 步到达 denoising_step_list[step_idx]
            # 注意：denoising_step_list 是降序的（例如 [1000, 750, 500, 250]），
            # 第 0 步输入 ts=1000 的 pure noise，第 s 步输入 ts=denoising_step_list[s]。
            # 因此要走到 step_idx 对应的 noisy_k：
            #   - 如果 step_idx == 0：noisy_k 就是 pure noise（ts=1000）
            #   - 否则：从 pure noise 开始，跑 step 0..step_idx-1 的 generator 预测，
            #     用 add_noise(next_ts) 把 pred_x0 再加噪回到 next_ts 对应的 noisy 状态。
            noisy_k = torch.randn([B, 1, C, H, W], device=self.device, dtype=self.dtype)
            if step_idx > 0:
                with torch.no_grad():
                    for s in range(step_idx):
                        ts_s = int(self.denoising_step_list[s].item())
                        ts_tensor_s = torch.full([B, 1], ts_s, device=self.device, dtype=torch.long)
                        _, pred_x0_s = self.generator(
                            noisy_image_or_video=noisy_k,
                            conditional_dict=cond_k,
                            timestep=ts_tensor_s,
                            kv_cache=kv_cache,
                            current_start=current_start,
                            cache_start=current_start,
                        )
                        next_ts = int(self.denoising_step_list[s + 1].item())
                        noisy_k = self.scheduler.add_noise(
                            pred_x0_s.flatten(0, 1),
                            torch.randn_like(pred_x0_s.flatten(0, 1)),
                            next_ts * torch.ones([B], device=self.device, dtype=torch.long),
                        ).unflatten(0, (B, 1))
                    # 重要：以上中间步 forward 会把 K/V 写入 kv_cache 的 k 位置，
                    # 导致最终带梯度 forward 时 cache 里已经有"自己上一步"的历史。
                    # 这会破坏 DMD 对"单步去噪"的假设（k 位置应该是空的）。
                    # 所以这里回滚 k 位置的 KV cache 指针：把 global/local_end_index
                    # 回退到 current_start（即 k-1 帧末尾）。
                    self._rewind_kv_cache_to(kv_cache, rewind_to=current_start)
        else:
            # (B) GT backward simulation（保留旧行为，供混合/消融使用）
            clean_k = image_latents[:, target_turn_k:target_turn_k + 1].to(dtype=self.dtype)
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

        # ---- 带梯度 forward ----
        ts_input = torch.full([B, 1], denoised_from, device=self.device, dtype=torch.long)
        _, pred_x0_k = self.generator(
            noisy_image_or_video=noisy_k,
            conditional_dict=cond_k,
            timestep=ts_input,
            kv_cache=kv_cache,
            current_start=current_start,
            cache_start=current_start,
        )

        return pred_x0_k, denoised_from, denoised_to

    # ------------------------------------------------------------------
    # DMD KL gradient（方案 B：整段 causal forward，只在第 k 帧加噪）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_kl_grad(
        self,
        clean_full: torch.Tensor,
        target_turn_k: int,
        cond_dict: Dict[str, torch.Tensor],
        uncond_dict: Dict[str, torch.Tensor],
        estimated_clean_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算 DMD KL gradient（eq 7 in https://arxiv.org/abs/2311.18828）。

        构造 noisy_full：只对第 k 帧加噪，其它帧 t=0。
        real/fake score 整段 causal forward（不带 KV cache），只取第 k 帧的输出。

        Args:
            clean_full: [B, T, C, H, W]，完整序列（history + pred_x0_k + GT 占位）
            target_turn_k: 目标 turn 索引
            cond_dict: {"prompt_embeds": [B, T*L, D]}
            uncond_dict: {"prompt_embeds": [B, T*L, D]}
            estimated_clean_k: [B, 1, C, H, W]，generator 对第 k 帧的预测
        Returns:
            grad_k: [B, 1, C, H, W]，DMD gradient
            log_dict: 日志
        """
        B, T, C, H, W = clean_full.shape

        # 对第 k 帧采样 timestep 并加噪
        score_t = torch.randint(self.min_step, self.max_step, [B], device=self.device, dtype=torch.long)
        if self.timestep_shift > 1.0:
            score_t_f = self.timestep_shift * (score_t.float() / 1000.0) / (
                1.0 + (self.timestep_shift - 1.0) * (score_t.float() / 1000.0)
            ) * 1000.0
            score_t = score_t_f.long().clamp(self.min_step, self.max_step - 1)

        # 构造 timestep_full: [B, T]，只有第 k 帧非零
        timestep_full = torch.zeros([B, T], device=self.device, dtype=torch.long)
        timestep_full[:, target_turn_k] = score_t

        # 构造 noisy_full
        noise_k = torch.randn([B, C, H, W], device=self.device, dtype=self.dtype)
        noisy_full = clean_full.clone()
        noisy_full[:, target_turn_k] = self.scheduler.add_noise(
            clean_full[:, target_turn_k],
            noise_k,
            score_t,
        )

        # --- fake_score forward ---
        _, x0_fake_cond = self.fake_score(
            noisy_image_or_video=noisy_full,
            conditional_dict=cond_dict,
            timestep=timestep_full,
        )
        if self.fake_guidance_scale != 0.0:
            _, x0_fake_uncond = self.fake_score(
                noisy_image_or_video=noisy_full,
                conditional_dict=uncond_dict,
                timestep=timestep_full,
            )
            x0_fake = x0_fake_cond + self.fake_guidance_scale * (x0_fake_cond - x0_fake_uncond)
        else:
            x0_fake = x0_fake_cond

        # --- real_score forward (with CFG) ---
        _, x0_real_cond = self.real_score(
            noisy_image_or_video=noisy_full,
            conditional_dict=cond_dict,
            timestep=timestep_full,
        )
        _, x0_real_uncond = self.real_score(
            noisy_image_or_video=noisy_full,
            conditional_dict=uncond_dict,
            timestep=timestep_full,
        )
        x0_real = x0_real_cond + self.real_guidance_scale * (x0_real_cond - x0_real_uncond)

        # 只取第 k 帧
        fake_k = x0_fake[:, target_turn_k:target_turn_k + 1]  # [B, 1, C, H, W]
        real_k = x0_real[:, target_turn_k:target_turn_k + 1]

        # DMD gradient
        grad_k = fake_k - real_k

        # Gradient normalization (DMD paper eq. 8)
        p_real = estimated_clean_k - real_k
        normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True).clamp_min(1e-6)
        grad_k = grad_k / normalizer
        grad_k = torch.nan_to_num(grad_k)

        log_dict = {
            "dmd_gradient_norm": torch.mean(torch.abs(grad_k)).item(),
            "score_timestep": score_t.float().mean().item(),
        }
        return grad_k, log_dict

    # ------------------------------------------------------------------
    # Generator loss（DMD loss）
    # ------------------------------------------------------------------
    def generator_step(
        self,
        image_latents: torch.Tensor,
        prompts_per_turn: List[List[str]],
        target_turn_k: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generator 训练一步：
          1. self-rollout history（no_grad）
          2. 在第 k turn backward simulation + 单步 forward（带梯度）-> pred_x0_k
          3. 构造 clean_full，计算 DMD loss

        Returns:
            loss: scalar
            log_dict: 日志
        """
        B, T, C, H, W = image_latents.shape

        # 防御：确保 target_turn_k 在本 rank 的 [1, T-1] 范围内。
        # 正常情况下 _sample_target_turn_k 已用本 rank 的 T 采样，这里只是兜底。
        if T <= 1:
            zero = torch.zeros([], device=self.device, dtype=self.dtype, requires_grad=True)
            return zero, {"generator_loss": 0.0, "skipped": 1, "target_turn_k": 0}
        target_turn_k = int(max(1, min(target_turn_k, T - 1)))

        # Step 1: self-rollout history
        history_pred, kv_cache = self._self_rollout_history(
            image_latents=image_latents,
            prompts_per_turn=prompts_per_turn,
            target_turn_k=target_turn_k,
        )

        # ---- 防御 + 诊断 ----
        # 期望 history_pred 形状: [B, target_turn_k, C, H, W]
        if history_pred.dim() != 5 or history_pred.shape[1] != target_turn_k:
            try:
                rank = dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                rank = 0
            if rank == 0:
                print(
                    f"[generator_step] WARN history_pred shape={tuple(history_pred.shape)} "
                    f"target_turn_k={target_turn_k} expected k_hist={target_turn_k}",
                    flush=True,
                )
            if history_pred.dim() == 4:
                history_pred = history_pred.unsqueeze(0)

        k_hist = history_pred.shape[1]
        n_write = min(k_hist, target_turn_k, T)

        # Step 2: generator forward at k（带梯度）
        pred_x0_k, denoised_from, denoised_to = self._run_generator_at_k(
            image_latents=image_latents,
            prompts_per_turn=prompts_per_turn,
            target_turn_k=target_turn_k,
            kv_cache=kv_cache,
        )

        # Step 3: 构造 clean_full [B, T, C, H, W]
        # 布局：
        #   [:, 0]            = 首帧 GT（来自 image_latents[:, 0]，是 reference image，必须保留）
        #   [:, 1:k]          = self-rollout 预测的历史帧（detach）
        #   [:, k]            = pred_x0_k（带梯度）
        #   [:, k+1:]         = future 帧占位
        #     - use_gt_future_frames=True : 用 GT 占位（旧行为）
        #     - use_gt_future_frames=False: 用 zeros 占位（纯蒸馏，避免 GT 颜色泄漏）
        # 注：因为模型是 causal，理论上后续帧不会影响第 k 帧的 score；
        # 但 cross-attn mask 维度按 num_prompts 构建，必须保持张量形状 [B, T, ...] 完整。
        if self.use_gt_future_frames:
            clean_full = image_latents.clone().detach().to(dtype=self.dtype)
        else:
            clean_full = torch.zeros_like(image_latents, dtype=self.dtype)
            # 保留首帧 GT 作为 reference image（与推理路径一致）
            clean_full[:, 0:1] = image_latents[:, 0:1].detach().to(dtype=self.dtype)

        B_h = history_pred.shape[0]
        B_eff = min(B, B_h)
        clean_full[:B_eff, :n_write] = history_pred[:B_eff, :n_write].detach()
        clean_full[:, target_turn_k:target_turn_k + 1] = pred_x0_k

        # Step 4: 编码 cond / uncond
        # 模型在 independent_first_frame=True 时，cross-attn mask 的 KV 维度按
        # num_prompts = (T - 1) // num_frame_per_block 构建（首帧 reference image 不分配 prompt）。
        # 因此整段 forward（_forward_train 路径）必须严格对齐 num_prompts，否则会触发
        # "block_mask was created for ... but got q_len/kv_len" 报错。
        nfp = int(getattr(self.generator.model, "num_frame_per_block", 1) or 1)
        num_prompts = max(1, (T - 1) // nfp)
        cond_dict = {"prompt_embeds": self._encode_prompts_for_k(prompts_per_turn, k=num_prompts)}
        uncond_dict = {"prompt_embeds": self._encode_uncond_for_k(B, k=num_prompts)}

        # Step 5: DMD KL gradient
        grad_k, kl_log = self._compute_kl_grad(
            clean_full=clean_full.detach(),  # 注意：clean_full 本身不需要梯度
            target_turn_k=target_turn_k,
            cond_dict=cond_dict,
            uncond_dict=uncond_dict,
            estimated_clean_k=pred_x0_k.detach(),
        )

        # Step 6: DMD loss = 0.5 * MSE(pred_x0_k, (pred_x0_k - grad_k).detach())
        target = (pred_x0_k.double() - grad_k.double()).detach()
        dmd_loss = 0.5 * F.mse_loss(pred_x0_k.double(), target, reduction="mean")

        log_dict = {
            "generator_loss": dmd_loss.detach().item(),
            "denoised_from": denoised_from,
            "denoised_to": denoised_to,
            "target_turn_k": target_turn_k,
        }
        log_dict.update(kl_log)

        # 释放 KV cache
        del kv_cache
        return dmd_loss, log_dict

    # ------------------------------------------------------------------
    # Critic loss（fake_score denoising loss）
    # ------------------------------------------------------------------
    def critic_step(
        self,
        image_latents: torch.Tensor,
        prompts_per_turn: List[List[str]],
        target_turn_k: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Critic（fake_score）训练一步：
          1. no_grad self-rollout 出 history + 第 k turn 的 pred_x0_k
          2. 构造 clean_full（pred_x0_k 放在第 k 帧）
          3. 对第 k 帧加噪，fake_score forward -> flow loss

        Returns:
            loss: scalar
            log_dict: 日志
        """
        B, T, C, H, W = image_latents.shape

        # 防御：确保 target_turn_k 在本 rank 的 [1, T-1] 范围内。
        # 正常情况下 _sample_target_turn_k 已用本 rank 的 T 采样，这里只是兜底。
        if T <= 1:
            # 退化场景：只有首帧，无法做 critic
            zero = torch.zeros([], device=self.device, dtype=self.dtype, requires_grad=True)
            return zero, {"critic_loss": 0.0, "critic_timestep": 0.0, "skipped": 1}
        target_turn_k = int(max(1, min(target_turn_k, T - 1)))

        with torch.no_grad():
            # self-rollout 到第 k turn（包含 k 本身）
            history_pred, kv_cache = self._self_rollout_history(
                image_latents=image_latents,
                prompts_per_turn=prompts_per_turn,
                target_turn_k=target_turn_k + 1,  # rollout 到 k（含）
            )

            # ---- 防御 + 诊断 ----
            # 期望 history_pred 形状: [B, k_hist=target_turn_k+1, C, H, W]
            # 若 dim 不对，先打印一次形状再尝试修正，避免后续 broadcast 错位。
            if history_pred.dim() != 5 or history_pred.shape[1] != target_turn_k + 1:
                try:
                    rank = dist.get_rank() if dist.is_initialized() else 0
                except Exception:
                    rank = 0
                if rank == 0:
                    print(
                        f"[critic_step] WARN history_pred shape={tuple(history_pred.shape)} "
                        f"target_turn_k={target_turn_k} expected k_hist={target_turn_k + 1} "
                        f"image_latents shape={tuple(image_latents.shape)}",
                        flush=True,
                    )
                # 4D 兜底：补回 batch 维
                if history_pred.dim() == 4:
                    history_pred = history_pred.unsqueeze(0)

            k_hist = history_pred.shape[1]
            # n_write 必须同时不超过 history_pred 的 k_hist 和 image_latents 的 T
            n_write = min(k_hist, target_turn_k + 1, T)
            pred_idx = min(target_turn_k, k_hist - 1)

            # history_pred[:, pred_idx] 就是 generator 对第 k 帧的预测
            pred_x0_k = history_pred[:, pred_idx:pred_idx + 1]  # [B, 1, C, H, W]

            # 构造 clean_full
            # 布局同 generator_step：首帧 GT + 1..k 用 self-rollout + k+1.. 占位
            if self.use_gt_future_frames:
                clean_full = image_latents.clone().detach().to(dtype=self.dtype)
            else:
                clean_full = torch.zeros_like(image_latents, dtype=self.dtype)
                clean_full[:, 0:1] = image_latents[:, 0:1].detach().to(dtype=self.dtype)
            # 同时确保 batch 维一致（若 history_pred batch 与 image_latents 不一致，做截断）
            B_h = history_pred.shape[0]
            B_eff = min(B, B_h)
            clean_full[:B_eff, :n_write] = history_pred[:B_eff, :n_write].detach()

            # 对第 k 帧采样 timestep 并加噪
            critic_t = torch.randint(self.min_step, self.max_step, [B], device=self.device, dtype=torch.long)
            if self.timestep_shift > 1.0:
                critic_t_f = self.timestep_shift * (critic_t.float() / 1000.0) / (
                    1.0 + (self.timestep_shift - 1.0) * (critic_t.float() / 1000.0)
                ) * 1000.0
                critic_t = critic_t_f.long().clamp(self.min_step, self.max_step - 1)

            timestep_full = torch.zeros([B, T], device=self.device, dtype=torch.long)
            timestep_full[:, target_turn_k] = critic_t

            critic_noise = torch.randn([B, C, H, W], device=self.device, dtype=self.dtype)
            noisy_full = clean_full.clone()
            noisy_full[:, target_turn_k] = self.scheduler.add_noise(
                clean_full[:, target_turn_k],
                critic_noise,
                critic_t,
            )

        # 编码 cond
        # 与 _compute_kl_grad 一致：fake_score 走 _forward_train 路径，cross-attn mask
        # 的 KV 维度按 num_prompts = (T-1)//num_frame_per_block 构建。
        nfp = int(getattr(self.generator.model, "num_frame_per_block", 1) or 1)
        num_prompts = max(1, (T - 1) // nfp)
        cond_dict = {"prompt_embeds": self._encode_prompts_for_k(prompts_per_turn, k=num_prompts)}

        # fake_score forward（带梯度）
        flow_pred_full, x0_pred_full = self.fake_score(
            noisy_image_or_video=noisy_full,
            conditional_dict=cond_dict,
            timestep=timestep_full,
        )

        # 只取第 k 帧计算 denoising loss
        flow_pred_k = flow_pred_full[:, target_turn_k]  # [B, C, H, W]
        x0_pred_k = x0_pred_full[:, target_turn_k]
        noisy_k = noisy_full[:, target_turn_k]
        generated_k = clean_full[:, target_turn_k].detach()

        # flow loss
        flow_pred_for_loss = WanDiffusionWrapper._convert_x0_to_flow_pred(
            scheduler=self.scheduler,
            x0_pred=x0_pred_k,
            xt=noisy_k,
            timestep=critic_t,
        )

        critic_loss = self.denoising_loss_func(
            x=generated_k,
            x_pred=x0_pred_k,
            noise=critic_noise,
            noise_pred=None,
            alphas_cumprod=getattr(self.scheduler, "alphas_cumprod", None),
            timestep=critic_t,
            flow_pred=flow_pred_for_loss,
        )

        log_dict = {
            "critic_loss": critic_loss.detach().item(),
            "critic_timestep": critic_t.float().mean().item(),
            "target_turn_k": int(target_turn_k),
        }

        del kv_cache
        return critic_loss, log_dict

    # ------------------------------------------------------------------
    # 一次完整 train step（包含 generator + critic 决策）
    # ------------------------------------------------------------------
    def _sample_target_turn_k(self, num_turns_in_batch: int) -> int:
        """
        跨 rank 同步采样目标 turn k ∈ [1, T - 1]。

        约束：
          - 本训练脚本用 FSDP 包装 generator/fake_score；FSDP 每次 module
            forward 前 all-gather 参数、backward reduce-scatter 梯度，**要求
            所有 rank 严格调用相同次数、相同顺序的 forward**。rollout 阶段
            会循环 (k-1) 次调用 generator，所以**所有 rank 的 k 必须一致**。
          - 数据侧已通过 `require_exact_turns=True` 保证所有 batch 的
            num_turns == max_turns，因此各 rank 拿到的 T 必然相同，无需
            额外的 all-reduce(MIN) 来对齐 T 范围。

        逻辑：rank0 在 [1, T-1] 采样一个 k，再 broadcast 给所有 rank。
        """
        T = int(num_turns_in_batch)
        if T <= 1:
            return 1  # 退化场景：极端样本，理论上被 require_exact_turns 过滤掉
        k_t = torch.randint(1, T, (1,), device=self.device, dtype=torch.long)
        if dist.is_initialized():
            dist.broadcast(k_t, src=0)
        return int(k_t.item())

    def fwd_bwd_one_step(
        self,
        prepared_batch: Dict,
        train_generator: bool,
    ) -> Dict:
        """
        Forward + Backward 一次。
          - train_generator=True: 训练 generator（DMD loss）
          - train_generator=False: 训练 fake_score（denoising loss）
        """
        image_latents = prepared_batch["image_latents"]
        prompts_per_turn = prepared_batch["prompts_per_turn"]
        T = image_latents.shape[1]  # 总帧数（含首帧）

        # 同步采样目标 turn k
        target_turn_k = self._sample_target_turn_k(T)

        if train_generator:
            self.generator.train()
            self.fake_score.eval()
            self.real_score.eval()
            loss, log_dict = self.generator_step(
                image_latents=image_latents,
                prompts_per_turn=prompts_per_turn,
                target_turn_k=target_turn_k,
            )
            loss_scaled = loss / self.gradient_accumulation_steps
            loss_scaled.backward()
            log_dict["generator_loss_raw"] = loss.detach().item()
            return log_dict
        else:
            self.generator.eval()
            self.fake_score.train()
            self.real_score.eval()
            loss, log_dict = self.critic_step(
                image_latents=image_latents,
                prompts_per_turn=prompts_per_turn,
                target_turn_k=target_turn_k,
            )
            loss_scaled = loss / self.gradient_accumulation_steps
            loss_scaled.backward()
            log_dict["critic_loss_raw"] = loss.detach().item()
            return log_dict

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(self) -> None:
        if self.is_main_process:
            print(f"[Save] gathering distributed model states at step {self.step} ...")

        gen_sd = fsdp_state_dict(self.generator)
        critic_sd = fsdp_state_dict(self.fake_score)

        if self.generator_ema is not None:
            state = {
                "generator": gen_sd,
                "critic": critic_sd,
                "generator_ema": self.generator_ema.state_dict(),
                "step": self.step,
            }
        else:
            state = {
                "generator": gen_sd,
                "critic": critic_sd,
                "step": self.step,
            }

        if self.is_main_process:
            ckpt_dir = os.path.join(self.output_path, f"checkpoint_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "model.pt")
            torch.save(state, ckpt_path)
            print(f"[Save] saved to {ckpt_path}")

    # ------------------------------------------------------------------
    # 主训练循环
    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.config
        max_steps = int(cfg.max_steps)
        log_iters = int(getattr(cfg, "log_iters", 200))
        save_iters = int(getattr(cfg, "save_iters", getattr(cfg, "log_iters", 200)))
        gc_interval = int(getattr(cfg, "gc_interval", 1000))
        ema_start_step = int(getattr(cfg, "ema_start_step", 0))
        no_save = bool(getattr(cfg, "no_save", False))

        if self.is_main_process:
            print(f"[Train] start, max_steps={max_steps}, log_iters={log_iters}, save_iters={save_iters}")

        while self.step < max_steps:
            train_gen = (self.step % self.dfake_gen_update_ratio == 0)

            # ----------------- Generator 更新（按累计步） -----------------
            if train_gen:
                self.generator_optimizer.zero_grad(set_to_none=True)
                gen_logs: List[Dict] = []
                for accum_idx in range(self.gradient_accumulation_steps):
                    batch = next(self.dataloader)
                    prepared = self._prepare_batch(batch)
                    log_dict = self.fwd_bwd_one_step(prepared, train_generator=True)
                    gen_logs.append(log_dict)

                gen_grad_norm = self.generator.clip_grad_norm_(self.max_grad_norm_generator)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.generator)

                gen_loss_avg = sum(d["generator_loss_raw"] for d in gen_logs) / max(len(gen_logs), 1)
            else:
                gen_loss_avg = None
                gen_grad_norm = None

            # ----------------- Critic 更新（每个 step 都跑） -----------------
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_logs: List[Dict] = []
            for accum_idx in range(self.gradient_accumulation_steps):
                batch = next(self.dataloader)
                prepared = self._prepare_batch(batch)
                log_dict = self.fwd_bwd_one_step(prepared, train_generator=False)
                critic_logs.append(log_dict)

            critic_grad_norm = self.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
            self.critic_optimizer.step()
            critic_loss_avg = sum(d["critic_loss_raw"] for d in critic_logs) / max(len(critic_logs), 1)

            # ----------------- step 推进 -----------------
            self.step += 1

            # 延迟初始化 EMA（如果配置在某 step 之后才启用）
            if self.generator_ema is None and (self.step >= ema_start_step) and float(getattr(cfg, "ema_weight", 0.0) or 0.0) > 0.0:
                self.generator_ema = EMA_FSDP(self.generator, decay=float(cfg.ema_weight))

            # ----------------- Logging -----------------
            if self.is_main_process and (self.step % log_iters == 0):
                if self.writer is not None:
                    # ---- generator 相关（仅在该 step 训了 generator 才有）----
                    if gen_loss_avg is not None:
                        self.writer.add_scalar("train/generator_loss", gen_loss_avg, self.step)
                        if gen_grad_norm is not None:
                            try:
                                gn = gen_grad_norm.item() if hasattr(gen_grad_norm, "item") else float(gen_grad_norm)
                                self.writer.add_scalar("train/generator_grad_norm", gn, self.step)
                            except Exception:
                                pass
                        # 来自 _compute_kl_grad 的辅助指标（DMD 真正的"距离"信号）
                        if len(gen_logs) > 0:
                            last_g = gen_logs[-1]
                            if "dmd_gradient_norm" in last_g:
                                self.writer.add_scalar(
                                    "train/dmd_gradient_norm",
                                    float(last_g["dmd_gradient_norm"]),
                                    self.step,
                                )
                            if "score_timestep" in last_g:
                                self.writer.add_scalar(
                                    "train/score_timestep",
                                    float(last_g["score_timestep"]),
                                    self.step,
                                )
                            if "denoised_from" in last_g:
                                self.writer.add_scalar(
                                    "train/denoised_from",
                                    float(last_g["denoised_from"]),
                                    self.step,
                                )
                            if "denoised_to" in last_g:
                                self.writer.add_scalar(
                                    "train/denoised_to",
                                    float(last_g["denoised_to"]),
                                    self.step,
                                )

                    # ---- critic 相关（每个 step 都训）----
                    self.writer.add_scalar("train/critic_loss", critic_loss_avg, self.step)
                    try:
                        cgn = critic_grad_norm.item() if hasattr(critic_grad_norm, "item") else float(critic_grad_norm)
                        self.writer.add_scalar("train/critic_grad_norm", cgn, self.step)
                    except Exception:
                        pass

                    # ---- 辅助统计（按 critic_logs 的均值，比单点更稳定）----
                    if len(critic_logs) > 0:
                        # target_turn_k：critic_step 现在已正确写入 log_dict，
                        # 取本次 accumulation 全部 batch 的均值更能反映分布。
                        ks = [int(d.get("target_turn_k", 0)) for d in critic_logs]
                        self.writer.add_scalar(
                            "train/target_turn_k",
                            sum(ks) / max(len(ks), 1),
                            self.step,
                        )
                        # critic_t 均值（应该围绕 (min+max)/2 ~ 500 波动）
                        cts = [
                            float(d.get("critic_timestep", 0.0))
                            for d in critic_logs if "critic_timestep" in d
                        ]
                        if cts:
                            self.writer.add_scalar(
                                "train/critic_timestep",
                                sum(cts) / len(cts),
                                self.step,
                            )

                msg = f"[Step {self.step}] critic_loss={critic_loss_avg:.4f}"
                if gen_loss_avg is not None:
                    msg = f"[Step {self.step}] gen_loss={gen_loss_avg:.4f} | critic_loss={critic_loss_avg:.4f}"
                print(msg)

            # ----------------- Save -----------------
            if (not no_save) and (self.step % save_iters == 0):
                torch.cuda.empty_cache()
                self.save_checkpoint()
                torch.cuda.empty_cache()

            # ----------------- GC -----------------
            if self.step % gc_interval == 0:
                if self.is_main_process:
                    logging.info("[GC] running garbage collection.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                cur_t = time.time()
                if self.previous_time is not None and self.writer is not None and self.step % log_iters == 0:
                    self.writer.add_scalar("train/iter_time", cur_t - self.previous_time, self.step)
                self.previous_time = cur_t