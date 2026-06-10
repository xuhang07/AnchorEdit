#!/usr/bin/env python3
"""
Entry point for multi-turn causal I2V ODE-init pretraining.

用途：在正式走 DMD 蒸馏（[`train_multi_turn_dmd.py`](train_multi_turn_dmd.py:1)）
之前，先用 ODE regression loss 把已经训练好的多轮 causal 模型预热到少步 timestep
（默认 [1000, 750, 500, 250]）上，产出的 ckpt 再作为 DMD 阶段 generator/fake_score
的初始化权重（填到 DMD config 的 `checkpoint.pretrained_pt` 里）。

配套：
  - [`trainer/multi_turn_ode_init.py`](trainer/multi_turn_ode_init.py:1)
  - [`configs/multi_turn_ode_init_config.yaml`](configs/multi_turn_ode_init_config.yaml:1)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

sys.path.append(str(Path(__file__).parent))

from trainer.multi_turn_ode_init import MultiTurnODEInitTrainer


def load_config(config_path: str) -> OmegaConf:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)
    # 保持与 DMD trainer 一致的默认开关
    if "i2v" not in cfg:
        cfg.i2v = False
    if "causal" not in cfg:
        cfg.causal = True
    return cfg


def merge_configs(base: OmegaConf, override: OmegaConf) -> OmegaConf:
    return OmegaConf.merge(base, override)


def setup_environment(config: OmegaConf) -> None:
    os.makedirs(config.logdir, exist_ok=True)
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn ODE-init pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", type=str, default=None, help="Optional override YAML")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to override generator_ckpt")
    parser.add_argument("--pretrained_pt", type=str, default=None,
                        help="Override checkpoint.pretrained_pt")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        config = merge_configs(config, load_config(args.override))

    if args.data_path:
        config.data.data_path = args.data_path
    if args.logdir:
        config.logdir = args.logdir
        config.tensorboard_logdir = os.path.join(args.logdir, "tensorboard")
    if args.pretrained_pt:
        if "checkpoint" not in config:
            config.checkpoint = OmegaConf.create({})
        config.checkpoint.pretrained_pt = args.pretrained_pt
    if args.resume:
        if "checkpoint" not in config:
            config.checkpoint = OmegaConf.create({})
        config.checkpoint.generator_ckpt = args.resume

    setup_environment(config)

    if "RANK" in os.environ and os.environ["RANK"] == "0":
        print("ODE-init configuration:")
        print(OmegaConf.to_yaml(config))

    trainer = MultiTurnODEInitTrainer(config)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("[Train] interrupted by user.")
    except Exception as e:
        print(f"[Train] failed: {e}")
        raise
    finally:
        if hasattr(trainer, "step") and trainer.step > 0:
            print("[Train] saving final checkpoint ...")
            try:
                trainer.save_checkpoint()
            except Exception as e:
                print(f"[Train] save final checkpoint failed: {e}")


if __name__ == "__main__":
    main()