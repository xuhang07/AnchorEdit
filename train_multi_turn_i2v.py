#!/usr/bin/env python3
"""
Training script for multi-turn image-to-video editing.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
from omegaconf import OmegaConf
import torch

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False) 

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from trainer.multi_turn_i2v import MultiTurnI2VTrainer


def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to OmegaConf
    config = OmegaConf.create(config_dict)
    
    # Add derived configurations
    config.i2v = False  # This is I2V training
    
    return config


def merge_configs(base_config: OmegaConf, override_config: OmegaConf) -> OmegaConf:
    """Merge base config with override config."""
    return OmegaConf.merge(base_config, override_config)


def setup_environment(config: OmegaConf):
    """Setup training environment."""
    # Create output directory
    os.makedirs(config.logging.logdir, exist_ok=True)
    
    # Set environment variables for distributed training
    if 'RANK' in os.environ:
        # Already in distributed environment
        pass
    else:
        # Single GPU training
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'


def main():
    parser = argparse.ArgumentParser(description="Train multi-turn I2V model")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--override", 
        type=str, 
        default=None,
        help="Path to override configuration file"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=None,
        help="Override data path"
    )
    parser.add_argument(
        "--logdir", 
        type=str, 
        default=None,
        help="Override log directory"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="Local rank for distributed training"
    )
    
    args = parser.parse_args()
    
    # Load base configuration
    config = load_config(args.config)
    
    # Load override configuration if provided
    if args.override:
        override_config = load_config(args.override)
        config = merge_configs(config, override_config)
    
    # Apply command line overrides
    if args.data_path:
        config.data.data_path = args.data_path
    if args.logdir:
        config.logging.logdir = args.logdir
    if args.resume:
        config.checkpoint.resume_from = args.resume
    
    # Setup environment
    setup_environment(config)
    
    # Print configuration
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print("Training configuration:")
        print(OmegaConf.to_yaml(config))
    
    # Create trainer and start training
    trainer = MultiTurnI2VTrainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Save final checkpoint
        if hasattr(trainer, 'step') and trainer.step > 0:
            print("Saving final checkpoint...")
            trainer.save_checkpoint()


if __name__ == "__main__":
    main()