<p align="center">
<h1 align="center">AnchorEdit</h1>
<h3 align="center">Maintaining Temporal Consistency in Multi-turn Imting via Causal Memory</h3>
</p>

---

AnchorEdit is the first autoregressive (AR) diffusion-based framework designed specifically for high-resolution, long-term multi-turn image editing. It bridges the gap between video priors and causal inference through a three-stage training curriculum and a novel streaming inference pipeline with memory mechanisms.

## Key Features

- **Causal AR Multi-turn Editing**: Frame-level causal attention enabling true sequential editing without bidirectional information leakage
- **Three-stage Training**: Single-turn pretraining → Causal multi-turn training with self-rollout → Few-step consistency distillation
- **Long-chain Streaming Inference**: Sink frame + sliding window KV cache + strided RoPE indexing for stable 10+ turn editing
- **High Resolution**: Native 1024×1024 generation with aspect ratio bucketing

## Architecture

```
Stage 1: Single-turn Edit Pretrain
  - Identity mapping learning (null-instruction reconstruction)
  - Expanded RoPE (stride s between source/target frames)

Stage 2: Causal Multi-turn Training
  - Frame-level causal attention mask
  - Synthetic degradation injection (color-shift, blur, JPEG, etc.)
  - Self-rollout fine-tuning (replace GT history with model predictions)
  - Temporally-progressive loss weighting

Stage 3: Few-step Distillation
  - Consistency Distillation (CD) initialization
  - Distribution Matching Distillation (DMD) for 4-step generation
```

## Requirements

- NVIDIA GPU with at least 40 GB memory (A100 or H100 recommended)
- Linux operating system
- 64 GB RAM
- Python 3.10+

## Installation

```bash
conda create -n anchor_edit python=3.10 -y
conda activate anchor_edit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Model

Our framework is built on [WAN2.1-T2V-14B](https://github.com/Wan-Video/Wan2.1). Download the base model:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir-use-symlinks False --local-dir /path/to/Wan2.1-T2V-14B
```

## Training

### Stage 1 & 2: Single-turn Pretraining + Causal Multi-turn Training

```bash
# Stage 1/2 unified training (configure via YAML)
torchrun --nnodes=1 --nproc_per_node=8 \
  train_multi_turn_i2v.py \
  --config configs/multi_turn_i2v_config_14b_1e5.yaml
```

### Stage 3: Few-step Distillation

**Step 3a: Consistency Distillation (CD) initialization**
```bash
torchrun --nnodes=1 --nproc_per_node=8 \
  train_multi_turn_cd.py \
  --config configs/multi_turn_cd_config.yaml
```

**Step 3b: Distribution Matching Distillation (DMD)**
```bash
torchrun --nnodes=1 --nproc_per_node=8 \
  train_multi_turn_dmd.py \
  --config configs/multi_turn_dmd_distill_config.yaml
```

## Inference

### Multi-turn Inference (Standard, multi-step)

```bash
torchrun --nnodes=1 --nproc_per_node=1 \
  inference_multi_turn_i2v_validation.py \
  --config configs/multi_turn_i2v_config_14b_1e5.yaml \
  --checkpoint /path/to/stage2_checkpoint/model.pt \
  --validation_root /path/to/validation_data \
  --output_dir outputs/multi_turn_results \
  --num_inference_steps 50 \
  --use_ema
```

### Multi-turn Inference (DMD 4-step)

```bash
torchrun --nnodes=1 --nproc_per_node=1 \
  inference_multi_turn_dmd.py \
  --config configs/multi_turn_dmd_distill_config.yaml \
  --checkpoint /path/to/dmd_checkpoint/model.pt \
  --validation_root /path/to/validation_data \
  --output_dir outputs/dmd_4step_results \
  --use_ema
```

## Data Preparation

The training data should be organized as WebDataset tar files. Each sample contains:
- Numbered images (`0.jpg`, `1.jpg`, ..., `N.jpg`): source image + N edited results
- A JSON file with editing instructions per turn

```json
{
  "instructions": [
    {"zh": "将背景改为蓝色", "en": "Change the background to blue"},
    {"zh": "添加一顶帽子", "en": "Add a hat"}
  ]
}
```

Use `scripts/prepare_multi_turn_dataset.py` and `scripts/pack_samples_to_tar.py` for data preparation.

## Project Structure

```
├── train_multi_turn_i2v.py          # Stage 1 & 2 training entry
├── train_multi_turn_cd.py           # Stage 3 CD training entry
├── train_multi_turn_dmd.py          # Stage 3 DMD training entry
├── train_multi_turn_ode_init.py     # Stage 3 ODE init entry
├── inference_multi_turn_i2v.py      # Multi-turn standard inference
├── inference_multi_turn_dmd.py      # Multi-turn DMD 4-step inference
├── inference_multi_turn_i2v_validation.py  # Validation inference
├── configs/                         # Training configurations
├── trainer/                         # Trainer implementations
│   ├── multi_turn_i2v.py           # Stage 1/2 trainer
│   ├── multi_turn_cd.py            # CD trainer
│   ├── multi_turn_dmd.py           # DMD trainer
│   └── multi_turn_ode_init.py      # ODE init trainer
├── pipeline/                        # Inference pipelines
│   └── multi_turn_inference.py     # Multi-turn I2V pipeline
├── utils/                           # Utilities
│   ├── multi_turn_dataset.py       # Multi-turn data loading
│   ├── multi_turn_tokenizer.py     # Text/image tokenization
│   ├── wan_wrapper.py              # WAN model wrappers
│   ├── scheduler.py                # Flow matching scheduler
│   ├── distributed.py              # FSDP utilities
│   └── ...
├── wan/                             # WAN2.1 model backbone
├── data_generation/                 # Data preparation tools
└── scripts/                         # Helper scripts
```

## Acknowledgements

This codebase is built on top of [Self-Forcing](https://github.com/self-forcing/self-forcing) and [Wan2.1](https://github.com/Wan-Video/Wan2.1).

## Citation

```bibtex
@article{anchoredit2026,
  title={AnchorEdit: Maintaining Temporal Consistency in Multi-turn Image Editing via Causal Memory},
  year={2026}
}
```
