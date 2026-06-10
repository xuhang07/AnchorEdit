#!/bin/bash
set -euo pipefail

# =========================
# 基础路径配置（按需修改）
# =========================
CONFIG="configs/multi_turn_i2v_config_14b_1e5.yaml"
CKPT_BASE_DIR="./logs/multi_turn_i2v/ckpt"
VALIDATION_ROOT="/path/to/validation_samples_multi_turn"
OUTPUT_BASE_DIR="./outputs"
OUTPUT_PREFIX="wan_multi_turn_validation"

# =========================
# 推理参数
# =========================
MAX_SAMPLES=0            # 0 表示不过滤，读取全部样本
MAX_TURN=5               # 单条样本最多推理多少轮（受 instructions 长度截断）
LANG="en"                # en / zh
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=5.0
TIMESHIFT=8.0
SEED=42
USE_EMA=0                # 1: 加 --use_ema, 0: 不加

# =========================
# 单机多卡参数
# =========================
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29541}

# checkpoint 循环区间
START_STEP=100
END_STEP=1500
STEP_GAP=100

if [ "${MAX_SAMPLES}" -gt 0 ]; then
    MAX_SAMPLES_ARGS=(--max_samples "${MAX_SAMPLES}")
else
    MAX_SAMPLES_ARGS=()
fi

if [ -n "${MAX_TURN}" ] && [ "${MAX_TURN}" -gt 0 ]; then
    MAX_TURN_ARGS=(--max_turn "${MAX_TURN}")
else
    MAX_TURN_ARGS=()
fi

if [ "${USE_EMA}" -eq 1 ]; then
    EMA_ARGS=(--use_ema)
else
    EMA_ARGS=()
fi

for STEP in $(seq ${START_STEP} ${STEP_GAP} ${END_STEP}); do
    PADDED_STEP=$(printf "%06d" "${STEP}")
    CHECKPOINT_PATH="${CKPT_BASE_DIR}/checkpoint_${PADDED_STEP}/model.pt"
    CURRENT_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${OUTPUT_PREFIX}_${STEP}"

    echo "=========================================================="
    echo "Running WAN multi-turn validation inference"
    echo "Config: ${CONFIG}"
    echo "Checkpoint: ${CHECKPOINT_PATH}"
    echo "Validation Root: ${VALIDATION_ROOT}"
    echo "Output Directory: ${CURRENT_OUTPUT_DIR}"
    echo "Lang: ${LANG}, MaxTurn: ${MAX_TURN}, MaxSamples: ${MAX_SAMPLES}"
    echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
    echo "MASTER_PORT: ${MASTER_PORT}"
    echo "=========================================================="

    if [ ! -f "${CHECKPOINT_PATH}" ]; then
        echo "[Warning] checkpoint not found, skip: ${CHECKPOINT_PATH}"
        continue
    fi

    mkdir -p "${CURRENT_OUTPUT_DIR}"

    torchrun \
        --standalone \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        inference_multi_turn_i2v_validation.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --validation_root "${VALIDATION_ROOT}" \
        --output_dir "${CURRENT_OUTPUT_DIR}" \
        --num_inference_steps "${NUM_INFERENCE_STEPS}" \
        --guidance_scale "${GUIDANCE_SCALE}" \
        --timeshift "${TIMESHIFT}" \
        --lang "${LANG}" \
        --seed "${SEED}" \
        --device "auto" \
        "${MAX_SAMPLES_ARGS[@]}" \
        "${MAX_TURN_ARGS[@]}" \
        "${EMA_ARGS[@]}"
done

echo "All WAN multi-turn validation inference tasks completed!"