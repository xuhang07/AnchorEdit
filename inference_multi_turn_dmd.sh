#!/bin/bash
set -euo pipefail

# =========================
# DMD 4-step distilled inference
# 与 trainer/multi_turn_dmd.py 的 _self_rollout_history 流程严格对齐
# =========================

CONFIG="configs/multi_turn_dmd_distill_config.yaml"
CKPT_BASE_DIR="./logs/multi_turn_dmd_4step_14b/ckpt"
VALIDATION_ROOT="/path/to/validation_samples_multi_turn"
OUTPUT_BASE_DIR="./outputs"
OUTPUT_PREFIX="wan_multi_turn_dmd_4step"

# =========================
# DMD 推理参数（默认从 config 读取，这里覆盖）
# =========================
MAX_SAMPLES=0                                    # 0 表示全部
MAX_TURN=5
LANG="en"
DENOISING_STEP_LIST="1000,750,500,250"            # 必须与训练 config 一致
TIMESHIFT=1.0                                     # DMD 默认 timestep_shift=1.0
GUIDANCE_SCALE=1.0                                # generator 默认不开 CFG（DMD 蒸馏后已内化 teacher CFG）
SEED=42
USE_EMA=0                                         # 1: 加载 generator_ema 权重

# =========================
# 单机多卡
# =========================
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29551}

# checkpoint 循环区间（DMD save_iters=100，从 100 开始扫）
START_STEP=100
END_STEP=2000
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
    echo "Running DMD 4-step distilled inference"
    echo "Config: ${CONFIG}"
    echo "Checkpoint: ${CHECKPOINT_PATH}"
    echo "Validation Root: ${VALIDATION_ROOT}"
    echo "Output Directory: ${CURRENT_OUTPUT_DIR}"
    echo "Lang=${LANG} MaxTurn=${MAX_TURN} MaxSamples=${MAX_SAMPLES}"
    echo "denoising_step_list=${DENOISING_STEP_LIST}"
    echo "timeshift=${TIMESHIFT} guidance=${GUIDANCE_SCALE} use_ema=${USE_EMA}"
    echo "NPROC_PER_NODE=${NPROC_PER_NODE} MASTER_PORT=${MASTER_PORT}"
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
        inference_multi_turn_dmd.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --validation_root "${VALIDATION_ROOT}" \
        --output_dir "${CURRENT_OUTPUT_DIR}" \
        --denoising_step_list "${DENOISING_STEP_LIST}" \
        --timeshift "${TIMESHIFT}" \
        --guidance_scale "${GUIDANCE_SCALE}" \
        --lang "${LANG}" \
        --seed "${SEED}" \
        --device "auto" \
        "${MAX_SAMPLES_ARGS[@]}" \
        "${MAX_TURN_ARGS[@]}" \
        "${EMA_ARGS[@]}"
done

echo "All DMD 4-step distilled inference tasks completed!"