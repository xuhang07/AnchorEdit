#!/bin/bash

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NNODES=${NNODES:-"1"}
NODE_RANK=${NODE_RANK:-"0"}

export PYTHONPATH=$PYTHONPATH:$(pwd)
export TOKENIZERS_PARALLELISM=false

torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    train_multi_turn_dmd.py \
    --config configs/multi_turn_dmd_distill_config.yaml

train_ret=$?
echo "DMD distillation done."
exit $train_ret
