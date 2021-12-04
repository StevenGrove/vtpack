#!/usr/bin/env bash

set -x

CONFIG_DIR=$(echo $0 | cut -d . -f1)
EXP_DIR=${CONFIG_DIR//configs/logs}
PY_ARGS=${@:1}

python -u main.py \
    --dist-eval \
    --output_dir ${EXP_DIR} \
    --model deit_dge_s124_tiny_patch16_256 \
    --batch-size 32 \
    --input-size 256 \
    --warmup-epochs 20 \
    --dge-enable \
    --dge-budget 0.5 \
    --dge-lr-scale 1.0 \
    ${PY_ARGS}
