#!/usr/bin/env bash

set -x

CONFIG_DIR=$(echo $0 | cut -d . -f1)
EXP_DIR=${CONFIG_DIR//configs/logs}
PY_ARGS=${@:1}

python -u main.py \
    --dist-eval \
    --output_dir ${EXP_DIR} \
    --model dpvt_small_256 \
    --batch-size 32 \
    --input-size 256 \
    --warmup-epochs 20 \
    ${PY_ARGS}
