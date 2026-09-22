#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for variant in A B C D; do
  "${PYTHON_BIN}" evaluate_crossclip_projection.py \
    --config "configs/baselines/H_${variant}.yaml" \
    --split test \
    --protocol vda
done
