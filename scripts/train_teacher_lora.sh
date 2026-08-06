#!/usr/bin/env bash
set -euo pipefail
conda run --no-capture-output -n vggtodistill3r \
  python train_teacher_lora.py --config configs/teacher_lora_finetune.yaml "$@"
