#!/usr/bin/env bash
set -euo pipefail
python train_teacher_lora.py --config configs/teacher_lora_finetune.yaml "$@"
