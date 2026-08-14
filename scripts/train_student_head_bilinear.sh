#!/usr/bin/env bash
set -euo pipefail
conda run --no-capture-output -n vggtodistill3r \
  python train_student_distillation.py \
  --config configs/student_distillation_head_bilinear.yaml "$@"
