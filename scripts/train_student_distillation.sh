#!/usr/bin/env bash
set -euo pipefail
python train_student_distillation.py --config configs/student_distillation.yaml "$@"
