#!/usr/bin/env bash
set -euo pipefail
conda run --no-capture-output -n vggtodistill3r \
  python generate_teacher_cache.py --config configs/student_distillation.yaml "$@"
