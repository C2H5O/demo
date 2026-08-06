#!/usr/bin/env bash
set -euo pipefail
conda run --no-capture-output -n vggtodistill3r \
  python evaluate.py --config configs/student_distillation.yaml "$@"
