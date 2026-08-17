#!/usr/bin/env bash
set -euo pipefail
# Optional retained Endo3R comparison; VDA is the experiment's default test.
conda run --no-capture-output -n vggtodistill3r \
  python evaluate.py \
  --config configs/student_distillation_head_bilinear.yaml "$@"
