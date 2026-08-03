#!/usr/bin/env bash
set -euo pipefail
python evaluate.py --config configs/student_distillation.yaml "$@"
