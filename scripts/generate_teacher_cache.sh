#!/usr/bin/env bash
set -euo pipefail
python generate_teacher_cache.py --config configs/student_distillation.yaml "$@"
