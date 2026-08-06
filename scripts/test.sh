#!/usr/bin/env bash
set -euo pipefail

conda run --no-capture-output -n vggtodistill3r \
  python scripts/verify_environment.py
conda run --no-capture-output -n vggtodistill3r \
  python -m pytest -q
