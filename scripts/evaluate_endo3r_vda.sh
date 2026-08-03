#!/usr/bin/env bash
set -euo pipefail

python evaluate_endo3r_vda.py --config configs/endo3r_vda_baseline.yaml "$@"
