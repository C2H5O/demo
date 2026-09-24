#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

OURS_PYTHON="${OURS_PYTHON:-/public/home/2024141520249/miniconda3/envs/vggtomast3r/bin/python}"
ENDODAV_PYTHON="${ENDODAV_PYTHON:-/public/home/2024141520249/miniconda3/envs/endodav/bin/python}"
ENDO3R_PYTHON="${ENDO3R_PYTHON:-/public/home/2024141520249/miniconda3/envs/endo3r/bin/python}"
HAMLYN_RESIZE_WORKERS="${HAMLYN_RESIZE_WORKERS:-4}"
HAMLYN_FRAME_CACHE_SIZE="${HAMLYN_FRAME_CACHE_SIZE:-96}"
export OURS_PYTHON ENDODAV_PYTHON ENDO3R_PYTHON
export HAMLYN_RESIZE_WORKERS HAMLYN_FRAME_CACHE_SIZE
HAMLYN_OUTPUT_ROOT="${HAMLYN_OUTPUT_ROOT:-outputs/hamlyn_eval}"
export HAMLYN_OUTPUT_ROOT

FORCE_ARGS=()
if [[ "${FORCE_INFERENCE:-0}" == "1" ]]; then
  FORCE_ARGS+=(--force-inference)
fi

echo "[preflight] Hamlyn, all weights, external repositories, Python environments, and CUDA"
"${OURS_PYTHON}" evaluate_hamlyn.py --preflight
mkdir -p "${HAMLYN_OUTPUT_ROOT}/logs"

echo "[1/4] Ours"
"${OURS_PYTHON}" evaluate_hamlyn.py --method ours --stage all "${FORCE_ARGS[@]}" \
  2>&1 | tee "${HAMLYN_OUTPUT_ROOT}/logs/ours.log"

echo "[2/4] Official DA3-Small"
"${OURS_PYTHON}" evaluate_hamlyn.py --method da3 --stage all "${FORCE_ARGS[@]}" \
  2>&1 | tee "${HAMLYN_OUTPUT_ROOT}/logs/da3.log"

echo "[3/4] EndoDAV"
bash scripts/eval_endodav.bash --stage all "${FORCE_ARGS[@]}" \
  2>&1 | tee "${HAMLYN_OUTPUT_ROOT}/logs/endodav.log"

echo "[4/4] Endo3R"
"${ENDO3R_PYTHON}" evaluate_hamlyn.py --method endo3r --stage all "${FORCE_ARGS[@]}" \
  2>&1 | tee "${HAMLYN_OUTPUT_ROOT}/logs/endo3r.log"

echo "[5/5] Collect summary"
"${OURS_PYTHON}" evaluate_hamlyn.py --collect-summary
