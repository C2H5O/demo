#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

ENDODAV_PYTHON="${ENDODAV_PYTHON:-/public/home/2024141520249/miniconda3/envs/endodav/bin/python}"
DISCOVERED_CUDA_LIBRARY_PATH="$(
  "${ENDODAV_PYTHON}" -m evaluation.hamlyn.cuda_libraries
)"
ENDODAV_CUDA_LIBRARY_PATH="${ENDODAV_CUDA_LIBRARY_PATH:-${DISCOVERED_CUDA_LIBRARY_PATH}}"

if [[ -n "${ENDODAV_CUDA_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${ENDODAV_CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "[EndoDAV CUDA libraries] ${ENDODAV_CUDA_LIBRARY_PATH:-system default}" >&2

if [[ "$#" -eq 0 ]]; then
  set -- --stage all
fi

exec "${ENDODAV_PYTHON}" evaluate_hamlyn.py --method endodav "$@"
