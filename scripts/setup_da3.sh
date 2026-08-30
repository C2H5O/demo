#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DA3_ROOT="${PROJECT_ROOT}/external/Depth-Anything-3"
DA3_COMMIT="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

if [[ ! -d "${DA3_ROOT}/.git" ]]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "${DA3_ROOT}"
fi
(
  cd "${DA3_ROOT}"
  git fetch origin "${DA3_COMMIT}"
  git checkout --detach "${DA3_COMMIT}"
)
python -m pip install -e "${DA3_ROOT}"
python -c "import depth_anything_3; print('Depth-Anything-3 import OK')"
