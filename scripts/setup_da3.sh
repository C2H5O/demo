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
# The upstream package declares visualization/export/server dependencies such as
# open3d and pycolmap as mandatory.  This experiment imports only the DA3-Small
# network components, so install the pinned source without those optional tools.
python -m pip install --no-deps -e "${DA3_ROOT}"
python -m pip install addict e3nn einops omegaconf safetensors
if ! python -c "import numpy; assert numpy.__version__ == '1.26.4', (numpy.__version__, numpy.__file__)"; then
  # Repair mixed pip/Conda or stale user-site installations deterministically.
  python -m pip install --force-reinstall --no-cache-dir "numpy==1.26.4"
fi
python -c "import numpy; assert numpy.__version__ == '1.26.4', (numpy.__version__, numpy.__file__); print('NumPy 1.26.4:', numpy.__file__)"
if ! python "${PROJECT_ROOT}/scripts/verify_numpy_abi.py"; then
  # OpenCV's four wheel variants all provide the same cv2 namespace. Remove
  # competing variants and install exactly one server/headless build.
  python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true
  python -m pip install --force-reinstall --no-cache-dir \
    "numpy==1.26.4" "opencv-python-headless==4.10.0.84"
fi
python "${PROJECT_ROOT}/scripts/verify_numpy_abi.py"
python -c "from depth_anything_3.cfg import create_object; from depth_anything_3.model.da3 import DepthAnything3Net; from depth_anything_3.model.dinov2.dinov2 import DinoV2; from depth_anything_3.model.dualdpt import DualDPT; from depth_anything_3.model.cam_enc import CameraEnc; from depth_anything_3.model.cam_dec import CameraDec; from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri; print('Depth-Anything-3 Small model imports OK')"
