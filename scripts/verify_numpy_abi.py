"""Verify that NumPy objects are accepted by the installed Torch and OpenCV binaries."""

from __future__ import annotations

import sys
from typing import Any, Dict

import cv2
import numpy as np
import torch


def verify_numpy_abi() -> Dict[str, Any]:
    diagnostics = {
        "python": sys.executable,
        "numpy_version": np.__version__,
        "numpy_file": np.__file__,
        "torch_version": torch.__version__,
        "cv2_version": cv2.__version__,
        "cv2_file": cv2.__file__,
    }
    image = np.zeros((8, 8), dtype=np.uint8)
    try:
        tensor = torch.from_numpy(image)
        if tensor.shape != (8, 8) or tensor.dtype is not torch.uint8:
            raise RuntimeError("Torch NumPy conversion returned an invalid tensor")
        count, labels, _, _ = cv2.connectedComponentsWithStats(image, 8)
        if count != 1 or labels.shape != image.shape:
            raise RuntimeError("OpenCV NumPy conversion returned invalid components")
    except Exception as error:
        raise RuntimeError(
            "NumPy binary ABI mismatch; diagnostics={}".format(diagnostics)
        ) from error
    return diagnostics


def main() -> None:
    print("NumPy/Torch/OpenCV ABI OK: {}".format(verify_numpy_abi()))


if __name__ == "__main__":
    main()
