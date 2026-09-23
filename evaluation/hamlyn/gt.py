"""Hamlyn uint16-millimeter ground-truth loading."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from evaluation.hamlyn.constants import EVALUATION_RESOLUTION_HW, HAMLYN_GT_SCALE


def read_hamlyn_gt_uint16(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("Could not read Hamlyn GT: {}".format(path))
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("Hamlyn GT must be 2D: {} has {}".format(path, depth.shape))
    if depth.dtype != np.uint16:
        raise ValueError(
            "Hamlyn GT must be uint16 direct millimeters: {} has dtype {}, min {}, max {}"
            .format(path, depth.dtype, depth.min(), depth.max())
        )
    return depth


def load_hamlyn_gt_depth(
    path: Path,
    target_shape: Tuple[int, int] = EVALUATION_RESOLUTION_HW,
) -> np.ndarray:
    depth_m = read_hamlyn_gt_uint16(path).astype(np.float32) * HAMLYN_GT_SCALE
    if tuple(depth_m.shape) == tuple(target_shape):
        return depth_m
    return cv2.resize(
        depth_m,
        (int(target_shape[1]), int(target_shape[0])),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32, copy=False)
