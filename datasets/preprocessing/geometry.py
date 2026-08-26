"""Geometry-preserving image and depth transforms.

The canonical 4:5 view is built with *contain + padding*, never anisotropic
resizing or an implicit crop.  Dataset entrypoints may replace this with a
published rectification/reprojection, but must pass its output to both image
branches here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

TEACHER_SIZE: Tuple[int, int] = (1280, 1024)  # (W, H)
STUDENT_SIZE: Tuple[int, int] = (560, 448)  # (W, H)


@dataclass(frozen=True)
class ContainTransform:
    source_size: tuple[int, int]
    canonical_size: tuple[int, int]
    scale: float
    pad_left: int
    pad_top: int


def _contain_size(source_size: tuple[int, int], target_size: tuple[int, int]) -> ContainTransform:
    sw, sh = source_size
    tw, th = target_size
    if sw <= 0 or sh <= 0:
        raise ValueError(f"invalid source size: {source_size}")
    scale = min(tw / sw, th / sh)
    rw, rh = max(1, round(sw * scale)), max(1, round(sh * scale))
    return ContainTransform(source_size, target_size, scale, (tw - rw) // 2, (th - rh) // 2)


def contain_rgb(image: Image.Image, target_size: tuple[int, int]) -> tuple[Image.Image, ContainTransform]:
    """Resize proportionally and pad black; the entire input FOV is retained."""
    image = image.convert("RGB")
    tx = _contain_size(image.size, target_size)
    rw = target_size[0] - 2 * tx.pad_left
    rh = target_size[1] - 2 * tx.pad_top
    # Rounding can leave one pixel on the far edge; PIL canvas accounts for it.
    resized = image.resize((rw, rh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, color=(0, 0, 0))
    canvas.paste(resized, (tx.pad_left, tx.pad_top))
    return canvas, tx


def make_rgb_pair(canonical: Image.Image) -> tuple[Image.Image, Image.Image, dict]:
    """Materialize teacher/student views from one canonical image, never separately."""
    teacher, teacher_tx = contain_rgb(canonical, TEACHER_SIZE)
    student, student_tx = contain_rgb(canonical, STUDENT_SIZE)
    return teacher, student, {
        "canonical_fov": "contain_pad_4_5",
        "teacher_transform": teacher_tx.__dict__,
        "student_transform": student_tx.__dict__,
    }


def resize_depth_valid_aware(depth: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour depth resize; zeros/NaNs cannot contaminate valid pixels."""
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, received {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0)
    safe = np.where(valid, depth, 0).astype(np.float32)
    value = Image.fromarray(safe, mode="F").resize(target_size, Image.Resampling.NEAREST)
    mask = Image.fromarray(valid.astype(np.uint8) * 255).resize(target_size, Image.Resampling.NEAREST)
    out = np.asarray(value, dtype=np.float32)
    return np.where(np.asarray(mask) > 0, out, 0.0).astype(np.float32)


def contain_depth_valid_aware(depth: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Apply the exact contain/pad geometry used for RGB, with invalid=0 intact."""
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, received {depth.shape}")
    tx = _contain_size((depth.shape[1], depth.shape[0]), target_size)
    rw = target_size[0] - 2 * tx.pad_left
    rh = target_size[1] - 2 * tx.pad_top
    resized = resize_depth_valid_aware(depth, (rw, rh))
    canvas = np.zeros((target_size[1], target_size[0]), dtype=np.float32)
    canvas[tx.pad_top:tx.pad_top + rh, tx.pad_left:tx.pad_left + rw] = resized
    return canvas
