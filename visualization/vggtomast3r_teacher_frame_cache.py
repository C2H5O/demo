"""Visualize a two-frame composition of independent teacher frame caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from PIL import Image

from datasets.teacher_frame_cache import compose_teacher_frame_caches
from utils.config import ensure_dir


def _adaptive_depth(depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[float]]:
    from visualization.scared_student import depth_to_magma

    values = depth[valid]
    if values.size == 0:
        raise RuntimeError("No valid teacher depths remain")
    low, high = np.percentile(values, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1e-6
    return depth_to_magma(depth, valid, float(low), float(high)), [float(low), float(high)]


def export_composed_teacher_frames(
    cache_root: Path,
    frames: Sequence[Dict[str, Any]],
    output_dir: Path,
    image_shape: tuple[int, int],
    expected_base_checkpoint: str,
    rgb: Optional[np.ndarray] = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    point_stride: int = 4,
) -> Path:
    # Keep CLI discovery/help lightweight; OpenCV is needed only when exporting.
    from visualization.scared_student import depth_to_magma, write_binary_ply

    if len(frames) != 2:
        raise ValueError("Teacher pair visualization requires exactly two frames")
    if min_depth >= max_depth:
        raise ValueError("min_depth must be smaller than max_depth")
    if point_stride <= 0:
        raise ValueError("point_stride must be positive")
    composed = compose_teacher_frame_caches(
        cache_root, frames, image_shape, expected_base_checkpoint
    )
    depth = composed["depth"].numpy()
    points = composed["xyz_local"].numpy()
    confidence = composed["confidence"].numpy()
    valid = composed["valid_mask"].numpy().astype(bool)
    valid &= np.isfinite(depth) & np.isfinite(points).all(axis=-1)
    valid &= (depth >= min_depth) & (depth <= max_depth)
    if rgb is not None and tuple(rgb.shape) != (2,) + image_shape + (3,):
        raise ValueError("RGB shape {} does not match composed frames".format(rgb.shape))

    fixed = np.stack(
        [depth_to_magma(depth[i], valid[i], min_depth, max_depth) for i in range(2)]
    )
    adaptive_values = [_adaptive_depth(depth[i], valid[i]) for i in range(2)]
    adaptive = np.stack([value[0] for value in adaptive_values])
    adaptive_ranges = [value[1] for value in adaptive_values]
    confidence_color = np.stack(
        [depth_to_magma(confidence[i], valid[i], 0.0, 1.0) for i in range(2)]
    )
    output = ensure_dir(output_dir)
    for index, suffix in enumerate(("a", "b")):
        Image.fromarray(fixed[index]).save(output / "depth_{}_fixed.png".format(suffix))
        Image.fromarray(adaptive[index]).save(output / "depth_{}_adaptive.png".format(suffix))
        Image.fromarray(confidence_color[index]).save(output / "confidence_{}.png".format(suffix))
        np.save(output / "depth_{}_local.npy".format(suffix), depth[index])
        frame_rgb = rgb[index] if rgb is not None else np.full((*image_shape, 3), 127, dtype=np.uint8)
        if rgb is not None:
            Image.fromarray(frame_rgb).save(output / "rgb_{}.png".format(suffix))
        sampled = np.zeros(image_shape, dtype=bool)
        sampled[::point_stride, ::point_stride] = True
        point_valid = valid[index] & sampled
        write_binary_ply(
            output / "teacher_camera_{}_local.ply".format(suffix),
            points[index][point_valid],
            frame_rgb[point_valid],
        )
    panels = []
    if rgb is not None:
        panels.extend((rgb[0], rgb[1]))
    panels.extend((fixed[0], fixed[1], adaptive[0], adaptive[1]))
    Image.fromarray(np.concatenate(panels, axis=1)).save(output / "teacher_frame_panel.png")
    report = {
        "cache_format": "independent frozen base-teacher frame caches",
        "cache_paths": composed["cache_paths"],
        "frame_names": composed["frame_names"],
        "frame_indices": composed["frame_indices"],
        "coordinate_convention": composed["coordinate_convention"],
        "fixed_depth_range": [min_depth, max_depth],
        "adaptive_depth_ranges_p1_p99": adaptive_ranges,
        "valid_fraction": [float(valid[i].mean()) for i in range(2)],
        "depth_stats": [
            {
                "min": float(depth[i][valid[i]].min()),
                "max": float(depth[i][valid[i]].max()),
                "mean": float(depth[i][valid[i]].mean()),
                "std": float(depth[i][valid[i]].std()),
            }
            for i in range(2)
        ],
        "warning": "The two point clouds are in separate camera-local coordinate systems and are not fused.",
    }
    (output / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


__all__ = ["export_composed_teacher_frames"]
