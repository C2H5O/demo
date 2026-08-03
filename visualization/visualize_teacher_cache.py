"""Visualize one offline VGGT-Omega teacher-cache clip.

Run from the project root:

    python -m visualization.visualize_teacher_cache --cache PATH_TO_CACHE.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from PIL import Image


REQUIRED_KEYS = (
    "xyz_local",
    "xyz_global",
    "conf_local",
    "conf_global",
    "valid_mask",
)


def _metadata(cache: Any) -> Dict[str, Any]:
    if "metadata_json" not in cache:
        return {}
    value = cache["metadata_json"]
    try:
        return json.loads(str(value.item()))
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Invalid metadata_json in teacher cache") from error


def _frame_names(cache: Any, frames: int) -> list[str]:
    if "frame_names" not in cache:
        return ["frame_{:04d}".format(index) for index in range(frames)]
    names = [str(value) for value in cache["frame_names"].tolist()]
    if len(names) != frames:
        raise RuntimeError(
            "frame_names length {} does not match cached frame count {}".format(
                len(names), frames
            )
        )
    return names


def _load_rgb_frames(
    metadata: Dict[str, Any],
    frames: int,
    height: int,
    width: int,
) -> Optional[np.ndarray]:
    paths = metadata.get("frame_paths")
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
        return None
    if len(paths) != frames:
        return None
    images = []
    for value in paths:
        path = Path(str(value))
        if not path.is_file():
            return None
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize(
                (width, height), resample=Image.Resampling.BILINEAR
            )
            images.append(np.asarray(rgb, dtype=np.uint8))
    return np.stack(images)


def _confidence_colors(
    confidence: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Teacher-cache visualization requires opencv-python. "
            "Install the project requirements first."
        ) from error
    colors = []
    for frame_index in range(confidence.shape[0]):
        uint8 = np.round(
            np.clip(confidence[frame_index], 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        bgr = cv2.applyColorMap(uint8, cv2.COLORMAP_VIRIDIS)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb[~valid[frame_index]] = 0
        colors.append(rgb)
    return np.stack(colors)


def _default_output_directory(cache_path: Path, output_root: Path) -> Path:
    # Expected cache hierarchy: split/dataset/keyframe/cache.npz.
    if len(cache_path.parents) >= 3:
        return (
            output_root
            / cache_path.parent.parent.parent.name
            / cache_path.parent.parent.name
            / cache_path.parent.name
            / cache_path.stem
        )
    return output_root / cache_path.stem


def visualize_teacher_cache(
    cache_path: Path,
    output_root: Path,
    point_stride: int = 4,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    confidence_threshold: Optional[float] = None,
    point_color: str = "rgb",
) -> Path:
    """Export 2D maps and an RGB/confidence-colored global PLY point cloud."""
    try:
        from visualization.scared_student import depth_to_magma, write_binary_ply
    except ImportError as error:
        raise RuntimeError(
            "Teacher-cache visualization dependencies are unavailable. "
            "Run `pip install -r requirements.txt`."
        ) from error
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError("Teacher cache not found: {}".format(cache_path))
    if point_stride <= 0:
        raise ValueError("point_stride must be positive")
    if point_color not in {"rgb", "confidence"}:
        raise ValueError("point_color must be 'rgb' or 'confidence'")

    with np.load(str(cache_path), allow_pickle=False) as cache:
        missing = [key for key in REQUIRED_KEYS if key not in cache]
        if missing:
            raise RuntimeError(
                "Teacher cache {} is missing keys {}".format(cache_path, missing)
            )
        xyz_local = cache["xyz_local"].astype(np.float32)
        xyz_global = cache["xyz_global"].astype(np.float32)
        confidence = cache["conf_global"].astype(np.float32)
        cached_valid = cache["valid_mask"].astype(bool)
        metadata = _metadata(cache)
        frame_names = _frame_names(cache, xyz_local.shape[0])
        cached_depth_range = (
            cache["teacher_depth_range"].astype(np.float32).tolist()
            if "teacher_depth_range" in cache
            else None
        )

    if xyz_local.ndim != 4 or xyz_local.shape[-1] != 3:
        raise RuntimeError(
            "xyz_local must have shape [T,H,W,3], got {}".format(
                tuple(xyz_local.shape)
            )
        )
    if xyz_global.shape != xyz_local.shape:
        raise RuntimeError("xyz_global shape does not match xyz_local")
    if confidence.shape != xyz_local.shape[:-1]:
        raise RuntimeError("conf_global shape does not match point maps")
    if cached_valid.shape != confidence.shape:
        raise RuntimeError("valid_mask shape does not match confidence")

    frames, height, width = confidence.shape
    depth = xyz_local[..., 2]
    if min_depth is None:
        min_depth = (
            float(cached_depth_range[0]) if cached_depth_range is not None else 0.0
        )
    if max_depth is None:
        max_depth = (
            float(cached_depth_range[1])
            if cached_depth_range is not None
            else float("inf")
        )
    if min_depth >= max_depth:
        raise ValueError("min_depth must be smaller than max_depth")

    valid = (
        cached_valid
        & np.isfinite(depth)
        & np.isfinite(xyz_global).all(axis=-1)
        & np.isfinite(confidence)
        & (depth >= min_depth)
        & (depth <= max_depth)
    )
    if not np.any(valid):
        raise RuntimeError("No valid teacher points remain in the selected depth range")

    rgb = _load_rgb_frames(metadata, frames, height, width)
    confidence_rgb = _confidence_colors(confidence, valid)
    valid_depth = depth[valid]
    color_low, color_high = np.percentile(valid_depth, (5.0, 95.0))

    output = _default_output_directory(cache_path, Path(output_root))
    directories = {
        name: output / name
        for name in (
            "rgb",
            "depth",
            "depth_color",
            "confidence",
            "confidence_color",
            "panels",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    for frame_index, frame_name in enumerate(frame_names):
        stem = Path(frame_name).stem or "frame_{:04d}".format(frame_index)
        depth_color = depth_to_magma(
            depth[frame_index],
            valid[frame_index],
            float(color_low),
            float(color_high),
        )
        np.save(
            directories["depth"] / "{}.npy".format(stem),
            depth[frame_index].astype(np.float32),
        )
        np.save(
            directories["confidence"] / "{}.npy".format(stem),
            confidence[frame_index].astype(np.float32),
        )
        Image.fromarray(depth_color).save(
            directories["depth_color"] / "{}.png".format(stem)
        )
        Image.fromarray(confidence_rgb[frame_index]).save(
            directories["confidence_color"] / "{}.png".format(stem)
        )
        panel_parts = [depth_color, confidence_rgb[frame_index]]
        if rgb is not None:
            Image.fromarray(rgb[frame_index]).save(
                directories["rgb"] / "{}.png".format(stem)
            )
            panel_parts.insert(0, rgb[frame_index])
        Image.fromarray(np.concatenate(panel_parts, axis=1)).save(
            directories["panels"] / "{}.png".format(stem)
        )

    sampled = np.zeros_like(valid)
    sampled[:, ::point_stride, ::point_stride] = True
    point_mask = valid & sampled
    if confidence_threshold is not None:
        point_mask &= confidence >= confidence_threshold
    if not np.any(point_mask):
        raise RuntimeError(
            "No point remains after stride/depth/confidence filtering"
        )

    used_point_color = point_color
    if point_color == "rgb" and rgb is None:
        print(
            "RGB frame paths are unavailable; using confidence colors for the PLY."
        )
        used_point_color = "confidence"
    color_source = rgb if used_point_color == "rgb" else confidence_rgb
    points = xyz_global[point_mask].astype(np.float32)
    colors = color_source[point_mask].astype(np.uint8)
    write_binary_ply(output / "teacher_global_point_cloud.ply", points, colors)
    np.savez_compressed(
        output / "teacher_reconstruction.npz",
        points=points,
        colors=colors,
        frame_ids=np.broadcast_to(
            np.arange(frames)[:, None, None], valid.shape
        )[point_mask].astype(np.int16),
        confidence=confidence[point_mask].astype(np.float32),
        frame_names=np.asarray(frame_names),
    )

    report = {
        "cache": str(cache_path),
        "output": str(output),
        "frame_count": frames,
        "frame_names": frame_names,
        "point_count": int(len(points)),
        "point_stride": point_stride,
        "point_color": used_point_color,
        "rgb_available": rgb is not None,
        "depth_range": [float(min_depth), float(max_depth)],
        "depth_color_percentiles": [float(color_low), float(color_high)],
        "confidence_threshold": confidence_threshold,
        "panel_order": (
            ["rgb", "depth_magma", "confidence_viridis"]
            if rgb is not None
            else ["depth_magma", "confidence_viridis"]
        ),
        "cache_metadata": metadata,
    }
    (output / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "Exported {} frames and {:,} points to {}".format(
            frames, len(points), output
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/teacher_cache_visualization"),
    )
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--min-depth", type=float, default=None)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument(
        "--point-color",
        choices=("rgb", "confidence"),
        default="rgb",
    )
    args = parser.parse_args()
    visualize_teacher_cache(
        cache_path=args.cache,
        output_root=args.output_root,
        point_stride=args.point_stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        confidence_threshold=args.confidence_threshold,
        point_color=args.point_color,
    )


if __name__ == "__main__":
    main()
