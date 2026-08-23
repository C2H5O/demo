"""Visualize one 16-frame cross-clip student prediction or teacher cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from datasets.crossclip_teacher_dataset import (
    CROSSCLIP_CACHE_PROTOCOL,
    crossclip_teacher_cache_path,
    make_crossclip_rgb_dataset,
    validate_crossclip_teacher_cache,
)
from datasets.scared_clip_dataset import clip_metadata
from datasets.transforms import unnormalize_image
from models.student.dune_fast3r_head import DuneFast3RHeadStudent
from utils.checkpoint import require_student_cache_protocol
from utils.config import ensure_dir, load_config


def _depth_to_magma(
    depth: np.ndarray, valid: np.ndarray, low: float, high: float
) -> np.ndarray:
    from visualization.scared_student import depth_to_magma

    return depth_to_magma(depth, valid, low, high)


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    from visualization.scared_student import write_binary_ply

    write_binary_ply(path, points, colors)


def _rgb(images: torch.Tensor, normalize_mode: str) -> np.ndarray:
    frames = torch.stack(
        [unnormalize_image(frame, normalize_mode) for frame in images]
    )
    return np.round(frames.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)


def _adaptive_range(
    depth: np.ndarray, valid: np.ndarray, percentiles: tuple[float, float]
) -> tuple[float, float]:
    if not np.any(valid):
        raise RuntimeError("No finite positive depth to visualize")
    low, high = np.percentile(depth[valid], percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(depth[valid]))
        high = float(np.max(depth[valid]))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _load_student_points(
    checkpoint_path: Path,
    config: Dict[str, Any],
    images: torch.Tensor,
) -> np.ndarray:
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Student checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    require_student_cache_protocol(checkpoint, CROSSCLIP_CACHE_PROTOCOL)
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA visualization requested but unavailable")
    model_config = checkpoint.get("config", {}).get("student", config["student"])
    model = DuneFast3RHeadStudent(model_config, device=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        prediction = model(images.unsqueeze(0).to(device))
    return prediction["pts3d_local"][0].float().cpu().numpy()


def _load_teacher_points(
    config: Dict[str, Any],
    dataset: Any,
    dataset_index: int,
    split: str,
) -> tuple[np.ndarray, Path, str]:
    teacher = config["teacher"]
    use_aligned = bool(teacher.get("use_aligned_cache", True))
    stage = "aligned" if use_aligned else "raw"
    root_key = "aligned_cache_root" if use_aligned else "raw_cache_root"
    metadata = clip_metadata(dataset, dataset_index)
    path = crossclip_teacher_cache_path(Path(str(teacher[root_key])) / split, metadata)
    if not path.is_file():
        raise FileNotFoundError("Teacher cache not found: {}".format(path))
    shape = (int(config["dataset"]["image_height"]), int(config["dataset"]["image_width"]))
    with np.load(str(path), allow_pickle=False) as cache:
        validate_crossclip_teacher_cache(
            cache,
            metadata,
            shape,
            str(teacher["pretrained_checkpoint"]),
            stage,
        )
        points = cache["xyz_local"].astype(np.float32, copy=True)
    return points, path, stage


def export_crossclip_visualization(
    config_path: Path,
    split: str,
    clip_index: int,
    output_root: Path,
    source: str = "student",
    checkpoint_path: Optional[Path] = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    point_stride: int = 4,
) -> Path:
    """Export fixed/adaptive depth and 16 independent camera-local PLY files."""
    if source not in {"student", "teacher"}:
        raise ValueError("source must be student or teacher")
    if point_stride <= 0:
        raise ValueError("point_stride must be positive")
    config = load_config(config_path)
    dataset_config = dict(config["dataset"])
    dataset_config["highlight"] = {"enabled": False}
    dataset = make_crossclip_rgb_dataset(dataset_config, split)
    if not 0 <= clip_index < len(dataset):
        raise IndexError("clip_index={} is outside [0,{})".format(clip_index, len(dataset)))
    sample = dataset[clip_index]
    metadata = clip_metadata(dataset, clip_index)
    rgb = _rgb(sample["images"], str(dataset.normalize_mode))
    teacher_cache: Optional[Path] = None
    cache_stage: Optional[str] = None
    if source == "student":
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required for source=student")
        points = _load_student_points(checkpoint_path, config, sample["images"])
    else:
        points, teacher_cache, cache_stage = _load_teacher_points(
            config, dataset, clip_index, split
        )
    depth = points[..., 2]
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth) & (depth > 0.0)
    adaptive = tuple(
        float(value)
        for value in config.get("visualization", {}).get(
            "adaptive_percentiles", [5.0, 95.0]
        )
    )
    if len(adaptive) != 2:
        raise ValueError("visualization.adaptive_percentiles must contain two values")
    adaptive_low, adaptive_high = _adaptive_range(depth, valid, adaptive)
    sequence_slug = str(metadata["sequence_id"]).replace("/", "_")
    output = ensure_dir(
        output_root
        / source
        / sequence_slug
        / "start_{:06d}".format(int(metadata["clip_start"]))
    )
    directories = {
        name: ensure_dir(output / name)
        for name in ("rgb", "depth", "depth_fixed", "depth_adaptive", "panels", "pointcloud_local")
    }
    for offset, name in enumerate(metadata["frame_names"]):
        stem = "{:02d}_{}".format(offset, Path(name).stem)
        fixed = _depth_to_magma(
            depth[offset],
            valid[offset] & (depth[offset] >= min_depth) & (depth[offset] <= max_depth),
            min_depth,
            max_depth,
        )
        adaptive_color = _depth_to_magma(
            depth[offset], valid[offset], adaptive_low, adaptive_high
        )
        Image.fromarray(rgb[offset]).save(directories["rgb"] / "{}.png".format(stem))
        np.save(directories["depth"] / "{}.npy".format(stem), depth[offset].astype(np.float32))
        Image.fromarray(fixed).save(directories["depth_fixed"] / "{}.png".format(stem))
        Image.fromarray(adaptive_color).save(directories["depth_adaptive"] / "{}.png".format(stem))
        Image.fromarray(np.concatenate((rgb[offset], fixed, adaptive_color), axis=1)).save(
            directories["panels"] / "{}.png".format(stem)
        )
        sampled = np.zeros(valid[offset].shape, dtype=bool)
        sampled[::point_stride, ::point_stride] = True
        point_mask = valid[offset] & sampled
        _write_binary_ply(
            directories["pointcloud_local"] / "{}.ply".format(stem),
            points[offset][point_mask].astype(np.float32),
            rgb[offset][point_mask],
        )
    record: Dict[str, Any] = {
        "source": source,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "teacher_cache": str(teacher_cache) if teacher_cache is not None else None,
        "teacher_cache_stage": cache_stage,
        "split": split,
        "dataset_index": clip_index,
        "sequence_id": metadata["sequence_id"],
        "clip_start": metadata["clip_start"],
        "absolute_frame_ids": metadata["frame_indices"],
        "frame_names": metadata["frame_names"],
        "coordinate_system": "each frame has its own independent camera-local coordinates",
        "fixed_depth_range": [min_depth, max_depth],
        "adaptive_percentiles": list(adaptive),
        "adaptive_depth_range": [adaptive_low, adaptive_high],
        "panel_order": ["rgb", "fixed depth", "adaptive depth"],
        "point_stride": point_stride,
    }
    (output / "metadata.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Exported cross-clip {} visualization to {}".format(source, output))
    return output


__all__ = ["export_crossclip_visualization"]
