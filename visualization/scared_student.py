"""Export depth previews and colored point clouds from one SCARED student clip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from datasets.scared_clip_dataset import make_scared_rgb_dataset
from datasets.transforms import unnormalize_image
from models.student.dune_model import DUNEViTSmallPointMapStudent
from utils.config import ensure_dir, load_config


def load_student(checkpoint_path: Path, config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Restore a complete student checkpoint without reloading the encoder seed."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    student_config = dict(checkpoint_config.get("student", config["student"]))
    student_config["encoder_checkpoint"] = None
    model = DUNEViTSmallPointMapStudent(student_config)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def depth_to_magma(depth: np.ndarray, valid: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map metric/relative depth to an EndoDAC-style MAGMA preview."""
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = np.clip((depth[valid] - low) / max(high - low, 1e-8), 0.0, 1.0)
    gray = np.round(normalized * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
    colored_bgr[~valid] = 0
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write XYZRGB vertices as a compact binary little-endian PLY file."""
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and colors must both have shape [N,3]")
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex {}\n".format(len(vertices))
        + "property float x\nproperty float y\nproperty float z\n"
        + "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def _select_clip(dataset: Any, sequence_id: Optional[str], clip_offset: int) -> Tuple[int, Any]:
    candidates = [
        index for index, record in enumerate(dataset.clips)
        if sequence_id is None or str(record.sequence["sequence_id"]) == sequence_id
    ]
    if not candidates:
        available = sorted({str(record.sequence["sequence_id"]) for record in dataset.clips})
        raise ValueError("No clips found for sequence {!r}. Available: {}".format(sequence_id, available))
    if not 0 <= clip_offset < len(candidates):
        raise IndexError("clip_offset={} is outside [0,{})".format(clip_offset, len(candidates)))
    index = candidates[clip_offset]
    return index, dataset.clips[index]


def predict_student_clip(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    sequence_id: Optional[str],
    clip_offset: int,
) -> Dict[str, Any]:
    """Load one deterministic clip and return CPU student predictions and RGB."""
    config = load_config(config_path)
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA visualization requested but CUDA is unavailable")
    dataset = make_scared_rgb_dataset(config["dataset"], split)
    dataset_index, record = _select_clip(dataset, sequence_id, clip_offset)
    sample = dataset[dataset_index]
    model = load_student(checkpoint_path, config, device)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        prediction = model(sample["images"].unsqueeze(0).to(device))
    rgb = torch.stack([
        unnormalize_image(frame, config["dataset"].get("normalize_mode", "imagenet"))
        for frame in sample["images"]
    ])
    return {
        "config": config,
        "sample": sample,
        "record": record,
        "dataset_index": dataset_index,
        "rgb": np.round(rgb.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8),
        "xyz_local": prediction["xyz_local"][0].float().cpu().numpy(),
        "xyz_global": prediction["xyz_global"][0].float().cpu().numpy(),
        "conf_local": prediction["conf_local"][0].float().cpu().numpy(),
        "conf_global": prediction["conf_global"][0].float().cpu().numpy(),
    }


def _clip_output_directory(output_root: Path, sample: Dict[str, Any], record: Any) -> Path:
    sequence_slug = str(sample["sequence_id"]).replace("/", "_")
    return ensure_dir(output_root / sequence_slug / "start_{:06d}".format(int(record.clip_start)))


def export_depth_visualization(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    sequence_id: Optional[str],
    clip_offset: int,
    output_root: Path,
    min_depth: float,
    max_depth: float,
) -> Path:
    """Save depth, local confidence, and aligned RGB/depth/confidence panels."""
    result = predict_student_clip(config_path, checkpoint_path, split, sequence_id, clip_offset)
    sample, record = result["sample"], result["record"]
    rgb, depth, confidence = result["rgb"], result["xyz_local"][..., 2], result["conf_local"]
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    if not np.any(valid):
        raise RuntimeError("Student produced no finite depth in the requested range")
    color_low, color_high = np.percentile(depth[valid], (5.0, 95.0))
    output = _clip_output_directory(output_root, sample, record)
    directories = {
        name: ensure_dir(output / name)
        for name in ("rgb", "depth", "depth_color", "confidence", "confidence_color", "depth_confidence")
    }
    for frame_index, frame_name in enumerate(sample["frame_names"]):
        stem = Path(frame_name).stem
        depth_color = depth_to_magma(
            depth[frame_index], valid[frame_index], float(color_low), float(color_high)
        )
        conf_uint8 = np.round(np.clip(confidence[frame_index], 0.0, 1.0) * 255.0).astype(np.uint8)
        confidence_bgr = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_VIRIDIS)
        confidence_color = cv2.cvtColor(confidence_bgr, cv2.COLOR_BGR2RGB)
        confidence_color[~valid[frame_index]] = 0
        panel = np.concatenate((rgb[frame_index], depth_color, confidence_color), axis=1)
        Image.fromarray(rgb[frame_index]).save(directories["rgb"] / "{}.png".format(stem))
        np.save(directories["depth"] / "{}.npy".format(stem), depth[frame_index].astype(np.float32))
        Image.fromarray(depth_color).save(directories["depth_color"] / "{}.png".format(stem))
        np.save(directories["confidence"] / "{}.npy".format(stem), confidence[frame_index].astype(np.float32))
        Image.fromarray(confidence_color).save(directories["confidence_color"] / "{}.png".format(stem))
        Image.fromarray(panel).save(directories["depth_confidence"] / "{}.png".format(stem))
    metadata = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "sequence_id": str(sample["sequence_id"]),
        "dataset_index": result["dataset_index"],
        "clip_offset": clip_offset,
        "clip_start": int(record.clip_start),
        "frame_names": sample["frame_names"],
        "depth_range": [min_depth, max_depth],
        "depth_color_percentiles": [float(color_low), float(color_high)],
        "panel_order": ["rgb", "depth_magma", "local_confidence_viridis"],
        "confidence_warning": "Student confidence is output but is not supervised by the current loss.",
    }
    (output / "depth_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Exported depth and confidence visualization to {}".format(output))
    return output


def export_cloud_visualization(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    sequence_id: Optional[str],
    clip_offset: int,
    output_root: Path,
    point_stride: int,
    min_depth: float,
    max_depth: float,
    confidence_threshold: Optional[float],
) -> Path:
    """Save one clip's RGB-colored global point map as PLY and NPZ."""
    if point_stride <= 0:
        raise ValueError("point_stride must be positive")
    result = predict_student_clip(config_path, checkpoint_path, split, sequence_id, clip_offset)
    sample, record = result["sample"], result["record"]
    rgb, xyz_global = result["rgb"], result["xyz_global"]
    depth, confidence = result["xyz_local"][..., 2], result["conf_global"]
    valid = np.isfinite(depth) & np.isfinite(xyz_global).all(axis=-1)
    valid &= (depth > min_depth) & (depth < max_depth)
    sampled = np.zeros(valid.shape, dtype=bool)
    sampled[:, ::point_stride, ::point_stride] = True
    point_mask = valid & sampled
    if confidence_threshold is not None:
        point_mask &= confidence >= confidence_threshold
    points, colors = xyz_global[point_mask].astype(np.float32), rgb[point_mask]
    if len(points) == 0:
        raise RuntimeError("No points remained after depth/confidence filtering")
    frame_ids = np.broadcast_to(np.arange(len(depth))[:, None, None], depth.shape)[point_mask].astype(np.int16)
    output = _clip_output_directory(output_root, sample, record)
    write_binary_ply(output / "point_cloud.ply", points, colors)
    np.savez_compressed(
        output / "reconstruction.npz",
        points=points,
        colors=colors,
        frame_ids=frame_ids,
        confidence=confidence[point_mask].astype(np.float32),
        frame_names=np.asarray(sample["frame_names"]),
    )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "sequence_id": str(sample["sequence_id"]),
        "dataset_index": result["dataset_index"],
        "clip_offset": clip_offset,
        "clip_start": int(record.clip_start),
        "frame_names": sample["frame_names"],
        "point_count": int(len(points)),
        "point_stride": point_stride,
        "depth_range": [min_depth, max_depth],
        "confidence_threshold": confidence_threshold,
        "confidence_warning": "Student confidence is currently unsupervised; keep threshold disabled for the baseline.",
    }
    (output / "cloud_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Exported {} colored points to {}".format(len(points), output))
    return output


def export_clip_visualization(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    sequence_id: Optional[str],
    clip_offset: int,
    output_root: Path,
    point_stride: int,
    min_depth: float,
    max_depth: float,
    confidence_threshold: Optional[float],
) -> Path:
    """Run one clip and export RGB, depth, PLY, and Endo3R-style reconstruction arrays."""
    if point_stride <= 0:
        raise ValueError("point_stride must be positive")
    config = load_config(config_path)
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA visualization requested but CUDA is unavailable")
    dataset = make_scared_rgb_dataset(config["dataset"], split)
    dataset_index, record = _select_clip(dataset, sequence_id, clip_offset)
    sample = dataset[dataset_index]
    model = load_student(checkpoint_path, config, device)
    amp_enabled = device.type == "cuda"
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=amp_enabled):
        prediction = model(sample["images"].unsqueeze(0).to(device))

    xyz_local = prediction["xyz_local"][0].float().cpu().numpy()
    xyz_global = prediction["xyz_global"][0].float().cpu().numpy()
    confidence = prediction["conf_global"][0].float().cpu().numpy()
    depth = xyz_local[..., 2]
    rgb = torch.stack([
        unnormalize_image(frame, config["dataset"].get("normalize_mode", "imagenet"))
        for frame in sample["images"]
    ])
    rgb = np.round(rgb.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)

    finite = np.isfinite(depth) & np.isfinite(xyz_global).all(axis=-1)
    valid = finite & (depth > min_depth) & (depth < max_depth)
    valid_values = depth[valid]
    if valid_values.size == 0:
        raise RuntimeError("Student produced no finite points in the requested depth range")
    color_low, color_high = np.percentile(valid_values, (5.0, 95.0))

    sequence_slug = str(sample["sequence_id"]).replace("/", "_")
    output = ensure_dir(output_root / sequence_slug / "start_{:06d}".format(int(record.clip_start)))
    rgb_dir, depth_dir, color_dir = ensure_dir(output / "rgb"), ensure_dir(output / "depth"), ensure_dir(output / "depth_color")
    for frame_index, frame_name in enumerate(sample["frame_names"]):
        stem = Path(frame_name).stem
        Image.fromarray(rgb[frame_index]).save(rgb_dir / "{}.png".format(stem))
        np.save(depth_dir / "{}.npy".format(stem), depth[frame_index].astype(np.float32))
        preview = depth_to_magma(depth[frame_index], valid[frame_index], float(color_low), float(color_high))
        Image.fromarray(preview).save(color_dir / "{}.png".format(stem))

    sampled = np.zeros(valid.shape, dtype=bool)
    sampled[:, ::point_stride, ::point_stride] = True
    point_mask = valid & sampled
    if confidence_threshold is not None:
        point_mask &= confidence >= confidence_threshold
    points = xyz_global[point_mask].astype(np.float32)
    colors = rgb[point_mask]
    frame_ids = np.broadcast_to(np.arange(len(depth))[:, None, None], depth.shape)[point_mask].astype(np.int16)
    point_confidence = confidence[point_mask].astype(np.float32)
    write_binary_ply(output / "point_cloud.ply", points, colors)
    np.savez_compressed(
        output / "reconstruction.npz",
        points=points,
        colors=colors,
        frame_ids=frame_ids,
        confidence=point_confidence,
        depth=depth.astype(np.float32),
        frame_names=np.asarray(sample["frame_names"]),
    )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "sequence_id": str(sample["sequence_id"]),
        "dataset_index": dataset_index,
        "clip_offset": clip_offset,
        "clip_start": int(record.clip_start),
        "frame_names": sample["frame_names"],
        "point_count": int(len(points)),
        "point_stride": point_stride,
        "depth_range": [min_depth, max_depth],
        "depth_color_percentiles": [float(color_low), float(color_high)],
        "confidence_threshold": confidence_threshold,
        "confidence_warning": "Student confidence is currently not supervised; leave threshold disabled for valid comparisons.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Exported {} colored points to {}".format(len(points), output))
    return output
