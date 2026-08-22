"""Visualize one strict VGG-to-MASt3R teacher pair cache without a model."""

from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.scared_pair_dataset import (
    PAIR_COORDINATE_CONVENTION,
    make_scared_pair_rgb_dataset,
    pair_metadata,
    teacher_pair_cache_path,
    validate_pair_cache,
)
from utils.config import ensure_dir, load_config


def _rgb_from_tensor(image: torch.Tensor) -> np.ndarray:
    return np.round(
        ((image.float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .permute(1, 2, 0)
        .numpy()
    ).astype(np.uint8)


def _load_metadata(cache: Any) -> Dict[str, Any]:
    try:
        return json.loads(str(cache["metadata_json"].item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("Pair cache has invalid metadata_json") from error


def _load_rgb_from_metadata(
    metadata: Dict[str, Any], height: int, width: int
) -> Optional[np.ndarray]:
    paths = metadata.get("frame_paths")
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
        return None
    if len(paths) != 2:
        return None
    images = []
    for value in paths:
        path = Path(str(value))
        if not path.is_file():
            return None
        with Image.open(path) as image:
            images.append(
                np.asarray(
                    image.convert("RGB").resize(
                        (width, height), Image.Resampling.BILINEAR
                    ),
                    dtype=np.uint8,
                )
            )
    return np.stack(images)


def _confidence_color(
    confidence: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as error:
        raise RuntimeError(
            "Teacher pair-cache visualization requires opencv-python"
        ) from error
    uint8 = np.round(np.clip(confidence, 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    color = cv2.cvtColor(
        cv2.applyColorMap(uint8, cv2.COLORMAP_VIRIDIS),
        cv2.COLOR_BGR2RGB,
    )
    color[~valid] = 0
    return color


def _labeled(image: np.ndarray, label: str) -> np.ndarray:
    banner_height = 24
    canvas = Image.new(
        "RGB", (image.shape[1], image.shape[0] + banner_height), (24, 24, 24)
    )
    canvas.paste(Image.fromarray(image), (0, banner_height))
    ImageDraw.Draw(canvas).text((5, 5), label, fill=(255, 255, 255))
    return np.asarray(canvas)


def _point_cloud(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    valid: np.ndarray,
    point_stride: int,
    confidence: np.ndarray,
    confidence_threshold: float,
) -> int:
    from visualization.scared_student import write_binary_ply

    sampled = np.zeros(valid.shape, dtype=bool)
    sampled[:, ::point_stride, ::point_stride] = True
    mask = (
        valid
        & sampled
        & np.isfinite(points).all(axis=-1)
        & np.isfinite(confidence)
        & (confidence >= confidence_threshold)
    )
    if not np.any(mask):
        raise RuntimeError("No valid points remain for {}".format(path.name))
    write_binary_ply(
        path, points[mask].astype(np.float32), colors[mask].astype(np.uint8)
    )
    return int(mask.sum())


def _default_output(cache_path: Path, output_root: Path) -> Path:
    try:
        with np.load(str(cache_path), allow_pickle=False) as cache:
            metadata = _load_metadata(cache)
        split = cache_path.parents[2].name
        dataset = "dataset_{:02d}".format(int(metadata["dataset_id"]))
        keyframe = str(metadata["keyframe_id"])
        return output_root / split / dataset / keyframe / cache_path.stem
    except (KeyError, IndexError, OSError, RuntimeError, ValueError):
        return output_root / cache_path.stem


def export_teacher_pair_cache_visualization(
    cache_path: Path,
    output_dir: Path,
    rgb: Optional[np.ndarray] = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    point_stride: int = 4,
    confidence_threshold: float = 0.0,
    expected_teacher_variant: Optional[str] = None,
    expected_lora_checkpoint: Optional[str] = None,
) -> Path:
    """Export maps, labeled panels, and local/global/reference-frame PLYs."""
    try:
        from visualization.scared_student import depth_to_magma
    except ImportError as error:
        raise RuntimeError(
            "Teacher pair-cache visualization requires opencv-python; "
            "install the project requirements first"
        ) from error
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError("Teacher pair cache not found: {}".format(cache_path))
    if min_depth >= max_depth:
        raise ValueError("min_depth must be smaller than max_depth")
    if point_stride <= 0:
        raise ValueError("point_stride must be positive")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0,1]")

    with np.load(str(cache_path), allow_pickle=False) as cache:
        validate_pair_cache(
            cache,
            expected_teacher_variant=expected_teacher_variant,
            expected_lora_checkpoint=expected_lora_checkpoint,
        )
        metadata = _load_metadata(cache)
        arrays = {
            name: cache[name].astype(np.float32)
            for name in (
                "depth_a",
                "depth_b",
                "xyz_local_a",
                "xyz_local_b",
                "xyz_global_a",
                "xyz_global_b",
                "pts3d_a_in_a",
                "pts3d_b_in_a",
                "confidence_a",
                "confidence_b",
            )
        }
        valid = np.stack(
            (cache["valid_mask_a"].astype(bool), cache["valid_mask_b"].astype(bool))
        )
        report_scalars = {
            "cache_format_version": str(cache["cache_format_version"].item()),
            "teacher_variant": str(cache["teacher_variant"].item()),
            "lora_checkpoint": str(cache["lora_checkpoint"].item()),
            "coordinate_convention": str(cache["coordinate_convention"].item()),
            "frame_names": [
                str(cache["frame_name_a"].item()),
                str(cache["frame_name_b"].item()),
            ],
            "pair_stride": int(cache["pair_stride"].item()),
        }

    height, width = arrays["depth_a"].shape
    if rgb is None:
        rgb = _load_rgb_from_metadata(metadata, height, width)
    if rgb is not None and tuple(rgb.shape) != (2, height, width, 3):
        raise ValueError(
            "RGB must have shape [2,{},{},3], got {}".format(
                height, width, tuple(rgb.shape)
            )
        )
    rgb_available = rgb is not None
    if rgb is None:
        rgb = np.zeros((2, height, width, 3), dtype=np.uint8)

    local_depth = np.stack((arrays["depth_a"], arrays["depth_b"]))
    confidence = np.stack((arrays["confidence_a"], arrays["confidence_b"]))
    local_points = np.stack((arrays["xyz_local_a"], arrays["xyz_local_b"]))
    global_points = np.stack((arrays["xyz_global_a"], arrays["xyz_global_b"]))
    pair_reference_points = np.stack(
        (arrays["pts3d_a_in_a"], arrays["pts3d_b_in_a"])
    )
    local_valid = (
        valid
        & np.isfinite(local_depth)
        & np.isfinite(local_points).all(axis=-1)
        & np.isfinite(confidence)
        & (local_depth >= min_depth)
        & (local_depth <= max_depth)
    )
    if not np.any(local_valid):
        raise RuntimeError("No valid cached teacher depths remain")

    depth_color = np.stack(
        [
            depth_to_magma(local_depth[i], local_valid[i], min_depth, max_depth)
            for i in range(2)
        ]
    )
    confidence_color = np.stack(
        [_confidence_color(confidence[i], local_valid[i]) for i in range(2)]
    )
    pair_z = pair_reference_points[..., 2]
    pair_z_valid = valid & np.isfinite(pair_reference_points).all(axis=-1)
    pair_z_color = np.stack(
        [
            depth_to_magma(pair_z[i], pair_z_valid[i], min_depth, max_depth)
            for i in range(2)
        ]
    )

    output = ensure_dir(output_dir)
    for name, values in (
        ("rgb_a", rgb[0]),
        ("rgb_b", rgb[1]),
        ("depth_a_local", depth_color[0]),
        ("depth_b_local", depth_color[1]),
        ("confidence_a", confidence_color[0]),
        ("confidence_b", confidence_color[1]),
        ("pts3d_a_in_a_z", pair_z_color[0]),
        ("pts3d_b_in_a_z_not_b_depth", pair_z_color[1]),
    ):
        Image.fromarray(values).save(output / "{}.png".format(name))
    np.save(output / "depth_a_local.npy", local_depth[0])
    np.save(output / "depth_b_local.npy", local_depth[1])
    np.save(output / "confidence_a.npy", confidence[0])
    np.save(output / "confidence_b.npy", confidence[1])
    np.save(output / "pts3d_a_in_a_z.npy", pair_z[0])
    np.save(output / "pts3d_b_in_a_z_not_b_depth.npy", pair_z[1])

    row_a = np.concatenate(
        [
            _labeled(rgb[0], "RGB A"),
            _labeled(depth_color[0], "A local depth"),
            _labeled(confidence_color[0], "A confidence"),
            _labeled(pair_z_color[0], "A-in-A Z"),
        ],
        axis=1,
    )
    row_b = np.concatenate(
        [
            _labeled(rgb[1], "RGB B"),
            _labeled(depth_color[1], "B local depth"),
            _labeled(confidence_color[1], "B confidence"),
            _labeled(pair_z_color[1], "B-in-A Z (not B depth)"),
        ],
        axis=1,
    )
    Image.fromarray(np.concatenate((row_a, row_b), axis=0)).save(
        output / "teacher_pair_panel.png"
    )

    colors = rgb
    if not rgb_available:
        colors = confidence_color
    point_counts = {
        "local_camera_a": _point_cloud(
            output / "teacher_local_camera_a.ply",
            local_points[0:1],
            colors[0:1],
            local_valid[0:1],
            point_stride,
            confidence[0:1],
            confidence_threshold,
        ),
        "local_camera_b": _point_cloud(
            output / "teacher_local_camera_b.ply",
            local_points[1:2],
            colors[1:2],
            local_valid[1:2],
            point_stride,
            confidence[1:2],
            confidence_threshold,
        ),
        "global": _point_cloud(
            output / "teacher_global.ply",
            global_points,
            colors,
            local_valid,
            point_stride,
            confidence,
            confidence_threshold,
        ),
        "pair_reference_camera": _point_cloud(
            output / "teacher_pair_reference_camera.ply",
            pair_reference_points,
            colors,
            local_valid,
            point_stride,
            confidence,
            confidence_threshold,
        ),
    }
    report: Dict[str, Any] = {
        "cache": str(cache_path),
        "output": str(output),
        **report_scalars,
        "rgb_available": rgb_available,
        "depth_color_range_m": [min_depth, max_depth],
        "confidence_threshold": confidence_threshold,
        "point_stride": point_stride,
        "point_counts": point_counts,
        "coordinate_semantics": {
            "depth_a": "camera-A local Z depth",
            "depth_b": "camera-B local Z depth",
            "pts3d_a_in_a": "A points in camera-A/reference coordinates",
            "pts3d_b_in_a": (
                "B points transformed into camera-A/reference coordinates; "
                "its Z is not camera-B depth"
            ),
            "xyz_global": "teacher world coordinates",
        },
        "metadata": metadata,
        "valid_fraction": [
            float(local_valid[0].mean()),
            float(local_valid[1].mean()),
        ],
        "depth_finite_minmax_m": [
            [
                float(local_depth[i][local_valid[i]].min()),
                float(local_depth[i][local_valid[i]].max()),
            ]
            for i in range(2)
        ],
    }
    if report_scalars["coordinate_convention"] != PAIR_COORDINATE_CONVENTION:
        raise RuntimeError("Unexpected pair coordinate convention")
    (output / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "Exported teacher pair cache {} to {} (points={})".format(
            cache_path, output, point_counts
        )
    )
    return output


def resolve_config_pair(
    config_path: Path,
    split: str,
    pair_index: Optional[int] = None,
    sequence_id: Optional[str] = None,
    frame_id_a: Optional[int] = None,
) -> tuple[Path, np.ndarray, Dict[str, Any], int]:
    """Resolve and validate one configured pair without loading teacher/student."""
    config = load_config(config_path)
    dataset = make_scared_pair_rgb_dataset(config["dataset"], split)
    if (sequence_id is None) != (frame_id_a is None):
        raise ValueError("--sequence-id and --frame-id-a must be used together")
    if sequence_id is not None:
        matches = []
        for index, record in enumerate(dataset.clips):
            if str(record.sequence["sequence_id"]) != sequence_id:
                continue
            if int(pair_metadata(dataset, index)["frame_id_a"]) == frame_id_a:
                matches.append(index)
        if len(matches) != 1:
            raise ValueError(
                "Expected one pair for sequence_id={!r} frame_id_a={}, found {}".format(
                    sequence_id, frame_id_a, len(matches)
                )
            )
        pair_index = matches[0]
    elif pair_index is None:
        pair_index = 0
    if not 0 <= pair_index < len(dataset):
        raise IndexError(
            "pair_index={} is outside [0,{})".format(pair_index, len(dataset))
        )
    metadata = pair_metadata(dataset, pair_index)
    cache_path = teacher_pair_cache_path(
        Path(config["teacher"]["cache_root"]) / split, metadata
    )
    sample = dataset[pair_index]
    rgb = np.stack([_rgb_from_tensor(image) for image in sample["images"]])
    return cache_path, rgb, config, pair_index


__all__ = [
    "_default_output",
    "export_teacher_pair_cache_visualization",
    "resolve_config_pair",
]
