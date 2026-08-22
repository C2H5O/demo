"""Versioned single-frame VGGT-Omega caches and runtime composition helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from datasets.ground_truth import frame_id
from datasets.scared_clip_dataset import make_scared_rgb_dataset
from datasets.scared_dataset import ScaredTemporalRGBDataset


FRAME_CACHE_FORMAT_VERSION = "vggtomega-base-frame-v1"
FRAME_COORDINATE_CONVENTION = (
    "camera-local; each frame was inferred independently with sequence length 1"
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def make_scared_frame_rgb_dataset(
    dataset_config: Dict[str, Any], split: str
) -> ScaredTemporalRGBDataset:
    """Build a one-record-per-source-frame dataset without changing RGB policy."""
    config = dict(dataset_config)
    for key in ("pair_mode", "pair_stride", "pair_step"):
        config.pop(key, None)
    config.update(
        {
            "clip_length": 1,
            "sample_stride": 1,
            "window_stride": 1,
            "drop_incomplete_clip": True,
        }
    )
    return make_scared_rgb_dataset(config, split)


def frame_metadata(
    dataset: ScaredTemporalRGBDataset, index: int
) -> Dict[str, Any]:
    record = dataset.clips[index]
    if len(record.frame_indices) != 1:
        raise RuntimeError(
            "Frame cache dataset produced {} frames".format(len(record.frame_indices))
        )
    frame_index = int(record.frame_indices[0])
    sequence = record.sequence
    frame_path = str(sequence["frame_paths"][frame_index])
    frame_name = Path(frame_path).name
    return {
        "dataset_id": int(sequence["dataset_id"]),
        "keyframe_id": str(sequence["keyframe_id"]),
        "sequence_id": str(sequence["sequence_id"]),
        "sequence_length": int(sequence["sequence_length"]),
        "frame_id": frame_id(frame_name),
        "frame_index": frame_index,
        "frame_name": frame_name,
        "frame_path": frame_path,
    }


def frame_metadata_from_pair(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert ordered pair metadata into two frame-cache identities."""
    common = {
        "dataset_id": int(pair["dataset_id"]),
        "keyframe_id": str(pair["keyframe_id"]),
        "sequence_id": str(pair["sequence_id"]),
        "sequence_length": int(pair["sequence_length"]),
    }
    return [
        {
            **common,
            "frame_id": int(pair["frame_id_a"]),
            "frame_index": int(pair["frame_index_a"]),
            "frame_name": str(pair["frame_name_a"]),
            "frame_path": str(pair["frame_path_a"]),
        },
        {
            **common,
            "frame_id": int(pair["frame_id_b"]),
            "frame_index": int(pair["frame_index_b"]),
            "frame_name": str(pair["frame_name_b"]),
            "frame_path": str(pair["frame_path_b"]),
        },
    ]


def frame_metadata_from_clip(clip: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a 2- or 8-frame clip identity into ordered frame identities."""
    paths = list(clip["frame_paths"])
    names = list(clip["frame_names"])
    indices = [int(value) for value in clip["frame_indices"]]
    if len(paths) not in (2, 8) or not (len(paths) == len(names) == len(indices)):
        raise ValueError("Clip metadata must contain matching 2 or 8 frame entries")
    common = {
        "dataset_id": int(clip["dataset_id"]),
        "keyframe_id": str(clip["keyframe_id"]),
        "sequence_id": str(clip["sequence_id"]),
        "sequence_length": int(clip["sequence_length"]),
    }
    return [
        {
            **common,
            "frame_id": frame_id(names[position]),
            "frame_index": indices[position],
            "frame_name": str(names[position]),
            "frame_path": str(paths[position]),
        }
        for position in range(len(paths))
    ]


def teacher_frame_cache_path(
    cache_root: Union[str, Path], metadata: Dict[str, Any]
) -> Path:
    return (
        Path(cache_root)
        / "dataset_{:02d}".format(int(metadata["dataset_id"]))
        / _safe_name(str(metadata["keyframe_id"]))
        / _safe_name(str(metadata["sequence_id"]))
        / "frame_{:06d}_{:06d}.npz".format(
            int(metadata["frame_index"]), int(metadata["frame_id"])
        )
    )


REQUIRED_FRAME_CACHE_KEYS = (
    "dataset_id",
    "keyframe_id",
    "sequence_id",
    "frame_id",
    "frame_index",
    "frame_name",
    "image_shape",
    "teacher_variant",
    "inference_frame_count",
    "depth",
    "xyz_local",
    "confidence",
    "valid_mask",
    "intrinsics",
    "extrinsics",
    "coordinate_convention",
    "cache_format_version",
    "base_checkpoint",
    "metadata_json",
)


def validate_teacher_frame_cache(
    cache: "np.lib.npyio.NpzFile",
    metadata: Optional[Dict[str, Any]] = None,
    expected_shape: Optional[tuple[int, int]] = None,
    expected_base_checkpoint: Optional[str] = None,
) -> None:
    missing = [key for key in REQUIRED_FRAME_CACHE_KEYS if key not in cache]
    if missing:
        raise RuntimeError("Teacher frame cache is missing keys {}".format(missing))
    version = str(cache["cache_format_version"].item())
    if version != FRAME_CACHE_FORMAT_VERSION:
        raise RuntimeError(
            "Stale/incompatible frame cache version {!r}; expected {!r}".format(
                version, FRAME_CACHE_FORMAT_VERSION
            )
        )
    if str(cache["teacher_variant"].item()) != "base":
        raise RuntimeError("Teacher frame cache must use frozen base weights")
    if int(cache["inference_frame_count"].item()) != 1:
        raise RuntimeError("Teacher frame cache was not inferred independently")
    convention = str(cache["coordinate_convention"].item())
    if convention != FRAME_COORDINATE_CONVENTION:
        raise RuntimeError(
            "Frame cache coordinate convention {!r} != {!r}".format(
                convention, FRAME_COORDINATE_CONVENTION
            )
        )
    shape = tuple(int(value) for value in cache["image_shape"].tolist())
    if expected_shape is not None and shape != tuple(expected_shape):
        raise RuntimeError("Frame cache resolution {} != {}".format(shape, expected_shape))
    if tuple(cache["depth"].shape) != shape:
        raise RuntimeError("Frame cache depth has invalid shape {}".format(cache["depth"].shape))
    if tuple(cache["xyz_local"].shape) != shape + (3,):
        raise RuntimeError(
            "Frame cache xyz_local has invalid shape {}".format(cache["xyz_local"].shape)
        )
    for key in ("confidence", "valid_mask"):
        if tuple(cache[key].shape) != shape:
            raise RuntimeError("Frame cache {} has invalid shape {}".format(key, cache[key].shape))
    if tuple(cache["intrinsics"].shape) != (3, 3):
        raise RuntimeError(
            "Frame cache intrinsics has invalid shape {}".format(cache["intrinsics"].shape)
        )
    if tuple(cache["extrinsics"].shape) != (3, 4):
        raise RuntimeError(
            "Frame cache extrinsics has invalid shape {}".format(cache["extrinsics"].shape)
        )
    for key in ("depth", "xyz_local", "confidence", "intrinsics", "extrinsics"):
        if cache[key].dtype != np.float32:
            raise RuntimeError(
                "Frame cache {} must be float32, got {}".format(key, cache[key].dtype)
            )
        if not np.isfinite(cache[key]).all():
            raise RuntimeError("Frame cache {} contains non-finite values".format(key))
    if cache["valid_mask"].dtype != np.bool_:
        raise RuntimeError(
            "Frame cache valid_mask must be boolean, got {}".format(
                cache["valid_mask"].dtype
            )
        )
    if expected_base_checkpoint is not None:
        actual_checkpoint = str(cache["base_checkpoint"].item())
        if actual_checkpoint != expected_base_checkpoint:
            raise RuntimeError(
                "Frame cache base checkpoint {!r} != configured {!r}".format(
                    actual_checkpoint, expected_base_checkpoint
                )
            )
    if metadata is not None:
        checks = {
            "dataset_id": int(metadata["dataset_id"]),
            "keyframe_id": str(metadata["keyframe_id"]),
            "sequence_id": str(metadata["sequence_id"]),
            "frame_id": int(metadata["frame_id"]),
            "frame_index": int(metadata["frame_index"]),
            "frame_name": str(metadata["frame_name"]),
        }
        for key, expected in checks.items():
            actual = cache[key].item()
            if actual != expected:
                raise RuntimeError(
                    "Frame cache metadata mismatch for {}: {} != {}".format(
                        key, actual, expected
                    )
                )


def compose_teacher_frame_caches(
    cache_root: Union[str, Path],
    frames: Sequence[Dict[str, Any]],
    expected_shape: tuple[int, int],
    expected_base_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Stack independent frame-local caches for a 2- or 8-frame sample."""
    if len(frames) not in (2, 8):
        raise ValueError(
            "Frame cache composition supports exactly 2 or 8 frames, got {}".format(
                len(frames)
            )
        )
    tensors: Dict[str, List[torch.Tensor]] = {
        "depth": [],
        "xyz_local": [],
        "confidence": [],
        "valid_mask": [],
        "intrinsics": [],
        "extrinsics": [],
    }
    paths: List[str] = []
    for metadata in frames:
        path = teacher_frame_cache_path(cache_root, metadata)
        if not path.is_file():
            raise FileNotFoundError("Teacher frame cache missing: {}".format(path))
        try:
            with np.load(str(path), allow_pickle=False) as cache:
                validate_teacher_frame_cache(
                    cache,
                    metadata,
                    expected_shape,
                    expected_base_checkpoint,
                )
                tensors["depth"].append(
                    torch.from_numpy(cache["depth"].astype(np.float32))
                )
                tensors["xyz_local"].append(
                    torch.from_numpy(cache["xyz_local"].astype(np.float32))
                )
                tensors["confidence"].append(
                    torch.from_numpy(cache["confidence"].astype(np.float32))
                )
                tensors["valid_mask"].append(
                    torch.from_numpy(cache["valid_mask"].astype(np.bool_))
                )
                tensors["intrinsics"].append(
                    torch.from_numpy(cache["intrinsics"].astype(np.float32))
                )
                tensors["extrinsics"].append(
                    torch.from_numpy(cache["extrinsics"].astype(np.float32))
                )
        except (OSError, ValueError) as error:
            raise RuntimeError("Failed to read frame cache {}: {}".format(path, error)) from error
        paths.append(str(path))
    result: Dict[str, Any] = {
        key: torch.stack(values, dim=0) for key, values in tensors.items()
    }
    result["cache_paths"] = paths
    result["frame_names"] = [str(metadata["frame_name"]) for metadata in frames]
    result["frame_indices"] = [int(metadata["frame_index"]) for metadata in frames]
    result["coordinate_convention"] = FRAME_COORDINATE_CONVENTION
    return result


__all__ = [
    "FRAME_CACHE_FORMAT_VERSION",
    "FRAME_COORDINATE_CONVENTION",
    "REQUIRED_FRAME_CACHE_KEYS",
    "compose_teacher_frame_caches",
    "frame_metadata",
    "frame_metadata_from_clip",
    "frame_metadata_from_pair",
    "make_scared_frame_rgb_dataset",
    "teacher_frame_cache_path",
    "validate_teacher_frame_cache",
]
