"""Stride-one 16-frame samples with neighboring frozen teacher clip caches."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.scared_clip_dataset import clip_metadata, make_scared_rgb_dataset
from datasets.scared_dataset import ClipRecord, ScaredTemporalRGBDataset, seed_worker


CROSSCLIP_CACHE_FORMAT_VERSION = "vggtomega-crossclip-local-v1"
CROSSCLIP_CACHE_PROTOCOL = "crossclip_local_v1"
LOCAL_CAMERA_COORDINATE_SYSTEM = "local_camera"
WORLD_TO_CAMERA_POSE_CONVENTION = "world_to_camera: X_camera = R @ X_world + t"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def make_crossclip_rgb_dataset(
    dataset_config: Dict[str, Any], split: str
) -> ScaredTemporalRGBDataset:
    """Build every legal 16-frame clip: C_0 ... C_(n-16), never crossing sequences."""
    config = dict(dataset_config)
    for key in ("random_clip_sampling", "teacher_neighbor_offset"):
        config.pop(key, None)
    config.update(
        {
            "clip_length": 16,
            "sample_stride": 1,
            "window_stride": 1,
            "drop_incomplete_clip": True,
        }
    )
    dataset = make_scared_rgb_dataset(config, split)
    if dataset.clip_length != 16 or dataset.sample_stride != 1 or dataset.window_stride != 1:
        raise RuntimeError("Cross-clip dataset failed to enforce 16-frame stride-one clips")
    return dataset


def crossclip_teacher_cache_path(
    cache_root: Union[str, Path], metadata: Dict[str, Any]
) -> Path:
    return (
        Path(cache_root)
        / "dataset_{:02d}".format(int(metadata["dataset_id"]))
        / _safe_name(str(metadata["keyframe_id"]))
        / _safe_name(str(metadata["sequence_id"]))
        / "start_{:06d}_len_016_stride_01.npz".format(
            int(metadata["clip_start"])
        )
    )


def build_neighbor_clip_indices(
    clips: Sequence[ClipRecord],
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Return dataset indices for C_(t-1), C_(t+1) within the same sequence."""
    lookup: Dict[Tuple[str, int], int] = {}
    for index, record in enumerate(clips):
        key = (str(record.sequence["sequence_id"]), int(record.clip_start))
        if key in lookup:
            raise RuntimeError("Duplicate clip identity {}".format(key))
        lookup[key] = index
    result = []
    for record in clips:
        sequence_id = str(record.sequence["sequence_id"])
        start = int(record.clip_start)
        result.append(
            (
                lookup.get((sequence_id, start - 1)),
                lookup.get((sequence_id, start + 1)),
            )
        )
    return result


REQUIRED_CROSSCLIP_CACHE_KEYS = (
    "sequence_id",
    "clip_start",
    "absolute_frame_ids",
    "frame_names",
    "input_height",
    "input_width",
    "depth",
    "xyz_local",
    "xyz_global",
    "confidence",
    "valid_mask",
    "highlight_mask",
    "intrinsics",
    "extrinsics",
    "pose_convention",
    "point_coordinate_system",
    "teacher_variant",
    "base_checkpoint",
    "cache_stage",
    "alignment_scale",
    "cache_format_version",
    "metadata_json",
)


def validate_crossclip_teacher_cache(
    cache: "np.lib.npyio.NpzFile",
    metadata: Optional[Dict[str, Any]] = None,
    expected_shape: Optional[Tuple[int, int]] = None,
    expected_base_checkpoint: Optional[str] = None,
    expected_stage: Optional[str] = None,
) -> None:
    missing = [key for key in REQUIRED_CROSSCLIP_CACHE_KEYS if key not in cache]
    if missing:
        raise RuntimeError("Cross-clip cache is missing keys {}".format(missing))
    if str(cache["cache_format_version"].item()) != CROSSCLIP_CACHE_FORMAT_VERSION:
        raise RuntimeError("Incompatible cross-clip teacher cache version")
    if str(cache["teacher_variant"].item()) != "base":
        raise RuntimeError("Cross-clip cache must come from the frozen base teacher")
    if str(cache["pose_convention"].item()) != WORLD_TO_CAMERA_POSE_CONVENTION:
        raise RuntimeError("Cross-clip cache pose convention mismatch")
    if str(cache["point_coordinate_system"].item()) != LOCAL_CAMERA_COORDINATE_SYSTEM:
        raise RuntimeError("Cross-clip teacher targets must be camera-local")
    stage = str(cache["cache_stage"].item())
    if stage not in {"raw", "aligned"}:
        raise RuntimeError("Cross-clip cache has invalid stage {!r}".format(stage))
    if expected_stage is not None and stage != expected_stage:
        raise RuntimeError("Cross-clip cache stage {!r} != {!r}".format(stage, expected_stage))
    height = int(cache["input_height"].item())
    width = int(cache["input_width"].item())
    shape = (height, width)
    if expected_shape is not None and shape != tuple(expected_shape):
        raise RuntimeError("Cross-clip cache resolution {} != {}".format(shape, expected_shape))
    expected_shapes = {
        "absolute_frame_ids": (16,),
        "frame_names": (16,),
        "depth": (16, height, width),
        "xyz_local": (16, height, width, 3),
        "xyz_global": (16, height, width, 3),
        "confidence": (16, height, width),
        "valid_mask": (16, height, width),
        "highlight_mask": (16, height, width),
        "intrinsics": (16, 3, 3),
        "extrinsics": (16, 3, 4),
    }
    for key, expected in expected_shapes.items():
        if tuple(cache[key].shape) != expected:
            raise RuntimeError(
                "Cross-clip cache {} shape {} != {}".format(
                    key, tuple(cache[key].shape), expected
                )
            )
    for key in ("depth", "xyz_local", "xyz_global", "confidence", "intrinsics", "extrinsics"):
        if cache[key].dtype != np.float32:
            raise RuntimeError("Cross-clip cache {} must be float32".format(key))
        if not np.isfinite(cache[key]).all():
            raise RuntimeError("Cross-clip cache {} contains non-finite values".format(key))
    for key in ("valid_mask", "highlight_mask"):
        if cache[key].dtype != np.bool_:
            raise RuntimeError("Cross-clip cache {} must be boolean".format(key))
    valid = cache["valid_mask"]
    valid_counts = valid.reshape(16, -1).sum(axis=1)
    if np.any(valid_counts <= 0):
        raise RuntimeError(
            "Cross-clip cache has frames without valid teacher pixels: {}".format(
                np.flatnonzero(valid_counts <= 0).tolist()
            )
        )
    depth = cache["depth"]
    points = cache["xyz_local"]
    confidence = cache["confidence"]
    if np.any(depth[valid] <= 0.0):
        raise RuntimeError("Cross-clip cache has non-positive valid depth")
    if np.any(confidence[valid] < 0.0):
        raise RuntimeError("Cross-clip cache has negative valid confidence")
    if not np.allclose(points[..., 2][valid], depth[valid], rtol=1e-3, atol=1e-4):
        raise RuntimeError("Cross-clip cache xyz_local Z is inconsistent with depth")
    intrinsics = cache["intrinsics"]
    if np.any(intrinsics[:, 0, 0] <= 1e-6) or np.any(intrinsics[:, 1, 1] <= 1e-6):
        raise RuntimeError("Cross-clip cache intrinsics have non-positive focal length")
    if np.any(np.abs(np.linalg.det(intrinsics.astype(np.float64))) <= 1e-9):
        raise RuntimeError("Cross-clip cache intrinsics are singular")
    expected_bottom = np.broadcast_to(
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32), (16, 3)
    )
    if not np.allclose(intrinsics[:, 2, :], expected_bottom, rtol=0.0, atol=1e-4):
        raise RuntimeError("Cross-clip cache intrinsics have an invalid bottom row")
    rotations = cache["extrinsics"][:, :3, :3].astype(np.float64)
    determinants = np.linalg.det(rotations)
    if np.any(np.abs(determinants - 1.0) > 5e-2):
        raise RuntimeError("Cross-clip cache extrinsic rotations are degenerate")
    try:
        cache_metadata = json.loads(str(cache["metadata_json"].item()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("Cross-clip cache metadata_json is invalid") from error
    statistic_keys = {
        "minimum_valid_fraction",
        "valid_fraction_per_frame",
        "valid_depth_min",
        "valid_depth_max",
        "valid_confidence_mean",
    }
    missing_statistics = statistic_keys - set(cache_metadata)
    if missing_statistics:
        raise RuntimeError(
            "Cross-clip cache metadata lacks integrity statistics {}".format(
                sorted(missing_statistics)
            )
        )
    fractions = valid_counts.astype(np.float64) / float(height * width)
    if not np.allclose(
        fractions,
        np.asarray(cache_metadata["valid_fraction_per_frame"], dtype=np.float64),
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError("Cross-clip cache valid-fraction metadata is stale")
    if np.any(fractions < float(cache_metadata["minimum_valid_fraction"])):
        raise RuntimeError("Cross-clip cache valid fraction is below its declared minimum")
    observed_statistics = (
        float(depth[valid].min()),
        float(depth[valid].max()),
        float(confidence[valid].mean()),
    )
    declared_statistics = (
        float(cache_metadata["valid_depth_min"]),
        float(cache_metadata["valid_depth_max"]),
        float(cache_metadata["valid_confidence_mean"]),
    )
    if not np.allclose(observed_statistics, declared_statistics, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Cross-clip cache integrity statistics are stale")
    scale = float(cache["alignment_scale"].item())
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("Cross-clip cache alignment_scale must be positive and finite")
    if expected_base_checkpoint is not None:
        actual = str(cache["base_checkpoint"].item())
        if actual != expected_base_checkpoint:
            raise RuntimeError(
                "Cross-clip cache base checkpoint {!r} != {!r}".format(
                    actual, expected_base_checkpoint
                )
            )
    if metadata is not None:
        checks = {
            "sequence_id": str(metadata["sequence_id"]),
            "clip_start": int(metadata["clip_start"]),
        }
        for key, expected in checks.items():
            if cache[key].item() != expected:
                raise RuntimeError("Cross-clip cache metadata mismatch for {}".format(key))
        if cache["absolute_frame_ids"].tolist() != list(metadata["frame_indices"]):
            raise RuntimeError("Cross-clip cache absolute frame IDs do not match RGB clip")
        if cache["frame_names"].tolist() != list(metadata["frame_names"]):
            raise RuntimeError("Cross-clip cache frame names do not match RGB clip")


def _load_overlap_side(
    path: Path,
    teacher_metadata: Dict[str, Any],
    student_absolute_ids: Sequence[int],
    side: str,
    expected_shape: Tuple[int, int],
    expected_base_checkpoint: str,
    expected_stage: str,
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Neighbor teacher cache missing: {}".format(path))
    with np.load(str(path), allow_pickle=False) as cache:
        validate_crossclip_teacher_cache(
            cache,
            teacher_metadata,
            expected_shape,
            expected_base_checkpoint,
            expected_stage,
        )
        teacher_slice = slice(1, 16) if side == "left" else slice(0, 15)
        student_slice = slice(0, 15) if side == "left" else slice(1, 16)
        expected_ids = list(student_absolute_ids[student_slice])
        actual_ids = cache["absolute_frame_ids"][teacher_slice].tolist()
        if actual_ids != expected_ids:
            raise RuntimeError(
                "{} teacher absolute-frame mapping mismatch: {} != {}".format(
                    side, actual_ids, expected_ids
                )
            )
        result: Dict[str, Any] = {
            "exists": torch.tensor(True),
            "depth": torch.from_numpy(cache["depth"][teacher_slice].copy()).detach(),
            "confidence": torch.from_numpy(cache["confidence"][teacher_slice].copy()).detach(),
            "valid_mask": torch.from_numpy(cache["valid_mask"][teacher_slice].copy()).detach(),
            "intrinsics": torch.from_numpy(cache["intrinsics"][teacher_slice].copy()).detach(),
            "absolute_frame_ids": torch.tensor(actual_ids, dtype=torch.long),
            "student_local_indices": torch.arange(
                0 if side == "left" else 1,
                15 if side == "left" else 16,
                dtype=torch.long,
            ),
            "teacher_local_indices": torch.arange(
                1 if side == "left" else 0,
                16 if side == "left" else 15,
                dtype=torch.long,
            ),
            "clip_start": torch.tensor(int(teacher_metadata["clip_start"])),
            "sequence_id": str(teacher_metadata["sequence_id"]),
            "cache_path": str(path),
        }
    return result


def _empty_overlap_side(
    shape: Tuple[int, int], sequence_id: str
) -> Dict[str, Any]:
    height, width = shape
    return {
        "exists": torch.tensor(False),
        "depth": torch.zeros(15, height, width),
        "confidence": torch.zeros(15, height, width),
        "valid_mask": torch.zeros(15, height, width, dtype=torch.bool),
        "intrinsics": torch.zeros(15, 3, 3),
        "absolute_frame_ids": torch.full((15,), -1, dtype=torch.long),
        "student_local_indices": torch.full((15,), -1, dtype=torch.long),
        "teacher_local_indices": torch.full((15,), -1, dtype=torch.long),
        "clip_start": torch.tensor(-1),
        "sequence_id": sequence_id,
        "cache_path": "",
    }


class ScaredCrossClipProjectionDataset(Dataset):
    """Load C_t RGB and only the existing C_(t-1)/C_(t+1) teacher targets."""

    def __init__(
        self,
        rgb_dataset: ScaredTemporalRGBDataset,
        cache_root: Union[str, Path],
        expected_base_checkpoint: str,
        expected_stage: str = "aligned",
    ) -> None:
        self.rgb_dataset = rgb_dataset
        self.cache_root = Path(cache_root)
        self.expected_base_checkpoint = expected_base_checkpoint
        self.expected_stage = expected_stage
        self.neighbors = build_neighbor_clip_indices(rgb_dataset.clips)

    def __len__(self) -> int:
        return len(self.rgb_dataset)

    def _metadata(self, index: int) -> Dict[str, Any]:
        return clip_metadata(self.rgb_dataset, index)

    def missing_neighbor_cache_paths(self, limit: int = 10) -> List[Path]:
        missing: List[Path] = []
        for left, right in self.neighbors:
            for neighbor in (left, right):
                if neighbor is None:
                    continue
                path = crossclip_teacher_cache_path(
                    self.cache_root, self._metadata(neighbor)
                )
                if not path.is_file() and path not in missing:
                    missing.append(path)
                    if len(missing) >= limit:
                        return missing
        return missing

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.rgb_dataset[index]
        metadata = self._metadata(index)
        shape = tuple(int(value) for value in sample["images"].shape[-2:])
        absolute_ids = metadata["frame_indices"]
        left_index, right_index = self.neighbors[index]

        def side_value(neighbor: Optional[int], side: str) -> Dict[str, Any]:
            if neighbor is None:
                return _empty_overlap_side(shape, metadata["sequence_id"])
            teacher_metadata = self._metadata(neighbor)
            if teacher_metadata["sequence_id"] != metadata["sequence_id"]:
                raise RuntimeError("Cross-clip neighbor crossed a sequence boundary")
            return _load_overlap_side(
                crossclip_teacher_cache_path(self.cache_root, teacher_metadata),
                teacher_metadata,
                absolute_ids,
                side,
                shape,
                self.expected_base_checkpoint,
                self.expected_stage,
            )

        highlight = sample.get(
            "highlight_masks",
            torch.zeros(16, 1, *shape, dtype=sample["images"].dtype),
        )
        clean = sample.get("inpainted_images")
        if clean is None:
            clean = sample["images"].add(1.0).div(2.0).clamp(0.0, 1.0)
        return {
            "images": sample["images"],
            "clean_images": clean,
            "highlight_masks": highlight.bool(),
            "frame_indices": sample["frame_indices"],
            "sequence_id": metadata["sequence_id"],
            "clip_start": sample["clip_start"],
            "absolute_frame_ids": sample["frame_indices"],
            "teacher_left": side_value(left_index, "left"),
            "teacher_right": side_value(right_index, "right"),
        }


_SIDE_TENSOR_KEYS = (
    "exists",
    "depth",
    "confidence",
    "valid_mask",
    "intrinsics",
    "absolute_frame_ids",
    "student_local_indices",
    "teacher_local_indices",
    "clip_start",
)


def crossclip_projection_collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty cross-clip batch")

    def collate_side(name: str) -> Dict[str, Any]:
        values: Dict[str, Any] = {
            key: torch.stack([sample[name][key] for sample in samples])
            for key in _SIDE_TENSOR_KEYS
        }
        values["sequence_id"] = [sample[name]["sequence_id"] for sample in samples]
        values["cache_path"] = [sample[name]["cache_path"] for sample in samples]
        return values

    return {
        "images": torch.stack([sample["images"] for sample in samples]),
        "clean_images": torch.stack([sample["clean_images"] for sample in samples]),
        "highlight_masks": torch.stack([sample["highlight_masks"] for sample in samples]),
        "frame_indices": torch.stack([sample["frame_indices"] for sample in samples]),
        "absolute_frame_ids": torch.stack([sample["absolute_frame_ids"] for sample in samples]),
        "clip_start": torch.stack([sample["clip_start"] for sample in samples]),
        "sequence_id": [sample["sequence_id"] for sample in samples],
        "teacher_left": collate_side("teacher_left"),
        "teacher_right": collate_side("teacher_right"),
    }


def build_crossclip_projection_dataloader(
    dataset: ScaredCrossClipProjectionDataset,
    loader_config: Dict[str, Any],
    seed: int,
    shuffle: bool,
) -> DataLoader:
    num_workers = int(loader_config.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(loader_config.get("batch_size", 1)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(loader_config.get("pin_memory", False)),
        "persistent_workers": bool(loader_config.get("persistent_workers", False)) if num_workers else False,
        "drop_last": bool(loader_config.get("drop_last", False)),
        "collate_fn": crossclip_projection_collate,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers:
        kwargs["prefetch_factor"] = int(loader_config.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


__all__ = [
    "CROSSCLIP_CACHE_FORMAT_VERSION",
    "CROSSCLIP_CACHE_PROTOCOL",
    "LOCAL_CAMERA_COORDINATE_SYSTEM",
    "REQUIRED_CROSSCLIP_CACHE_KEYS",
    "ScaredCrossClipProjectionDataset",
    "WORLD_TO_CAMERA_POSE_CONVENTION",
    "build_crossclip_projection_dataloader",
    "build_neighbor_clip_indices",
    "crossclip_projection_collate",
    "crossclip_teacher_cache_path",
    "make_crossclip_rgb_dataset",
    "validate_crossclip_teacher_cache",
]
