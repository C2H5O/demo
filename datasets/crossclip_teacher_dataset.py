"""Stride-eight 16-frame samples with lazy neighboring teacher cache reads."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.precomputed_highlight import parse_highlight_options, precomputed_highlight_paths
from datasets.scared_clip_dataset import clip_metadata, make_scared_rgb_dataset
from datasets.scared_dataset import ClipRecord, ScaredTemporalRGBDataset, seed_worker
from datasets.transforms import (
    load_precomputed_highlight_mask_tensor,
    load_precomputed_student_rgb_tensor,
    load_rgb_tensor,
    tensor_from_numpy_buffer,
)


CROSSCLIP_CACHE_FORMAT_VERSION = "vggtomega-crossclip-local-v1"
SUPPORTED_CROSSCLIP_CACHE_FORMAT_VERSIONS = {
    "vggtomega-crossclip-local-v1",
    "vggtomega-crossclip-local-v2",
}
CROSSCLIP_CACHE_PROTOCOL = "crossclip_local_v1"
LOCAL_CAMERA_COORDINATE_SYSTEM = "local_camera"
WORLD_TO_CAMERA_POSE_CONVENTION = "world_to_camera: X_camera = R @ X_world + t"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


class CacheMetadataRGBDataset(Dataset):
    """RGB clips indexed by frame paths embedded in independent teacher caches."""

    def __init__(self, cache_root: Union[str, Path], dataset_config: Dict[str, Any]) -> None:
        self.cache_root = Path(cache_root).expanduser()
        if not self.cache_root.is_dir():
            raise FileNotFoundError(
                "Teacher cache root does not exist or is not a directory: {}".format(
                    self.cache_root
                )
            )
        self.clip_length, self.sample_stride, self.window_stride = 16, 1, 8
        self.image_height = int(dataset_config.get("image_height", 448))
        self.image_width = int(dataset_config.get("image_width", 560))
        self.resize_mode = str(dataset_config.get("resize_mode", "resize"))
        self.normalize_mode = str(dataset_config.get("normalize_mode", "zero_one"))
        (
            self.highlight_enabled,
            _,
            self.highlight_mask_directory,
            self.highlight_inpainted_directory,
        ) = parse_highlight_options(dataset_config.get("highlight", {}))
        self.sequences: List[Dict[str, Any]] = []
        self.clips: List[ClipRecord] = []
        identities = set()
        for cache_path in sorted(self.cache_root.rglob("*.npz")):
            with np.load(str(cache_path), allow_pickle=False) as cache:
                if "metadata_json" not in cache:
                    continue
                try:
                    metadata = json.loads(str(cache["metadata_json"].item()))
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        "Invalid metadata_json in teacher cache {}".format(cache_path)
                    ) from error
                frame_paths = metadata.get("frame_paths")
                if not isinstance(frame_paths, list) or len(frame_paths) != 16:
                    raise RuntimeError(
                        "Teacher cache {} cannot index student RGB: metadata_json must "
                        "contain 16 frame_paths".format(cache_path)
                    )
                absolute_ids = [int(value) for value in cache["absolute_frame_ids"].tolist()]
                clip_start = int(cache["clip_start"].item())
                cached_dataset_name = (
                    str(cache["dataset_name"].item())
                    if "dataset_name" in cache else "SCARED"
                )
                dataset_name = str(metadata.get("dataset_name", cached_dataset_name))
                sequence_id = str(cache["sequence_id"].item())
                identity = (dataset_name, sequence_id, clip_start)
                if identity in identities:
                    raise RuntimeError("Duplicate teacher cache clip identity {}".format(identity))
                identities.add(identity)
                if clip_start % 8 or absolute_ids != list(range(absolute_ids[0], absolute_ids[0] + 16)):
                    raise RuntimeError(
                        "Teacher cache {} does not describe a consecutive stride-8 clip".format(
                            cache_path
                        )
                    )
                paths = [str(Path(value).expanduser()) for value in frame_paths]
                teacher_paths = metadata.get("teacher_frame_paths", paths)
                if not isinstance(teacher_paths, list) or len(teacher_paths) != 16:
                    teacher_paths = paths
                sequence = {
                    "dataset_name": dataset_name,
                    "dataset_id": int(metadata.get("dataset_id", -1)),
                    "keyframe_id": str(metadata.get("keyframe_id", "cache")),
                    "sequence_id": sequence_id,
                    "sequence_length": 16,
                    "frame_paths": paths,
                    "teacher_frame_paths": [str(Path(value).expanduser()) for value in teacher_paths],
                    "absolute_frame_ids": absolute_ids,
                    "frame_directory": str(Path(paths[0]).parent),
                    "keyframe_directory": str(
                        Path(paths[0]).parent.parent
                        if Path(paths[0]).parent.name == "student_rgb"
                        else metadata.get("keyframe_directory", Path(paths[0]).parent)
                    ),
                    "preprocessing_identity": str(metadata.get("preprocessing_identity", "teacher_cache_metadata")),
                    "cache_metadata_path": str(cache_path),
                }
                self.sequences.append(sequence)
                self.clips.append(ClipRecord(sequence, tuple(range(16)), clip_start))
        if not self.clips:
            raise RuntimeError(
                "No teacher caches with embedded RGB frame metadata found under {}".format(
                    self.cache_root
                )
            )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.clips[index]
        sequence = record.sequence
        paths = [str(sequence["frame_paths"][item]) for item in record.frame_indices]
        missing = next((path for path in paths if not Path(path).is_file()), None)
        if missing is not None:
            raise FileNotFoundError(
                "Student RGB path embedded in teacher cache is missing: {} "
                "(cache={}). Regenerate the cache on this machine or provide a manifest "
                "with valid RGB frame_paths.".format(missing, sequence["cache_metadata_path"])
            )
        images = torch.stack([
            load_rgb_tensor(
                path, self.image_height, self.image_width,
                self.resize_mode, self.normalize_mode,
            )
            for path in paths
        ])
        absolute_ids = [int(value) for value in sequence["absolute_frame_ids"]]
        sample: Dict[str, Any] = {
            "images": images,
            "frame_paths": paths,
            "frame_names": [Path(path).name for path in paths],
            "frame_indices": torch.tensor(absolute_ids, dtype=torch.long),
            "clip_start": torch.tensor(record.clip_start, dtype=torch.long),
        }
        if self.highlight_enabled:
            materialized = [
                precomputed_highlight_paths(
                    sequence["keyframe_directory"], Path(path).name,
                    self.highlight_mask_directory, self.highlight_inpainted_directory,
                )
                for path in paths
            ]
            missing = next(
                (value for pair in materialized for value in pair if not value.is_file()), None
            )
            if missing is not None:
                raise FileNotFoundError(
                    "Precomputed highlight artifact is missing: {}. "
                    "Run precompute_highlights.py first.".format(missing)
                )
            sample["highlight_masks"] = torch.stack(
                [load_precomputed_highlight_mask_tensor(mask) for mask, _ in materialized]
            )
            sample["inpainted_images"] = torch.stack(
                [load_precomputed_student_rgb_tensor(rgb, "zero_one") for _, rgb in materialized]
            )
        return sample


def make_crossclip_rgb_dataset(
    dataset_config: Dict[str, Any], split: str,
    cache_root: Optional[Union[str, Path]] = None,
) -> Any:
    """Build only C_0, C_8, C_16...; frames inside each clip stay consecutive."""
    config = dict(dataset_config)
    for key in ("random_clip_sampling", "teacher_neighbor_offset"):
        config.pop(key, None)
    config.update(
        {
            "clip_length": 16,
            "sample_stride": 1,
            "window_stride": 8,
            "drop_incomplete_clip": True,
        }
    )
    try:
        dataset = make_scared_rgb_dataset(config, split)
    except (FileNotFoundError, RuntimeError) as error:
        discovery_error = (
            "No usable SCARED temporal RGB sequences discovered" in str(error)
            or "No legacy SCARED or complete canonical sequences were discovered" in str(error)
            or "Missing required SCARED dataset IDs" in str(error)
            or "SCARED dataset root does not exist" in str(error)
        )
        if cache_root is None or not discovery_error:
            raise
        dataset = CacheMetadataRGBDataset(cache_root, config)
        print(
            "RGB discovery fallback: indexed {} stride-8 clips from teacher cache "
            "metadata under {}".format(len(dataset), cache_root)
        )
    if dataset.clip_length != 16 or dataset.sample_stride != 1 or dataset.window_stride != 8:
        raise RuntimeError("Cross-clip dataset failed to enforce length=16 sample_stride=1 window_stride=8")
    if any(int(record.clip_start) % 8 for record in dataset.clips):
        raise RuntimeError("Training dataset contains a clip_start not divisible by 8")
    return dataset


def crossclip_teacher_cache_path(
    cache_root: Union[str, Path], metadata: Dict[str, Any]
) -> Path:
    dataset_name = str(metadata.get("dataset_name", "SCARED"))
    dataset_directory = (
        "dataset_{:02d}".format(int(metadata["dataset_id"]))
        if dataset_name == "SCARED"
        else _safe_name(dataset_name)
    )
    return (
        Path(cache_root)
        / dataset_directory
        / _safe_name(str(metadata["keyframe_id"]))
        / _safe_name(str(metadata["sequence_id"]))
        / "start_{:06d}_len_016_stride_01.npz".format(
            int(metadata["clip_start"])
        )
    )


def build_neighbor_clip_indices(
    clips: Sequence[ClipRecord],
    window_stride: int = 8,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Return dataset indices for C_(t-8), C_(t+8) in the same dataset/sequence."""
    if window_stride != 8:
        raise ValueError("Cross-clip neighbor stride must be 8")
    lookup: Dict[Tuple[str, str, int], int] = {}
    for index, record in enumerate(clips):
        key = (str(record.sequence.get("dataset_name", "SCARED")), str(record.sequence["sequence_id"]), int(record.clip_start))
        if key in lookup:
            raise RuntimeError("Duplicate clip identity {}".format(key))
        lookup[key] = index
    result = []
    for record in clips:
        dataset_name = str(record.sequence.get("dataset_name", "SCARED"))
        sequence_id = str(record.sequence["sequence_id"])
        start = int(record.clip_start)
        result.append(
            (
                lookup.get((dataset_name, sequence_id, start - window_stride)),
                lookup.get((dataset_name, sequence_id, start + window_stride)),
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
    version = str(cache["cache_format_version"].item())
    if version not in SUPPORTED_CROSSCLIP_CACHE_FORMAT_VERSIONS:
        raise RuntimeError("Incompatible cross-clip teacher cache version {!r}".format(version))
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
    if "supervision_height" in cache and "supervision_width" in cache:
        if (int(cache["supervision_height"].item()), int(cache["supervision_width"].item())) != (height, width):
            raise RuntimeError("Cross-clip cache input and supervision resolution disagree")
    # Native teacher geometry is retained and audited when v2 metadata exists.
    if "teacher_input_height" in cache and "teacher_input_width" in cache:
        teacher_shape = (int(cache["teacher_input_height"].item()), int(cache["teacher_input_width"].item()))
        if min(teacher_shape) <= 0 or teacher_shape[0] < height or teacher_shape[1] < width:
            raise RuntimeError("Invalid native teacher input resolution {}".format(teacher_shape))
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
    cached_frame_names = [str(value) for value in cache["frame_names"].tolist()]
    if any(not value for value in cached_frame_names) or len(set(cached_frame_names)) != 16:
        raise RuntimeError("Cross-clip cache frame_names must contain 16 unique names")
    declared_frame_names = cache_metadata.get("frame_names")
    if declared_frame_names is not None and list(declared_frame_names) != cached_frame_names:
        raise RuntimeError("Cross-clip cache frame_names disagree with metadata_json")
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
        if "dataset_name" in cache:
            checks["dataset_name"] = str(metadata.get("dataset_name", "SCARED"))
        for key, expected in checks.items():
            if cache[key].item() != expected:
                raise RuntimeError("Cross-clip cache metadata mismatch for {}".format(key))
        if cache["absolute_frame_ids"].tolist() != list(metadata["frame_indices"]):
            raise RuntimeError("Cross-clip cache absolute frame IDs do not match RGB clip")
        # Processed student_rgb files may be renamed to 000000.png while an
        # existing cache retains source-frame names. Sequence identity,
        # clip_start and absolute_frame_ids are the stable cross-preprocess key.


def _load_overlap_side(
    path: Path,
    teacher_metadata: Dict[str, Any],
    current_metadata: Dict[str, Any],
    student_absolute_ids: Sequence[int],
    side: str,
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "Neighbor teacher cache missing: dataset={} sequence={} clip_start={} "
            "expected_neighbor_start={} expected_cache_path={}".format(
                teacher_metadata.get("dataset_name", teacher_metadata.get("dataset_id")),
                teacher_metadata["sequence_id"],
                current_metadata["clip_start"],
                teacher_metadata["clip_start"],
                path,
            )
        )
    with np.load(str(path), allow_pickle=False) as cache:
        # Full cache integrity is checked once by audit_vggtoda3.py before a
        # run.  Do not call validate_crossclip_teacher_cache here: it reads and
        # scans every dense member (including both unused XYZ maps) for every
        # sample and every epoch.  The training hot path only reads supervision
        # consumed by the loss plus the IDs required for exact overlap mapping.
        teacher_slice = slice(8, 16) if side == "left" else slice(0, 8)
        student_slice = slice(0, 8) if side == "left" else slice(8, 16)
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
            "depth": tensor_from_numpy_buffer(cache["depth"][teacher_slice]),
            "confidence": tensor_from_numpy_buffer(cache["confidence"][teacher_slice]),
            "valid_mask": tensor_from_numpy_buffer(cache["valid_mask"][teacher_slice]),
            "intrinsics": tensor_from_numpy_buffer(cache["intrinsics"][teacher_slice]),
            "extrinsics": tensor_from_numpy_buffer(cache["extrinsics"][teacher_slice]),
            "absolute_frame_ids": torch.tensor(actual_ids, dtype=torch.long),
            "student_local_indices": torch.arange(0, 8, dtype=torch.long)
            if side == "left" else torch.arange(8, 16, dtype=torch.long),
            "teacher_local_indices": torch.arange(8, 16, dtype=torch.long)
            if side == "left" else torch.arange(0, 8, dtype=torch.long),
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
        "depth": torch.zeros(8, height, width),
        "confidence": torch.zeros(8, height, width),
        "valid_mask": torch.zeros(8, height, width, dtype=torch.bool),
        "intrinsics": torch.zeros(8, 3, 3),
        "extrinsics": torch.zeros(8, 3, 4),
        "absolute_frame_ids": torch.full((8,), -1, dtype=torch.long),
        "student_local_indices": torch.full((8,), -1, dtype=torch.long),
        "teacher_local_indices": torch.full((8,), -1, dtype=torch.long),
        "clip_start": torch.tensor(-1),
        "sequence_id": sequence_id,
        "cache_path": "",
    }


class ScaredCrossClipProjectionDataset(Dataset):
    """Load C_s RGB plus only the required C_(s-8)/C_(s+8) cache slices."""

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
        if int(rgb_dataset.window_stride) != 8:
            raise ValueError("RGB dataset window_stride must be 8")
        self.neighbors = build_neighbor_clip_indices(rgb_dataset.clips, rgb_dataset.window_stride)

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
                metadata,
                absolute_ids,
                side,
            )

        highlight = sample.get(
            "highlight_masks",
            torch.zeros(16, 1, *shape, dtype=sample["images"].dtype),
        )
        clean = sample.get("inpainted_images")
        if clean is None:
            clean = sample["images"].clamp(0.0, 1.0)
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
    "extrinsics",
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
