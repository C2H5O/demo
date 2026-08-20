"""Strict two-frame SCARED pairs and versioned VGGT-Omega pair caches."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.ground_truth import frame_id, load_clip_ground_truth
from datasets.scared_clip_dataset import make_scared_rgb_dataset
from datasets.scared_dataset import ScaredTemporalRGBDataset, seed_worker


PAIR_CACHE_FORMAT_VERSION = "vggtomast3r-pair-v1"
PAIR_COORDINATE_CONVENTION = "camera-from-world; both pts3d targets expressed in reference-camera coordinates"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def make_scared_pair_rgb_dataset(
    dataset_config: Dict[str, Any], split: str
) -> ScaredTemporalRGBDataset:
    """Reuse the temporal loader while enforcing ordered ``(t,t+stride)`` pairs."""
    config = dict(dataset_config)
    if not bool(config.pop("pair_mode", False)):
        raise ValueError("vggtomast3r requires dataset.pair_mode=true")
    pair_stride = int(config.pop("pair_stride", 2))
    pair_step = int(config.pop("pair_step", 1))
    if pair_stride <= 0 or pair_step <= 0:
        raise ValueError("pair_stride and pair_step must be positive")
    configured_clip_length = config.pop("clip_length", None)
    if configured_clip_length not in (None, 2):
        raise ValueError("pair mode rejects clip_length other than 2")
    config["clip_length"] = 2
    config["sample_stride"] = pair_stride
    config["window_stride"] = pair_step
    config["drop_incomplete_clip"] = True
    dataset = make_scared_rgb_dataset(config, split)
    dataset.pair_stride = pair_stride
    dataset.pair_step = pair_step
    return dataset


def pair_metadata(dataset: ScaredTemporalRGBDataset, index: int) -> Dict[str, Any]:
    record = dataset.clips[index]
    if len(record.frame_indices) != 2:
        raise RuntimeError("Pair dataset produced {} frames".format(len(record.frame_indices)))
    index_a, index_b = map(int, record.frame_indices)
    if index_b - index_a != int(dataset.sample_stride):
        raise RuntimeError("Pair direction/stride changed unexpectedly")
    sequence = record.sequence
    paths = [str(sequence["frame_paths"][i]) for i in (index_a, index_b)]
    names = [Path(path).name for path in paths]
    return {
        "dataset_id": int(sequence["dataset_id"]),
        "keyframe_id": str(sequence["keyframe_id"]),
        "sequence_id": str(sequence["sequence_id"]),
        "sequence_length": int(sequence["sequence_length"]),
        "frame_id_a": frame_id(names[0]),
        "frame_id_b": frame_id(names[1]),
        "frame_index_a": index_a,
        "frame_index_b": index_b,
        "frame_path_a": paths[0],
        "frame_path_b": paths[1],
        "frame_name_a": names[0],
        "frame_name_b": names[1],
        "frame_paths": paths,
        "frame_names": names,
        "pair_stride": int(dataset.sample_stride),
        "pair_step": int(dataset.window_stride),
    }


def teacher_pair_cache_path(
    cache_root: Union[str, Path], metadata: Dict[str, Any]
) -> Path:
    filename = "pair_{:06d}_{:06d}_stride_{:02d}.npz".format(
        int(metadata["frame_id_a"]),
        int(metadata["frame_id_b"]),
        int(metadata["pair_stride"]),
    )
    return (
        Path(cache_root)
        / "dataset_{:02d}".format(int(metadata["dataset_id"]))
        / _safe_name(str(metadata["keyframe_id"]))
        / filename
    )


REQUIRED_PAIR_CACHE_KEYS = (
    "frame_id_a", "frame_id_b", "frame_name_a", "frame_name_b",
    "pair_stride", "image_shape", "teacher_variant", "depth_a", "depth_b",
    "xyz_local_a", "xyz_local_b", "xyz_global_a", "xyz_global_b",
    "pts3d_a_in_a", "pts3d_b_in_a", "confidence_a", "confidence_b",
    "valid_mask_a", "valid_mask_b", "intrinsics_a", "intrinsics_b",
    "extrinsics_a", "extrinsics_b", "coordinate_convention",
    "cache_format_version", "lora_checkpoint", "metadata_json",
)


def validate_pair_cache(
    cache: "np.lib.npyio.NpzFile",
    metadata: Optional[Dict[str, Any]] = None,
    expected_shape: Optional[tuple[int, int]] = None,
    expected_teacher_variant: Optional[str] = None,
    expected_lora_checkpoint: Optional[str] = None,
) -> None:
    missing = [key for key in REQUIRED_PAIR_CACHE_KEYS if key not in cache]
    if missing:
        raise RuntimeError("Pair cache is missing keys {}".format(missing))
    version = str(cache["cache_format_version"].item())
    if version != PAIR_CACHE_FORMAT_VERSION:
        raise RuntimeError(
            "Stale/incompatible teacher cache version {!r}; expected {!r}. "
            "Eight-frame caches cannot be used for vggtomast3r.".format(
                version, PAIR_CACHE_FORMAT_VERSION
            )
        )
    convention = str(cache["coordinate_convention"].item())
    if convention != PAIR_COORDINATE_CONVENTION:
        raise RuntimeError(
            "Pair cache coordinate convention {!r} != {!r}".format(
                convention, PAIR_COORDINATE_CONVENTION
            )
        )
    variant = str(cache["teacher_variant"].item())
    if expected_teacher_variant is not None and variant != expected_teacher_variant:
        raise RuntimeError(
            "Pair cache teacher variant {!r} != configured {!r}".format(
                variant, expected_teacher_variant
            )
        )
    lora_checkpoint = str(cache["lora_checkpoint"].item())
    if expected_lora_checkpoint is not None and lora_checkpoint != expected_lora_checkpoint:
        raise RuntimeError(
            "Pair cache LoRA checkpoint {!r} != configured {!r}".format(
                lora_checkpoint, expected_lora_checkpoint
            )
        )
    shape = tuple(int(value) for value in cache["image_shape"].tolist())
    if expected_shape is not None and shape != tuple(expected_shape):
        raise RuntimeError("Pair cache resolution {} != {}".format(shape, expected_shape))
    for key in ("pts3d_a_in_a", "pts3d_b_in_a"):
        if tuple(cache[key].shape) != shape + (3,):
            raise RuntimeError("{} has invalid shape {}".format(key, cache[key].shape))
    if metadata is not None:
        checks = {
            "frame_id_a": int(metadata["frame_id_a"]),
            "frame_id_b": int(metadata["frame_id_b"]),
            "frame_name_a": str(metadata["frame_name_a"]),
            "frame_name_b": str(metadata["frame_name_b"]),
            "pair_stride": int(metadata["pair_stride"]),
        }
        for key, expected in checks.items():
            actual = cache[key].item()
            if actual != expected:
                raise RuntimeError(
                    "Pair cache metadata mismatch for {}: {} != {}".format(
                        key, actual, expected
                    )
                )


class ScaredPairDistillDataset(Dataset):
    """Ordered RGB pair, two pair-local teacher targets, and reference GT depth."""

    def __init__(
        self,
        rgb_dataset: ScaredTemporalRGBDataset,
        cache_root: Union[str, Path],
        ground_truth_config: Optional[Dict[str, Any]] = None,
        expected_teacher_variant: str = "lora",
        expected_lora_checkpoint: Optional[str] = None,
    ) -> None:
        self.rgb_dataset = rgb_dataset
        self.cache_root = Path(cache_root)
        self.ground_truth_config = dict(ground_truth_config or {})
        self.expected_teacher_variant = expected_teacher_variant
        self.expected_lora_checkpoint = expected_lora_checkpoint

    def __len__(self) -> int:
        return len(self.rgb_dataset)

    def cache_path(self, index: int) -> Path:
        return teacher_pair_cache_path(self.cache_root, pair_metadata(self.rgb_dataset, index))

    def missing_cache_paths(self, limit: int = 10) -> List[Path]:
        missing: List[Path] = []
        for index in range(len(self)):
            path = self.cache_path(index)
            if not path.is_file():
                missing.append(path)
                if len(missing) >= limit:
                    break
        return missing

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.rgb_dataset[index]
        metadata = pair_metadata(self.rgb_dataset, index)
        path = self.cache_path(index)
        if not path.is_file():
            raise FileNotFoundError("Teacher pair cache missing: {}".format(path))
        height, width = map(int, sample["images"].shape[-2:])
        try:
            with np.load(str(path), allow_pickle=False) as cache:
                validate_pair_cache(
                    cache,
                    metadata,
                    (height, width),
                    self.expected_teacher_variant,
                    self.expected_lora_checkpoint,
                )
                target = {
                    "pts3d_ref": torch.from_numpy(cache["pts3d_a_in_a"].astype(np.float32)),
                    "pts3d_other_in_ref": torch.from_numpy(cache["pts3d_b_in_a"].astype(np.float32)),
                    "confidence_ref": torch.from_numpy(cache["confidence_a"].astype(np.float32)),
                    "confidence_other": torch.from_numpy(cache["confidence_b"].astype(np.float32)),
                    "valid_mask_ref": torch.from_numpy(cache["valid_mask_a"].astype(np.bool_)),
                    "valid_mask_other": torch.from_numpy(cache["valid_mask_b"].astype(np.bool_)),
                }
        except (OSError, ValueError) as error:
            raise RuntimeError("Failed to read pair cache {}: {}".format(path, error)) from error
        result: Dict[str, Any] = {
            "images": sample["images"],
            "target": target,
            "frame_indices": sample["frame_indices"],
            "frame_paths": sample["frame_paths"],
            "frame_names": sample["frame_names"],
            "dataset_id": sample["dataset_id"],
            "keyframe_id": sample["keyframe_id"],
            "sequence_id": sample["sequence_id"],
            "cache_path": str(path),
            "pair_metadata": metadata,
        }
        if bool(self.ground_truth_config.get("enabled", False)):
            candidates: List[Optional[str]] = []
            for key in self.ground_truth_config.get(
                "directory_keys", ["depth_directory", "scene_points_directory"]
            ):
                candidates.append(sample.get(str(key)))
            keyframe_directory = Path(sample["frame_directory"]).parent.parent
            for relative in self.ground_truth_config.get("relative_directories", []):
                candidates.append(str(keyframe_directory / str(relative)))
            depth, valid = load_clip_ground_truth(
                [sample["frame_names"][0]],
                candidates,
                (height, width),
                scale=float(self.ground_truth_config.get("scale", 1.0)),
                channel=int(self.ground_truth_config.get("channel", 0)),
                required=bool(self.ground_truth_config.get("required", True)),
            )
            result["ground_truth_depth_ref"] = depth[0]
            result["ground_truth_valid_mask_ref"] = valid[0]
        return result


def pair_collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty pair batch")
    result: Dict[str, Any] = {
        "images": torch.stack([sample["images"] for sample in samples]),
        "target": {
            key: torch.stack([sample["target"][key] for sample in samples])
            for key in samples[0]["target"]
        },
        "frame_indices": torch.stack([sample["frame_indices"] for sample in samples]),
        "dataset_id": torch.stack([sample["dataset_id"] for sample in samples]),
    }
    for key in ("frame_paths", "frame_names", "keyframe_id", "sequence_id", "cache_path", "pair_metadata"):
        result[key] = [sample[key] for sample in samples]
    for key in ("ground_truth_depth_ref", "ground_truth_valid_mask_ref"):
        present = [key in sample for sample in samples]
        if any(present) and not all(present):
            raise ValueError("{} missing from part of pair batch".format(key))
        if all(present):
            result[key] = torch.stack([sample[key] for sample in samples])
    return result


def build_pair_dataloader(
    dataset: ScaredPairDistillDataset,
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
        "collate_fn": pair_collate,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers:
        kwargs["prefetch_factor"] = int(loader_config.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


__all__ = [
    "PAIR_CACHE_FORMAT_VERSION", "PAIR_COORDINATE_CONVENTION",
    "REQUIRED_PAIR_CACHE_KEYS", "ScaredPairDistillDataset",
    "build_pair_dataloader", "make_scared_pair_rgb_dataset", "pair_metadata",
    "teacher_pair_cache_path", "validate_pair_cache",
]
