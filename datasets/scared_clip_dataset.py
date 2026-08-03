"""SCARED RGB clips paired with offline VGGT-Omega teacher caches."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets.scared_dataset import ScaredTemporalRGBDataset, seed_worker
from datasets.ground_truth import load_clip_ground_truth


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def _absolute_path(root: Path, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def make_scared_rgb_dataset(dataset_config: Dict[str, Any], split: str) -> ScaredTemporalRGBDataset:
    """Build the RGB dataset and normalize every discovered path to absolute."""
    config = dict(dataset_config)
    config.pop("name", None)
    config.pop("cache_root", None)
    config.pop("train_manifest_path", None)
    config.pop("test_manifest_path", None)
    config.pop("ground_truth", None)
    config["split"] = split
    manifest_key = "{}_manifest_path".format(split)
    manifest_path = dataset_config.get(manifest_key)
    config["manifest_path"] = manifest_path
    dataset = ScaredTemporalRGBDataset(**config)

    # The discovery implementation stores portable relative paths. Resolve them
    # once here so teacher export and student training share exact frame paths.
    for sequence in dataset.sequences:
        sequence["frame_paths"] = [
            _absolute_path(dataset.root, str(path)) for path in sequence["frame_paths"]
        ]
        for key in (
            "keyframe_directory",
            "frame_directory",
            "calibration_path",
            "depth_directory",
            "disparity_directory",
            "frame_data_directory",
            "reprojection_directory",
            "scene_points_directory",
            "point_cloud_path",
            "video_path",
        ):
            if key in sequence:
                sequence[key] = _absolute_path(dataset.root, sequence.get(key))
    return dataset


def clip_metadata(dataset: ScaredTemporalRGBDataset, index: int) -> Dict[str, Any]:
    """Read one clip identity without decoding its RGB images."""
    record = dataset.clips[index]
    sequence = record.sequence
    frame_paths = [str(sequence["frame_paths"][frame_index]) for frame_index in record.frame_indices]
    return {
        "dataset_id": int(sequence["dataset_id"]),
        "keyframe_id": str(sequence["keyframe_id"]),
        "sequence_id": str(sequence["sequence_id"]),
        "sequence_length": int(sequence["sequence_length"]),
        "clip_start": int(record.clip_start),
        "clip_length": int(dataset.clip_length),
        "sample_stride": int(dataset.sample_stride),
        "frame_indices": list(record.frame_indices),
        "frame_paths": frame_paths,
        "frame_names": [Path(path).name for path in frame_paths],
    }


def teacher_cache_path(cache_root: Union[str, Path], metadata: Dict[str, Any]) -> Path:
    """Build a stable cache filename from the temporal clip identity."""
    filename = "start_{:06d}_len_{:03d}_stride_{:02d}.npz".format(
        int(metadata["clip_start"]),
        int(metadata["clip_length"]),
        int(metadata["sample_stride"]),
    )
    return (
        Path(cache_root)
        / "dataset_{:02d}".format(int(metadata["dataset_id"]))
        / _safe_name(str(metadata["keyframe_id"]))
        / filename
    )


def _resize_map(tensor: torch.Tensor, height: int, width: int, mode: str) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[-1] == 3:
        if tensor.shape[1:3] == (height, width):
            return tensor
        data = tensor.permute(0, 3, 1, 2)
        data = F.interpolate(data, size=(height, width), mode=mode, align_corners=False)
        return data.permute(0, 2, 3, 1).contiguous()
    if tensor.ndim == 3:
        if tensor.shape[1:] == (height, width):
            return tensor
        data = tensor.unsqueeze(1)
        if mode == "nearest":
            data = F.interpolate(data, size=(height, width), mode=mode)
        else:
            data = F.interpolate(data, size=(height, width), mode=mode, align_corners=False)
        return data[:, 0]
    raise ValueError("Expected [T,H,W,3] or [T,H,W], got {}".format(tuple(tensor.shape)))


class ScaredDistillDataset(Dataset):
    """Return normalized student RGB and matching cached teacher geometry."""

    REQUIRED_CACHE_KEYS = ("xyz_global", "xyz_local", "conf_global", "conf_local", "valid_mask")

    def __init__(
        self,
        rgb_dataset: ScaredTemporalRGBDataset,
        cache_root: Union[str, Path],
        ground_truth_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rgb_dataset = rgb_dataset
        self.cache_root = Path(cache_root)
        self.ground_truth_config = dict(ground_truth_config or {})

    def __len__(self) -> int:
        return len(self.rgb_dataset)

    def cache_path(self, index: int) -> Path:
        return teacher_cache_path(self.cache_root, clip_metadata(self.rgb_dataset, index))

    def missing_cache_paths(self, limit: int = 10) -> List[Path]:
        missing = []
        for index in range(len(self)):
            path = self.cache_path(index)
            if not path.is_file():
                missing.append(path)
                if len(missing) >= limit:
                    break
        return missing

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.rgb_dataset[index]
        cache_path = self.cache_path(index)
        if not cache_path.is_file():
            raise FileNotFoundError(
                "Teacher cache missing for {}. Run the cache exporter first: {}".format(
                    sample["sequence_id"], cache_path
                )
            )
        try:
            with np.load(str(cache_path), allow_pickle=False) as cache:
                missing = [key for key in self.REQUIRED_CACHE_KEYS if key not in cache]
                if missing:
                    raise RuntimeError("Cache {} is missing keys {}".format(cache_path, missing))
                target = {
                    "xyz_global": torch.from_numpy(cache["xyz_global"].astype(np.float32)),
                    "xyz_local": torch.from_numpy(cache["xyz_local"].astype(np.float32)),
                    "conf_global": torch.from_numpy(cache["conf_global"].astype(np.float32)),
                    "conf_local": torch.from_numpy(cache["conf_local"].astype(np.float32)),
                }
                valid_mask = torch.from_numpy(cache["valid_mask"].astype(np.bool_))
                cached_names = cache["frame_names"].tolist() if "frame_names" in cache else None
        except (OSError, ValueError) as error:
            raise RuntimeError("Failed to read teacher cache {}: {}".format(cache_path, error)) from error

        if cached_names is not None and list(cached_names) != sample["frame_names"]:
            raise RuntimeError(
                "Cache/RGB frame mismatch at {}. Cached {}; current {}".format(
                    cache_path, cached_names, sample["frame_names"]
                )
            )
        height, width = sample["images"].shape[-2:]
        cache_shape = tuple(target["xyz_local"].shape[1:3])
        if cache_shape != (height, width):
            raise RuntimeError(
                "Teacher cache resolution {} does not match student/GT "
                "resolution {} at {}. Regenerate this cache; implicit "
                "teacher-target resizing is disabled.".format(
                    cache_shape, (height, width), cache_path
                )
            )
        target["xyz_global"] = _resize_map(target["xyz_global"], height, width, "bilinear")
        target["xyz_local"] = _resize_map(target["xyz_local"], height, width, "bilinear")
        target["conf_global"] = _resize_map(target["conf_global"], height, width, "bilinear")
        target["conf_local"] = _resize_map(target["conf_local"], height, width, "bilinear")
        valid_mask = _resize_map(valid_mask.float(), height, width, "nearest").bool()
        result = {
            "images": sample["images"],
            "target": target,
            "valid_mask": valid_mask,
            "frame_paths": sample["frame_paths"],
            "frame_names": sample["frame_names"],
            "frame_indices": sample["frame_indices"],
            "dataset_id": sample["dataset_id"],
            "keyframe_id": sample["keyframe_id"],
            "sequence_id": sample["sequence_id"],
            "clip_start": sample["clip_start"],
            "cache_path": str(cache_path),
        }
        if bool(self.ground_truth_config.get("enabled", False)):
            candidates = []
            for key in self.ground_truth_config.get(
                "directory_keys", ["depth_directory", "scene_points_directory"]
            ):
                candidates.append(sample.get(str(key)))
            keyframe_directory = Path(sample["frame_directory"]).parent.parent
            for relative in self.ground_truth_config.get(
                "relative_directories", []
            ):
                candidates.append(str(keyframe_directory / str(relative)))
            ground_truth, ground_truth_valid = load_clip_ground_truth(
                sample["frame_names"],
                candidates,
                (height, width),
                scale=float(self.ground_truth_config.get("scale", 1.0)),
                channel=int(self.ground_truth_config.get("channel", 0)),
                required=bool(self.ground_truth_config.get("required", True)),
            )
            result["ground_truth_depth"] = ground_truth
            result["ground_truth_valid_mask"] = ground_truth_valid
        return result


def distill_collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate fixed tensors while preserving clip metadata as lists."""
    result = {
        "images": torch.stack([sample["images"] for sample in samples]),
        "target": {
            key: torch.stack([sample["target"][key] for sample in samples])
            for key in samples[0]["target"]
        },
        "valid_mask": torch.stack([sample["valid_mask"] for sample in samples]),
        "frame_indices": torch.stack([sample["frame_indices"] for sample in samples]),
        "dataset_id": torch.stack([sample["dataset_id"] for sample in samples]),
        "clip_start": torch.stack([sample["clip_start"] for sample in samples]),
        "frame_paths": [sample["frame_paths"] for sample in samples],
        "frame_names": [sample["frame_names"] for sample in samples],
        "keyframe_id": [sample["keyframe_id"] for sample in samples],
        "sequence_id": [sample["sequence_id"] for sample in samples],
        "cache_path": [sample["cache_path"] for sample in samples],
    }
    optional = ("ground_truth_depth", "ground_truth_valid_mask")
    for key in optional:
        present = [key in sample for sample in samples]
        if any(present) and not all(present):
            raise ValueError("Ground-truth key {!r} is missing from part of the batch".format(key))
        if all(present):
            result[key] = torch.stack([sample[key] for sample in samples])
    return result


def build_distill_dataloader(
    dataset: ScaredDistillDataset,
    loader_config: Dict[str, Any],
    seed: int,
    shuffle: bool,
    distributed: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> DataLoader:
    """Build a deterministic cache-backed loader for student training."""
    num_workers = int(loader_config.get("num_workers", 0))
    sampler = None
    if distributed:
        if rank is None or world_size is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Distributed loading requires an initialized process group")
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": int(loader_config.get("batch_size", 1)),
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(loader_config.get("pin_memory", False)),
        "persistent_workers": bool(loader_config.get("persistent_workers", False)) if num_workers > 0 else False,
        "drop_last": bool(loader_config.get("drop_last", False)),
        "collate_fn": distill_collate,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(loader_config.get("prefetch_factor", 2))
    return DataLoader(**kwargs)
