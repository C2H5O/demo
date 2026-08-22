"""Strict two-frame SCARED pairs composed from reusable frame caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.ground_truth import frame_id, load_clip_ground_truth
from datasets.scared_clip_dataset import make_scared_rgb_dataset
from datasets.scared_dataset import ScaredTemporalRGBDataset, seed_worker
from datasets.teacher_frame_cache import (
    compose_teacher_frame_caches,
    frame_metadata_from_pair,
    teacher_frame_cache_path,
)


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


class ScaredPairDistillDataset(Dataset):
    """Ordered RGB pair composed from two independent frame-local teacher caches."""

    def __init__(
        self,
        rgb_dataset: ScaredTemporalRGBDataset,
        cache_root: Union[str, Path],
        ground_truth_config: Optional[Dict[str, Any]] = None,
        expected_base_checkpoint: Optional[str] = None,
    ) -> None:
        self.rgb_dataset = rgb_dataset
        self.cache_root = Path(cache_root)
        self.ground_truth_config = dict(ground_truth_config or {})
        self.expected_base_checkpoint = expected_base_checkpoint

    def __len__(self) -> int:
        return len(self.rgb_dataset)

    def cache_paths(self, index: int) -> List[Path]:
        frames = frame_metadata_from_pair(pair_metadata(self.rgb_dataset, index))
        return [teacher_frame_cache_path(self.cache_root, frame) for frame in frames]

    def missing_cache_paths(self, limit: int = 10) -> List[Path]:
        missing: List[Path] = []
        for index in range(len(self)):
            for path in self.cache_paths(index):
                if not path.is_file() and path not in missing:
                    missing.append(path)
                    if len(missing) >= limit:
                        return missing
        return missing

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.rgb_dataset[index]
        metadata = pair_metadata(self.rgb_dataset, index)
        height, width = map(int, sample["images"].shape[-2:])
        frames = frame_metadata_from_pair(metadata)
        composed = compose_teacher_frame_caches(
            self.cache_root,
            frames,
            (height, width),
            self.expected_base_checkpoint,
        )
        target = {
            "pts3d_ref": composed["xyz_local"][0],
            "pts3d_other_local": composed["xyz_local"][1],
            "confidence_ref": composed["confidence"][0],
            "confidence_other": composed["confidence"][1],
            "valid_mask_ref": composed["valid_mask"][0],
            "valid_mask_other": composed["valid_mask"][1],
        }
        result: Dict[str, Any] = {
            "images": sample["images"],
            "target": target,
            "frame_indices": sample["frame_indices"],
            "frame_paths": sample["frame_paths"],
            "frame_names": sample["frame_names"],
            "dataset_id": sample["dataset_id"],
            "keyframe_id": sample["keyframe_id"],
            "sequence_id": sample["sequence_id"],
            "cache_paths": composed["cache_paths"],
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
    for key in ("frame_paths", "frame_names", "keyframe_id", "sequence_id", "cache_paths", "pair_metadata"):
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
    "ScaredPairDistillDataset",
    "build_pair_dataloader", "make_scared_pair_rgb_dataset", "pair_metadata",
]
