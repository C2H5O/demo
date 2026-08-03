"""PyTorch Dataset and DataLoader builder for temporal left-camera SCARED RGB clips."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets.collate import scared_collate
from datasets.scared_discovery import SequenceRecord, discover_scared_sequences
from datasets.scared_manifest import load_scared_manifest, resolve_manifest_sequences
from datasets.transforms import load_rgb_tensor, unnormalize_image
from datasets.highlight import HighlightDetectionConfig, SpecularHighlightProcessor


@dataclass(frozen=True)
class ClipRecord:
    """Index-only descriptor for one fixed-length clip from one RGB sequence."""

    sequence: Dict[str, Any]
    frame_indices: Tuple[int, ...]
    clip_start: int


def _validate_temporal_config(clip_length: int, sample_stride: int, window_stride: int) -> None:
    if clip_length <= 0:
        raise ValueError("clip_length must be positive, got {}".format(clip_length))
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive, got {}".format(sample_stride))
    if window_stride <= 0:
        raise ValueError("window_stride must be positive, got {}".format(window_stride))


class ScaredTemporalRGBDataset(Dataset):
    """Build [T,3,H,W] RGB clips without loading video, geometry, or model outputs."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        frame_source: str = "auto",
        clip_length: int = 8,
        sample_stride: int = 1,
        window_stride: int = 4,
        drop_incomplete_clip: bool = True,
        image_height: int = 448,
        image_width: int = 448,
        resize_mode: str = "resize",
        normalize_mode: str = "imagenet",
        manifest_path: Optional[Union[str, Path]] = None,
        highlight: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        _validate_temporal_config(clip_length, sample_stride, window_stride)
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError("SCARED dataset root does not exist or is not a directory: {}".format(self.root))
        self.split = split.lower()
        self.frame_source = frame_source
        self.clip_length = clip_length
        self.sample_stride = sample_stride
        self.window_stride = window_stride
        self.drop_incomplete_clip = drop_incomplete_clip
        self.image_height = image_height
        self.image_width = image_width
        self.resize_mode = resize_mode
        self.normalize_mode = normalize_mode
        highlight_config = dict(highlight or {})
        self.highlight_processor = None
        if bool(highlight_config.pop("enabled", False)):
            highlight_config["enabled"] = True
            self.highlight_processor = SpecularHighlightProcessor(
                HighlightDetectionConfig(**highlight_config)
            )
        self.malformed_sequences: List[str] = []
        if manifest_path:
            manifest = load_scared_manifest(manifest_path)
            self.sequences = resolve_manifest_sequences(self.root, manifest, self.split)
            self.malformed_sequences = list(manifest.get("malformed_sequences", []))
        else:
            discovered, malformed = discover_scared_sequences(self.root, self.split, self.frame_source, strict=True)
            portable_sequences = [sequence.to_manifest_dict(self.root) for sequence in discovered]
            self.sequences = resolve_manifest_sequences(
                self.root,
                {"sequences": portable_sequences},
                self.split,
            )
            self.malformed_sequences = malformed
        self.clips = self._build_clip_index()
        if not self.clips:
            raise RuntimeError("No complete temporal clips generated under {} for split {}. Check clip_length={}, sample_stride={}, and sequence lengths.".format(self.root, self.split, self.clip_length, self.sample_stride))

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    def _clip_starts(self, sequence_length: int, sequence_id: str) -> List[int]:
        required_span = (self.clip_length - 1) * self.sample_stride
        last_complete_start = sequence_length - required_span - 1
        if last_complete_start < 0:
            raise RuntimeError("Sequence {} has {} frames but needs at least {} for clip_length={} and sample_stride={}".format(sequence_id, sequence_length, required_span + 1, self.clip_length, self.sample_stride))
        starts = list(range(0, last_complete_start + 1, self.window_stride))
        if not self.drop_incomplete_clip and starts[-1] != last_complete_start:
            # Fixed-shape video batches cannot include a partial final clip.
            # Include the final complete window ending at the last source frame.
            starts.append(last_complete_start)
        return starts

    def _build_clip_index(self) -> List[ClipRecord]:
        clips: List[ClipRecord] = []
        for sequence in self.sequences:
            length = int(sequence["sequence_length"])
            if length != len(sequence["frame_paths"]):
                raise RuntimeError("Sequence length mismatch in {}".format(sequence.get("sequence_id", sequence)))
            for start in self._clip_starts(length, str(sequence["sequence_id"])):
                indices = tuple(start + step * self.sample_stride for step in range(self.clip_length))
                clips.append(ClipRecord(sequence=sequence, frame_indices=indices, clip_start=start))
        return clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.clips[index]
        sequence = record.sequence
        frame_paths = [str(sequence["frame_paths"][frame_index]) for frame_index in record.frame_indices]
        images = torch.stack([
            load_rgb_tensor(path, self.image_height, self.image_width, self.resize_mode, self.normalize_mode)
            for path in frame_paths
        ], dim=0)
        sample = {
            "images": images,
            "frame_paths": frame_paths,
            "frame_names": [Path(path).name for path in frame_paths],
            "frame_indices": torch.tensor(record.frame_indices, dtype=torch.long),
            "dataset_id": torch.tensor(int(sequence["dataset_id"]), dtype=torch.long),
            "keyframe_id": str(sequence["keyframe_id"]),
            "sequence_id": str(sequence["sequence_id"]),
            "sequence_length": torch.tensor(int(sequence["sequence_length"]), dtype=torch.long),
            "clip_start": torch.tensor(record.clip_start, dtype=torch.long),
            "clip_length": torch.tensor(self.clip_length, dtype=torch.long),
            "sample_stride": torch.tensor(self.sample_stride, dtype=torch.long),
            "frame_directory": str(sequence["frame_directory"]),
            "calibration_path": sequence.get("calibration_path"),
            "depth_directory": sequence.get("depth_directory"),
            "scene_points_directory": sequence.get("scene_points_directory"),
            "disparity_directory": sequence.get("disparity_directory"),
            "point_cloud_path": sequence.get("point_cloud_path"),
            "video_path": sequence.get("video_path"),
        }
        if self.highlight_processor is not None:
            processed = [
                self.highlight_processor(
                    unnormalize_image(frame, self.normalize_mode)
                )
                for frame in images
            ]
            sample["highlight_masks"] = torch.stack(
                [value["highlight_mask"] for value in processed], dim=0
            )
            sample["inpainted_images"] = torch.stack(
                [value["inpainted_image"] for value in processed], dim=0
            )
        return sample


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy RNGs from the DataLoader-provided PyTorch seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_scared_dataloader(
    dataset: ScaredTemporalRGBDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    seed: int = 42,
    distributed: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> DataLoader:
    """Build a CPU-safe, deterministic DataLoader with optional distributed sampling."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive, got {}".format(batch_size))
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative, got {}".format(num_workers))
    sampler = None
    if distributed:
        if rank is None or world_size is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("distributed=True requires initialized torch.distributed or explicit rank and world_size")
            rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last)
        shuffle = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": scared_collate,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)
