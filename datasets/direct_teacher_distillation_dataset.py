"""Same-clip RGB and raw VGGT-Omega pseudo-label loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.crossclip_teacher_dataset import (
    WORLD_TO_CAMERA_POSE_CONVENTION,
    crossclip_teacher_cache_path,
)
from datasets.scared_clip_dataset import clip_metadata
from datasets.scared_dataset import seed_worker
from datasets.transforms import tensor_from_numpy_buffer


TEACHER_TENSOR_KEYS = (
    "depth",
    "confidence",
    "valid_mask",
    "intrinsics",
    "extrinsics",
    "absolute_frame_ids",
    "clip_start",
)


def _load_same_clip_teacher(
    path: Path,
    metadata: Dict[str, Any],
    student_absolute_ids: torch.Tensor,
    spatial_shape: tuple[int, int],
    expected_base_checkpoint: str,
) -> Dict[str, Any]:
    """Load only fields consumed by direct distillation and fail on any mismatch."""
    if not path.is_file():
        raise FileNotFoundError("Matching raw teacher cache is missing: {}".format(path))
    with np.load(str(path), allow_pickle=False) as cache:
        required = {
            "sequence_id", "clip_start", "absolute_frame_ids", "input_height",
            "input_width", "depth", "confidence", "valid_mask", "intrinsics",
            "extrinsics", "pose_convention", "base_checkpoint", "cache_stage",
        }
        missing = sorted(required - set(cache.files))
        if missing:
            raise RuntimeError("Teacher cache {} is missing {}".format(path, missing))
        teacher_start = int(cache["clip_start"].item())
        student_start = int(metadata["clip_start"])
        if teacher_start != student_start:
            raise RuntimeError(
                "Student clip_start {} != teacher clip_start {} for {}".format(
                    student_start, teacher_start, path
                )
            )
        if str(cache["sequence_id"].item()) != str(metadata["sequence_id"]):
            raise RuntimeError("Student and teacher sequence IDs differ for {}".format(path))
        teacher_ids = torch.tensor(
            cache["absolute_frame_ids"].tolist(), dtype=torch.long
        )
        student_ids = student_absolute_ids.detach().cpu().to(torch.long)
        if tuple(student_ids.shape) != (16,) or tuple(teacher_ids.shape) != (16,):
            raise RuntimeError("Student and teacher absolute IDs must both contain 16 frames")
        if not torch.equal(student_ids, teacher_ids):
            raise RuntimeError(
                "Student absolute_frame_ids {} != teacher absolute_frame_ids {} for {}".format(
                    student_ids.tolist(), teacher_ids.tolist(), path
                )
            )
        cache_shape = (
            int(cache["input_height"].item()), int(cache["input_width"].item())
        )
        if cache_shape != tuple(spatial_shape):
            raise RuntimeError(
                "Student spatial resolution {} != teacher cache resolution {} for {}".format(
                    spatial_shape, cache_shape, path
                )
            )
        expected_shapes = {
            "depth": (16,) + cache_shape,
            "confidence": (16,) + cache_shape,
            "valid_mask": (16,) + cache_shape,
            "intrinsics": (16, 3, 3),
            "extrinsics": (16, 3, 4),
        }
        wrong = {
            key: tuple(cache[key].shape)
            for key, expected in expected_shapes.items()
            if tuple(cache[key].shape) != expected
        }
        if wrong:
            raise RuntimeError(
                "Teacher cache supervision shapes {} do not match {}".format(
                    wrong, expected_shapes
                )
            )
        if str(cache["cache_stage"].item()) != "raw":
            raise RuntimeError("Direct distillation requires a raw teacher cache: {}".format(path))
        if str(cache["pose_convention"].item()) != WORLD_TO_CAMERA_POSE_CONVENTION:
            raise RuntimeError("Teacher cache does not use the required W2C convention")
        if str(cache["base_checkpoint"].item()) != expected_base_checkpoint:
            raise RuntimeError("Teacher cache base checkpoint mismatch for {}".format(path))
        teacher = {
            "depth": tensor_from_numpy_buffer(cache["depth"]),
            "confidence": tensor_from_numpy_buffer(cache["confidence"]),
            "valid_mask": tensor_from_numpy_buffer(cache["valid_mask"]).bool(),
            "intrinsics": tensor_from_numpy_buffer(cache["intrinsics"]),
            "extrinsics": tensor_from_numpy_buffer(cache["extrinsics"]),
            "absolute_frame_ids": teacher_ids,
            "clip_start": torch.tensor(teacher_start, dtype=torch.long),
            "sequence_id": str(cache["sequence_id"].item()),
            "cache_path": str(path),
        }
    return teacher


class DirectTeacherDistillationDataset(Dataset):
    """Pair each legal Student C_n with exactly the existing raw Teacher C_n."""

    def __init__(
        self,
        rgb_dataset: Any,
        cache_root: Union[str, Path],
        expected_base_checkpoint: str,
    ) -> None:
        self.rgb_dataset = rgb_dataset
        self.cache_root = Path(cache_root)
        self.expected_base_checkpoint = expected_base_checkpoint
        if int(rgb_dataset.clip_length) != 16 or int(rgb_dataset.sample_stride) != 1:
            raise ValueError("Direct distillation requires consecutive 16-frame RGB clips")
        if int(rgb_dataset.window_stride) != 8:
            raise ValueError("dataset.window_stride must be the cache sampling stride 8")
        self.rgb_indices: List[int] = []
        self.cache_paths: List[Path] = []
        for rgb_index in range(len(rgb_dataset)):
            metadata = clip_metadata(rgb_dataset, rgb_index)
            path = crossclip_teacher_cache_path(self.cache_root, metadata)
            if path.is_file():
                self.rgb_indices.append(rgb_index)
                self.cache_paths.append(path)
        if not self.rgb_indices:
            raise RuntimeError(
                "No RGB clip has an exactly matching raw teacher cache under {}".format(
                    self.cache_root
                )
            )
        self.skipped_without_cache = len(rgb_dataset) - len(self.rgb_indices)

    def __len__(self) -> int:
        return len(self.rgb_indices)

    def metadata(self, index: int) -> Dict[str, Any]:
        return clip_metadata(self.rgb_dataset, self.rgb_indices[index])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rgb_index = self.rgb_indices[index]
        sample = self.rgb_dataset[rgb_index]
        metadata = clip_metadata(self.rgb_dataset, rgb_index)
        images = sample["images"]
        if images.ndim != 4 or tuple(images.shape[:2]) != (16, 3):
            raise RuntimeError("Student RGB must have shape [16,3,H,W]")
        absolute_ids = sample["frame_indices"].to(torch.long)
        teacher = _load_same_clip_teacher(
            self.cache_paths[index],
            metadata,
            absolute_ids,
            tuple(int(value) for value in images.shape[-2:]),
            self.expected_base_checkpoint,
        )
        highlight = sample.get(
            "highlight_masks",
            torch.zeros(16, 1, *images.shape[-2:], dtype=torch.bool),
        ).bool()
        clean = sample.get("inpainted_images", images.clamp(0.0, 1.0))
        return {
            "images": images,
            "clean_images": clean,
            "highlight_masks": highlight,
            "absolute_frame_ids": absolute_ids,
            "clip_start": sample["clip_start"].to(torch.long),
            "sequence_id": str(metadata["sequence_id"]),
            "teacher": teacher,
        }


def direct_teacher_distillation_collate(
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty direct-distillation batch")
    teacher: Dict[str, Any] = {
        key: torch.stack([sample["teacher"][key] for sample in samples])
        for key in TEACHER_TENSOR_KEYS
    }
    teacher["sequence_id"] = [sample["teacher"]["sequence_id"] for sample in samples]
    teacher["cache_path"] = [sample["teacher"]["cache_path"] for sample in samples]
    return {
        "images": torch.stack([sample["images"] for sample in samples]),
        "clean_images": torch.stack([sample["clean_images"] for sample in samples]),
        "highlight_masks": torch.stack([sample["highlight_masks"] for sample in samples]),
        "absolute_frame_ids": torch.stack([sample["absolute_frame_ids"] for sample in samples]),
        "clip_start": torch.stack([sample["clip_start"] for sample in samples]),
        "sequence_id": [sample["sequence_id"] for sample in samples],
        "teacher": teacher,
    }


def build_direct_teacher_distillation_dataloader(
    dataset: DirectTeacherDistillationDataset,
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
        "collate_fn": direct_teacher_distillation_collate,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers:
        kwargs["prefetch_factor"] = int(loader_config.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


__all__ = [
    "DirectTeacherDistillationDataset",
    "build_direct_teacher_distillation_dataloader",
    "direct_teacher_distillation_collate",
]
