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
from datasets.multidataset import TeacherClipInputDataset
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
        temporal_length = int(student_ids.numel())
        if student_ids.ndim != 1 or tuple(teacher_ids.shape) != (temporal_length,):
            raise RuntimeError(
                "Student and teacher absolute IDs must contain the same temporal length"
            )
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
            "depth": (temporal_length,) + cache_shape,
            "confidence": (temporal_length,) + cache_shape,
            "valid_mask": (temporal_length,) + cache_shape,
            "intrinsics": (temporal_length, 3, 3),
            "extrinsics": (temporal_length, 3, 4),
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
        online_teacher_attention: bool = False,
    ) -> None:
        self.rgb_dataset = rgb_dataset
        self.cache_root = Path(cache_root)
        self.expected_base_checkpoint = expected_base_checkpoint
        self.online_teacher_attention = bool(online_teacher_attention)
        self.teacher_rgb_dataset = (
            TeacherClipInputDataset(rgb_dataset)
            if self.online_teacher_attention
            else None
        )
        if int(rgb_dataset.clip_length) < 2 or int(rgb_dataset.sample_stride) <= 0:
            raise ValueError("Direct distillation requires at least two sampled RGB frames")
        if int(rgb_dataset.window_stride) not in (1, 8):
            raise ValueError("RGB candidates must use window_stride 1 or 8")
        self.rgb_indices: List[int] = []
        self.cache_paths: List[Path] = []
        self.skipped_off_stride = 0
        self.skipped_without_cache = 0
        for rgb_index in range(len(rgb_dataset)):
            metadata = clip_metadata(rgb_dataset, rgb_index)
            # clip_start is a zero-based position within THIS video, not the
            # source filename ID or the candidate's global dataset index.
            # A dense teacher-cache root must behave like a stride-8 root.
            start = int(metadata["clip_start"])
            if start < 0 or start % 8:
                self.skipped_off_stride += 1
                continue
            path = crossclip_teacher_cache_path(self.cache_root, metadata)
            if path.is_file():
                self.rgb_indices.append(rgb_index)
                self.cache_paths.append(path)
            else:
                self.skipped_without_cache += 1
        if not self.rgb_indices:
            raise RuntimeError(
                "No RGB clip has an exactly matching raw teacher cache under {}".format(
                    self.cache_root
                )
            )

    def __len__(self) -> int:
        return len(self.rgb_indices)

    def metadata(self, index: int) -> Dict[str, Any]:
        return clip_metadata(self.rgb_dataset, self.rgb_indices[index])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rgb_index = self.rgb_indices[index]
        sample = self.rgb_dataset[rgb_index]
        metadata = clip_metadata(self.rgb_dataset, rgb_index)
        images = sample["images"]
        temporal_length = int(self.rgb_dataset.clip_length)
        if images.ndim != 4 or tuple(images.shape[:2]) != (temporal_length, 3):
            raise RuntimeError("Student RGB must have shape [T,3,H,W]")
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
            torch.zeros(temporal_length, 1, *images.shape[-2:], dtype=torch.bool),
        ).bool()
        clean = sample.get("inpainted_images", images.clamp(0.0, 1.0))
        result = {
            "images": images,
            "clean_images": clean,
            "highlight_masks": highlight,
            "absolute_frame_ids": absolute_ids,
            "clip_start": sample["clip_start"].to(torch.long),
            "sequence_id": str(metadata["sequence_id"]),
            "teacher": teacher,
        }
        if self.teacher_rgb_dataset is not None:
            teacher_images, _ = self.teacher_rgb_dataset.load_images(rgb_index)
            if tuple(teacher_images.shape) != (temporal_length, 3, 1024, 1280):
                raise RuntimeError(
                    "Online VGGT-Omega RGB must have shape [T,3,1024,1280]; got {}"
                    .format(tuple(teacher_images.shape))
                )
            result["teacher_images"] = teacher_images
        return result


class FullOnlineTeacherDistillationDataset(Dataset):
    """Load one ordinary RGB clip for both Student and a cache-free Teacher."""

    def __init__(
        self,
        rgb_dataset: Any,
        teacher_input_height: int,
        teacher_input_width: int,
    ) -> None:
        self.rgb_dataset = rgb_dataset
        self.teacher_input_height = int(teacher_input_height)
        self.teacher_input_width = int(teacher_input_width)
        if self.teacher_input_height <= 0 or self.teacher_input_width <= 0:
            raise ValueError("Teacher input dimensions must be positive")
        clip_length = int(rgb_dataset.clip_length)
        sample_stride = int(rgb_dataset.sample_stride)
        window_stride = int(rgb_dataset.window_stride)
        if clip_length < 2:
            raise ValueError("Full-online distillation requires at least two frames")
        if sample_stride <= 0:
            raise ValueError(
                "Full-online distillation requires a positive sample_stride; got {}"
                .format(sample_stride)
            )
        if window_stride <= 0:
            raise ValueError("Full-online distillation requires a positive window_stride")
        # Resize each decoded source frame on CPU before stacking the temporal
        # clip, so full-online training never constructs or transfers a native
        # 1024x1280 Teacher clip when inference uses a smaller grid.
        self.teacher_rgb_dataset = TeacherClipInputDataset(
            rgb_dataset,
            output_shape=(self.teacher_input_height, self.teacher_input_width),
        )
        if not len(rgb_dataset):
            raise RuntimeError("No complete full-online training clips were discovered")

    def __len__(self) -> int:
        return len(self.rgb_dataset)

    def metadata(self, index: int) -> Dict[str, Any]:
        return clip_metadata(self.rgb_dataset, index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.rgb_dataset[index]
        metadata = self.metadata(index)
        images = sample["images"]
        temporal_length = int(self.rgb_dataset.clip_length)
        if images.ndim != 4 or tuple(images.shape[:2]) != (temporal_length, 3):
            raise RuntimeError(
                "Full-online Student RGB must have shape [T,3,H,W]; got {}".format(
                    tuple(images.shape)
                )
            )
        absolute_ids = sample["frame_indices"].to(torch.long)
        expected_ids = torch.tensor(metadata["frame_indices"], dtype=torch.long)
        if tuple(absolute_ids.shape) != (temporal_length,) or not torch.equal(
            absolute_ids.cpu(), expected_ids
        ):
            raise RuntimeError("Full-online RGB frame IDs do not match clip metadata")
        sample_stride = int(self.rgb_dataset.sample_stride)
        if any(
            int(right) != int(left) + sample_stride
            for left, right in zip(absolute_ids[:-1], absolute_ids[1:])
        ):
            raise RuntimeError(
                "Full-online frame IDs do not match configured sample_stride={}"
                .format(sample_stride)
            )

        teacher_images, teacher_paths = self.teacher_rgb_dataset.load_images(index)
        expected_teacher_shape = (
            temporal_length,
            3,
            self.teacher_input_height,
            self.teacher_input_width,
        )
        if tuple(teacher_images.shape) != expected_teacher_shape:
            raise RuntimeError(
                "Full-online Teacher RGB must have shape {}; got {}".format(
                    expected_teacher_shape, tuple(teacher_images.shape)
                )
            )
        expected_teacher_paths = [str(value) for value in metadata["teacher_frame_paths"]]
        if teacher_paths != expected_teacher_paths:
            raise RuntimeError("Full-online Teacher frame paths do not match Student clip metadata")

        highlight = sample.get(
            "highlight_masks",
            torch.zeros(temporal_length, 1, *images.shape[-2:], dtype=torch.bool),
        ).bool()
        clean = sample.get("inpainted_images", images.clamp(0.0, 1.0))
        return {
            "images": images,
            "clean_images": clean,
            "highlight_masks": highlight,
            "absolute_frame_ids": absolute_ids,
            "teacher_absolute_frame_ids": absolute_ids.clone(),
            "clip_start": sample["clip_start"].to(torch.long),
            "sequence_id": str(metadata["sequence_id"]),
            "teacher_sequence_id": str(metadata["sequence_id"]),
            "teacher_images": teacher_images,
            "teacher_frame_paths": teacher_paths,
        }


def direct_teacher_distillation_collate(
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty direct-distillation batch")
    batch = {
        "images": torch.stack([sample["images"] for sample in samples]),
        "clean_images": torch.stack([sample["clean_images"] for sample in samples]),
        "highlight_masks": torch.stack([sample["highlight_masks"] for sample in samples]),
        "absolute_frame_ids": torch.stack([sample["absolute_frame_ids"] for sample in samples]),
        "clip_start": torch.stack([sample["clip_start"] for sample in samples]),
        "sequence_id": [sample["sequence_id"] for sample in samples],
    }
    cached_flags = ["teacher" in sample for sample in samples]
    if any(cached_flags) and not all(cached_flags):
        raise RuntimeError("Cached Teacher availability differs within a batch")
    if all(cached_flags):
        teacher: Dict[str, Any] = {
            key: torch.stack([sample["teacher"][key] for sample in samples])
            for key in TEACHER_TENSOR_KEYS
        }
        teacher["sequence_id"] = [sample["teacher"]["sequence_id"] for sample in samples]
        teacher["cache_path"] = [sample["teacher"]["cache_path"] for sample in samples]
        batch["teacher"] = teacher
    online_flags = ["teacher_images" in sample for sample in samples]
    if any(online_flags) and not all(online_flags):
        raise RuntimeError("Online Teacher RGB availability differs within a batch")
    if all(online_flags):
        batch["teacher_images"] = torch.stack(
            [sample["teacher_images"] for sample in samples]
        )
        batch["teacher_absolute_frame_ids"] = torch.stack(
            [sample.get("teacher_absolute_frame_ids", sample["absolute_frame_ids"])
             for sample in samples]
        )
        batch["teacher_sequence_id"] = [
            sample.get("teacher_sequence_id", sample["sequence_id"])
            for sample in samples
        ]
    return batch


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
    "FullOnlineTeacherDistillationDataset",
    "build_direct_teacher_distillation_dataloader",
    "direct_teacher_distillation_collate",
]
