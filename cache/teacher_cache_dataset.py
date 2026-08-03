"""Stable public location for the migrated cache-backed SCARED dataset."""

from datasets.scared_clip_dataset import (
    ScaredDistillDataset,
    build_distill_dataloader,
    teacher_cache_path,
)

TeacherCacheDataset = ScaredDistillDataset

__all__ = [
    "ScaredDistillDataset",
    "TeacherCacheDataset",
    "build_distill_dataloader",
    "teacher_cache_path",
]
