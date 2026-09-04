from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cache.generate_crossclip_teacher_cache import _resolve_start_index
from datasets.crossclip_teacher_dataset import (
    CROSSCLIP_CACHE_FORMAT_VERSION,
    LOCAL_CAMERA_COORDINATE_SYSTEM,
    WORLD_TO_CAMERA_POSE_CONVENTION,
    CacheMetadataRGBDataset,
    crossclip_teacher_cache_path,
    make_teacher_cache_rgb_dataset,
    validate_crossclip_teacher_cache,
)
from datasets.direct_teacher_distillation_dataset import (
    DirectTeacherDistillationDataset,
    direct_teacher_distillation_collate,
)
from datasets.scared_clip_dataset import clip_metadata
from datasets.scared_dataset import ClipRecord


BASE_CHECKPOINT = "./checkpoints/vggt_omega/vggt_omega_1b_512.pt"


def _sequence(sequence_id: str, length: int):
    return {
        "dataset_name": "SCARED",
        "dataset_id": 1,
        "keyframe_id": "keyframe_1",
        "sequence_id": sequence_id,
        "sequence_length": length,
        "frame_paths": ["frame_{:06d}.png".format(index) for index in range(length)],
        "absolute_frame_ids": list(range(length)),
    }


def _clips(sequence):
    return [
        ClipRecord(sequence, tuple(range(start, start + 16)), start)
        for start in range(0, sequence["sequence_length"] - 15, 8)
    ]


class _FakeRGBDataset:
    clip_length = 16
    sample_stride = 1
    window_stride = 8
    resize_mode = "resize"
    normalize_mode = "zero_one"

    def __init__(self, sequences, shape=(4, 6)):
        self.sequences = sequences
        self.clips = [clip for sequence in sequences for clip in _clips(sequence)]
        self.shape = shape

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        record = self.clips[index]
        return {
            "images": torch.zeros(16, 3, *self.shape),
            "inpainted_images": torch.full((16, 3, *self.shape), 0.5),
            "highlight_masks": torch.zeros(16, 1, *self.shape),
            "frame_indices": torch.tensor(record.frame_indices),
            "clip_start": torch.tensor(record.clip_start),
        }


def _write_cache(root, dataset, index, stage="raw"):
    metadata = clip_metadata(dataset, index)
    height, width = dataset.shape
    depth = np.ones((16, height, width), dtype=np.float32)
    points = np.zeros((16, height, width, 3), dtype=np.float32)
    points[..., 2] = depth
    metadata_record = {
        **metadata,
        "minimum_valid_fraction": 0.001,
        "valid_fraction_per_frame": [1.0] * 16,
        "valid_depth_min": 1.0,
        "valid_depth_max": 1.0,
        "valid_confidence_mean": 1.0,
    }
    path = crossclip_teacher_cache_path(root, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dataset_name=np.asarray(metadata["dataset_name"]),
        sequence_id=np.asarray(metadata["sequence_id"]),
        clip_start=np.asarray(metadata["clip_start"]),
        absolute_frame_ids=np.asarray(metadata["frame_indices"]),
        frame_names=np.asarray(metadata["frame_names"]),
        input_height=np.asarray(height),
        input_width=np.asarray(width),
        teacher_input_height=np.asarray(1024),
        teacher_input_width=np.asarray(1280),
        supervision_height=np.asarray(height),
        supervision_width=np.asarray(width),
        depth=depth,
        xyz_local=points,
        xyz_global=points.copy(),
        confidence=np.ones_like(depth),
        valid_mask=np.ones_like(depth, dtype=np.bool_),
        highlight_mask=np.zeros_like(depth, dtype=np.bool_),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0),
        extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None, :3], 16, axis=0),
        pose_convention=np.asarray(WORLD_TO_CAMERA_POSE_CONVENTION),
        point_coordinate_system=np.asarray(LOCAL_CAMERA_COORDINATE_SYSTEM),
        teacher_variant=np.asarray("base"),
        base_checkpoint=np.asarray(BASE_CHECKPOINT),
        cache_stage=np.asarray(stage),
        alignment_scale=np.asarray(1.0, dtype=np.float32),
        cache_format_version=np.asarray(CROSSCLIP_CACHE_FORMAT_VERSION),
        metadata_json=np.asarray(json.dumps(metadata_record)),
    )
    return path


def test_same_clip_dataset_loads_all_16_matching_teacher_frames(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 32)])
    path = _write_cache(tmp_path, rgb, 1)
    dataset = DirectTeacherDistillationDataset(rgb, tmp_path, BASE_CHECKPOINT)
    sample = dataset[0]
    assert dataset.rgb_indices == [1]
    assert sample["clip_start"].item() == sample["teacher"]["clip_start"].item() == 8
    assert sample["absolute_frame_ids"].tolist() == list(range(8, 24))
    assert sample["teacher"]["absolute_frame_ids"].tolist() == list(range(8, 24))
    assert sample["teacher"]["depth"].shape == (16, 4, 6)
    assert sample["teacher"]["extrinsics"].shape == (16, 3, 4)
    assert "xyz_local" not in sample["teacher"]
    assert "xyz_global" not in sample["teacher"]
    assert sample["teacher"]["cache_path"] == str(path)


def test_online_attention_loads_native_teacher_rgb_but_not_attention_cache(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 16)])
    _write_cache(tmp_path, rgb, 0)
    dataset = DirectTeacherDistillationDataset(
        rgb, tmp_path, BASE_CHECKPOINT, online_teacher_attention=True
    )

    class FakeTeacherRGB:
        def load_images(self, index):
            del index
            return torch.zeros(1).expand(16, 3, 1024, 1280), ["teacher.png"] * 16

    dataset.teacher_rgb_dataset = FakeTeacherRGB()
    sample = dataset[0]
    assert sample["teacher_images"].shape == (16, 3, 1024, 1280)
    assert "attention" not in sample["teacher"]


def test_dataset_filters_legal_rgb_clips_without_matching_cache(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 40)])
    _write_cache(tmp_path, rgb, 2)
    dataset = DirectTeacherDistillationDataset(rgb, tmp_path, BASE_CHECKPOINT)
    assert len(dataset) == 1
    assert dataset.rgb_indices == [2]
    assert dataset.skipped_without_cache == 3


def test_absolute_frame_mismatch_raises_in_dataset_layer(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 16)])
    path = _write_cache(tmp_path, rgb, 0)
    with np.load(path, allow_pickle=False) as cache:
        arrays = {key: cache[key].copy() for key in cache.files}
    arrays["absolute_frame_ids"][5] = 99
    np.savez(path, **arrays)
    dataset = DirectTeacherDistillationDataset(rgb, tmp_path, BASE_CHECKPOINT)
    with pytest.raises(RuntimeError, match="absolute_frame_ids"):
        dataset[0]


def test_aligned_cache_is_rejected_by_direct_training(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 16)])
    _write_cache(tmp_path, rgb, 0, stage="aligned")
    dataset = DirectTeacherDistillationDataset(rgb, tmp_path, BASE_CHECKPOINT)
    with pytest.raises(RuntimeError, match="raw teacher cache"):
        dataset[0]


def test_collate_contains_one_teacher_without_point_maps(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 16)])
    _write_cache(tmp_path, rgb, 0)
    dataset = DirectTeacherDistillationDataset(rgb, tmp_path, BASE_CHECKPOINT)
    batch = direct_teacher_distillation_collate([dataset[0]])
    assert batch["teacher"]["depth"].shape == (1, 16, 4, 6)
    assert batch["teacher"]["intrinsics"].shape == (1, 16, 3, 3)
    assert "teacher_left" not in batch and "teacher_right" not in batch
    assert "xyz_local" not in batch["teacher"]


def test_training_hot_path_does_not_read_teacher_xyz(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 16)])
    path = _write_cache(tmp_path, rgb, 0)
    with np.load(path, allow_pickle=False) as cache:
        arrays = {key: cache[key].copy() for key in cache.files}
    arrays["xyz_local"][:] = np.nan
    arrays["xyz_global"][:] = np.nan
    np.savez(path, **arrays)
    dataset = DirectTeacherDistillationDataset(rgb, tmp_path, BASE_CHECKPOINT)
    assert torch.isfinite(dataset[0]["teacher"]["depth"]).all()


def test_explicit_full_cache_audit_still_validates_legacy_format(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 16)])
    path = _write_cache(tmp_path, rgb, 0)
    with np.load(path, allow_pickle=False) as cache:
        validate_crossclip_teacher_cache(
            cache, clip_metadata(rgb, 0), rgb.shape, BASE_CHECKPOINT, "raw"
        )


def test_cache_resume_start_and_metadata_fallback_remain_compatible(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 32)])
    assert _resolve_start_index(rgb, 1, None, None, None) == 1
    cache_root = tmp_path / "cache" / "train"
    for index in range(len(rgb)):
        _write_cache(cache_root, rgb, index)
    empty = tmp_path / "processed"
    empty.mkdir()
    fallback = make_teacher_cache_rgb_dataset(
        {
            "root": str(empty), "frame_source": "auto", "clip_length": 16,
            "sample_stride": 1, "window_stride": 8, "drop_incomplete_clip": True,
            "image_height": 448, "image_width": 560, "resize_mode": "resize",
            "normalize_mode": "zero_one",
        },
        "train",
        cache_root=cache_root,
    )
    assert isinstance(fallback, CacheMetadataRGBDataset)
    assert [record.clip_start for record in fallback.clips] == [0, 8, 16]
