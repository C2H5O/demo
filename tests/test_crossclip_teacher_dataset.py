from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cache.align_crossclip_teacher_cache import estimate_teacher_overlap_scale
from cache.generate_crossclip_teacher_cache import _resolve_start_index
from datasets.crossclip_teacher_dataset import (
    CROSSCLIP_CACHE_FORMAT_VERSION,
    LOCAL_CAMERA_COORDINATE_SYSTEM,
    WORLD_TO_CAMERA_POSE_CONVENTION,
    ScaredCrossClipProjectionDataset,
    build_neighbor_clip_indices,
    crossclip_projection_collate,
    crossclip_teacher_cache_path,
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

    def __init__(self, sequences, shape=(4, 6)):
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


def _write_cache(root, dataset, index, stage="aligned"):
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
    np.savez_compressed(
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


def test_stride_eight_clip_neighbors_and_exact_overlap_mappings() -> None:
    first = _sequence("sequence_a", 40)
    second = _sequence("sequence_b", 16)
    clips = _clips(first) + _clips(second)
    neighbors = build_neighbor_clip_indices(clips)
    assert len(_clips(first)) == 4
    assert [clip.clip_start for clip in _clips(first)] == [0, 8, 16, 24]
    assert neighbors[1] == (0, 2)
    assert list(clips[1].frame_indices[0:8]) == list(clips[0].frame_indices[8:16])
    assert list(clips[1].frame_indices[8:16]) == list(clips[2].frame_indices[0:8])
    assert neighbors[0] == (None, 1)
    assert neighbors[3] == (2, None)
    assert neighbors[4] == (None, None)


def test_cache_resume_start_resolves_before_existing_cache_scan() -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 32)])
    assert _resolve_start_index(rgb, 1, None, None, None) == 1
    assert _resolve_start_index(rgb, None, 1, "keyframe_1", 8) == 1
    with pytest.raises(ValueError, match="not both"):
        _resolve_start_index(rgb, 1, 1, "keyframe_1", 8)
    with pytest.raises(ValueError, match="No clip matches"):
        _resolve_start_index(rgb, None, 1, "keyframe_1", 99)


def test_crossclip_dataset_loads_only_8_shared_frames(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 40)])
    _write_cache(tmp_path, rgb, 0)
    _write_cache(tmp_path, rgb, 2)
    dataset = ScaredCrossClipProjectionDataset(
        rgb, tmp_path, BASE_CHECKPOINT, expected_stage="aligned"
    )
    sample = dataset[1]
    assert sample["teacher_left"]["exists"]
    assert sample["teacher_right"]["exists"]
    assert sample["teacher_left"]["student_local_indices"].tolist() == list(range(8))
    assert sample["teacher_left"]["teacher_local_indices"].tolist() == list(range(8, 16))
    assert sample["teacher_right"]["student_local_indices"].tolist() == list(range(8, 16))
    assert sample["teacher_right"]["teacher_local_indices"].tolist() == list(range(8))
    assert sample["teacher_left"]["absolute_frame_ids"].tolist() == list(range(8, 16))
    assert sample["teacher_right"]["absolute_frame_ids"].tolist() == list(range(16, 24))


def test_raw_teacher_samples_collate_without_unused_point_maps(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 40)])
    _write_cache(tmp_path, rgb, 0, stage="raw")
    _write_cache(tmp_path, rgb, 2, stage="raw")
    dataset = ScaredCrossClipProjectionDataset(
        rgb, tmp_path, BASE_CHECKPOINT, expected_stage="raw"
    )
    batch = crossclip_projection_collate([dataset[1]])
    assert batch["teacher_left"]["depth"].shape == (1, 8, 4, 6)
    assert batch["teacher_right"]["depth"].shape == (1, 8, 4, 6)
    assert batch["teacher_left"]["extrinsics"].shape == (1, 8, 3, 4)
    assert "xyz_local" not in batch["teacher_left"]
    assert "xyz_local" not in batch["teacher_right"]


def test_single_16_frame_sequence_has_no_projection_teacher(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("only_sequence", 16)])
    dataset = ScaredCrossClipProjectionDataset(
        rgb, tmp_path, BASE_CHECKPOINT, expected_stage="aligned"
    )
    sample = dataset[0]
    assert not sample["teacher_left"]["exists"]
    assert not sample["teacher_right"]["exists"]
    assert not sample["teacher_left"]["valid_mask"].any()


def test_degenerate_teacher_intrinsics_fail_closed(tmp_path) -> None:
    rgb = _FakeRGBDataset([_sequence("sequence_a", 40)])
    path = _write_cache(tmp_path, rgb, 0)
    with np.load(path, allow_pickle=False) as cache:
        arrays = {key: cache[key].copy() for key in cache.files}
    arrays["intrinsics"][:, 0, 0] = 0.0
    np.savez_compressed(path, **arrays)
    dataset = ScaredCrossClipProjectionDataset(
        rgb, tmp_path, BASE_CHECKPOINT, expected_stage="aligned"
    )
    with pytest.raises(RuntimeError, match="focal length"):
        dataset[1]


def test_teacher_overlap_scale_uses_common_frames_and_ignores_highlight() -> None:
    previous = np.full((15, 2, 3), 2.0, dtype=np.float32)
    current = np.ones_like(previous)
    valid = np.ones_like(previous, dtype=np.bool_)
    confidence = np.ones_like(previous)
    highlight = np.zeros_like(previous, dtype=np.bool_)
    current[0, 0, 0] = 1000.0
    highlight[0, 0, 0] = True
    scale, frame_ratios = estimate_teacher_overlap_scale(
        previous,
        current,
        valid,
        valid,
        confidence,
        confidence,
        highlight,
        highlight,
        minimum_valid_pixels=2,
    )
    assert scale == 2.0
    assert len(frame_ratios) == 15
