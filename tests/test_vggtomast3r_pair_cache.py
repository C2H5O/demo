from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cache.generate_teacher_pair_cache import generate_teacher_pair_cache
from datasets.scared_dataset import ClipRecord
from datasets.scared_pair_dataset import (
    ScaredPairDistillDataset,
    make_scared_pair_rgb_dataset,
    pair_metadata,
)
from datasets.teacher_frame_cache import (
    FRAME_CACHE_FORMAT_VERSION,
    FRAME_COORDINATE_CONVENTION,
    REQUIRED_FRAME_CACHE_KEYS,
    compose_teacher_frame_caches,
    frame_metadata_from_clip,
    frame_metadata_from_pair,
    teacher_frame_cache_path,
    validate_teacher_frame_cache,
)


BASE_CHECKPOINT = "./checkpoints/vggt_omega/vggt_omega_1b_512.pt"


class _FakePairDataset:
    def __init__(self, count: int = 10, shape: tuple[int, int] = (4, 6)) -> None:
        sequence = {
            "dataset_id": 1,
            "keyframe_id": "keyframe_1",
            "sequence_id": "seq",
            "sequence_length": count,
            "frame_paths": ["frame_{:06d}.png".format(i) for i in range(count)],
        }
        self.clips = [ClipRecord(sequence, (i, i + 2), i) for i in range(count - 2)]
        self.sample_stride = 2
        self.window_stride = 1
        self.pair_stride = 2
        self.pair_step = 1
        self.shape = shape

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int):
        metadata = pair_metadata(self, index)
        return {
            "images": torch.zeros(2, 3, *self.shape),
            "frame_indices": torch.tensor(
                [metadata["frame_index_a"], metadata["frame_index_b"]]
            ),
            "frame_paths": metadata["frame_paths"],
            "frame_names": metadata["frame_names"],
            "dataset_id": torch.tensor(metadata["dataset_id"]),
            "keyframe_id": metadata["keyframe_id"],
            "sequence_id": metadata["sequence_id"],
            "frame_directory": "frames",
        }


def _frame_arrays(metadata, shape=(4, 6), value=1.0, version=FRAME_CACHE_FORMAT_VERSION):
    h, w = shape
    depth = np.full((h, w), value, dtype=np.float32)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[..., 2] = depth
    arrays = {
        "dataset_id": np.asarray(metadata["dataset_id"]),
        "keyframe_id": np.asarray(metadata["keyframe_id"]),
        "sequence_id": np.asarray(metadata["sequence_id"]),
        "frame_id": np.asarray(metadata["frame_id"]),
        "frame_index": np.asarray(metadata["frame_index"]),
        "frame_name": np.asarray(metadata["frame_name"]),
        "image_shape": np.asarray(shape),
        "teacher_variant": np.asarray("base"),
        "inference_frame_count": np.asarray(1),
        "depth": depth,
        "xyz_local": points,
        "confidence": np.full((h, w), 0.8, dtype=np.float32),
        "valid_mask": np.ones((h, w), dtype=bool),
        "intrinsics": np.eye(3, dtype=np.float32),
        "extrinsics": np.eye(4, dtype=np.float32)[:3],
        "coordinate_convention": np.asarray(FRAME_COORDINATE_CONVENTION),
        "cache_format_version": np.asarray(version),
        "base_checkpoint": np.asarray(BASE_CHECKPOINT),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    assert set(REQUIRED_FRAME_CACHE_KEYS).issubset(arrays)
    return arrays


def _write_frame(root, metadata, value=1.0, version=FRAME_CACHE_FORMAT_VERSION):
    path = teacher_frame_cache_path(root, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **_frame_arrays(metadata, value=value, version=version))
    return path


def test_pair_dataset(monkeypatch) -> None:
    captured = {}

    def fake_make(config, split):
        captured.update(config)
        captured["split"] = split
        return SimpleNamespace(
            sample_stride=config["sample_stride"], window_stride=config["window_stride"]
        )

    monkeypatch.setattr("datasets.scared_pair_dataset.make_scared_rgb_dataset", fake_make)
    result = make_scared_pair_rgb_dataset(
        {"pair_mode": True, "pair_stride": 2, "pair_step": 1}, "train"
    )
    assert captured["clip_length"] == 2
    assert captured["sample_stride"] == 2
    assert captured["window_stride"] == 1
    assert result.pair_stride == 2


def test_pair_stride_2_and_fixed_order() -> None:
    dataset = _FakePairDataset(5)
    for index in range(3):
        metadata = pair_metadata(dataset, index)
        assert metadata["frame_index_b"] - metadata["frame_index_a"] == 2
        assert metadata["frame_id_b"] - metadata["frame_id_a"] == 2


def test_frame_cache_metadata_and_float32(tmp_path) -> None:
    metadata = frame_metadata_from_pair(pair_metadata(_FakePairDataset(), 0))[0]
    path = _write_frame(tmp_path, metadata)
    with np.load(path, allow_pickle=False) as cache:
        validate_teacher_frame_cache(cache, metadata, (4, 6), BASE_CHECKPOINT)
        assert cache["depth"].dtype == np.float32
        assert cache["xyz_local"].dtype == np.float32


def test_frame_cache_path_separates_manifest_sequences_and_indices(tmp_path) -> None:
    metadata = frame_metadata_from_pair(pair_metadata(_FakePairDataset(), 0))[0]
    other_sequence = {**metadata, "sequence_id": "custom/sequence"}
    other_index = {**metadata, "frame_index": metadata["frame_index"] + 9}
    paths = {
        teacher_frame_cache_path(tmp_path, metadata),
        teacher_frame_cache_path(tmp_path, other_sequence),
        teacher_frame_cache_path(tmp_path, other_index),
    }
    assert len(paths) == 3


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("depth", np.ones((4, 6), dtype=np.float16), "must be float32"),
        ("valid_mask", np.ones((4, 6), dtype=np.uint8), "must be boolean"),
        ("intrinsics", np.eye(4, dtype=np.float32), "invalid shape"),
    ],
)
def test_frame_cache_validator_rejects_invalid_storage(
    tmp_path, key, value, message
) -> None:
    metadata = frame_metadata_from_pair(pair_metadata(_FakePairDataset(), 0))[0]
    arrays = _frame_arrays(metadata)
    arrays[key] = value
    path = teacher_frame_cache_path(tmp_path, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as cache, pytest.raises(
        RuntimeError, match=message
    ):
        validate_teacher_frame_cache(cache, metadata, (4, 6), BASE_CHECKPOINT)


def test_pair_compatibility_generator_rejects_ambiguous_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not accept --limit"):
        generate_teacher_pair_cache(tmp_path / "config.yaml", "train", limit=1)


def test_stale_pair_or_clip_cache_rejected(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, cache_format_version=np.asarray("vggtomast3r-pair-v1"))
    with np.load(path, allow_pickle=False) as cache, pytest.raises(RuntimeError, match="missing keys"):
        validate_teacher_frame_cache(cache)


def test_compose_two_frame_caches(tmp_path) -> None:
    frames = frame_metadata_from_pair(pair_metadata(_FakePairDataset(), 0))
    for index, metadata in enumerate(frames):
        _write_frame(tmp_path, metadata, value=float(index + 1))
    result = compose_teacher_frame_caches(
        tmp_path, frames, (4, 6), BASE_CHECKPOINT
    )
    assert result["depth"].shape == (2, 4, 6)
    assert result["xyz_local"].shape == (2, 4, 6, 3)
    assert result["depth"][:, 0, 0].tolist() == [1.0, 2.0]
    assert "independently" in result["coordinate_convention"]


def test_compose_eight_frame_caches(tmp_path) -> None:
    common = {
        "dataset_id": 1,
        "keyframe_id": "keyframe_1",
        "sequence_id": "seq",
        "sequence_length": 8,
    }
    clip = {
        **common,
        "frame_indices": list(range(8)),
        "frame_names": ["frame_{:06d}.png".format(index) for index in range(8)],
        "frame_paths": ["frame_{:06d}.png".format(index) for index in range(8)],
    }
    frames = frame_metadata_from_clip(clip)
    for index in range(8):
        _write_frame(tmp_path, frames[index], value=float(index + 1))
    result = compose_teacher_frame_caches(
        tmp_path, frames, (4, 6), BASE_CHECKPOINT
    )
    assert result["depth"].shape == (8, 4, 6)
    assert result["frame_indices"] == list(range(8))


def test_pair_training_dataset_uses_two_local_frame_targets(tmp_path) -> None:
    rgb = _FakePairDataset()
    frames = frame_metadata_from_pair(pair_metadata(rgb, 0))
    for index, metadata in enumerate(frames):
        _write_frame(tmp_path, metadata, value=float(index + 1))
    dataset = ScaredPairDistillDataset(
        rgb, tmp_path, expected_base_checkpoint=BASE_CHECKPOINT
    )
    sample = dataset[0]
    assert set(sample["target"]) == {
        "pts3d_ref",
        "pts3d_other_local",
        "confidence_ref",
        "confidence_other",
        "valid_mask_ref",
        "valid_mask_other",
    }
    assert sample["target"]["pts3d_ref"][0, 0, 2] == 1.0
    assert sample["target"]["pts3d_other_local"][0, 0, 2] == 2.0
    assert len(sample["cache_paths"]) == 2
