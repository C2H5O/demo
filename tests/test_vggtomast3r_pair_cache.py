from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from datasets.scared_dataset import ClipRecord
from datasets.scared_pair_dataset import (
    PAIR_CACHE_FORMAT_VERSION,
    PAIR_COORDINATE_CONVENTION,
    REQUIRED_PAIR_CACHE_KEYS,
    ScaredPairDistillDataset,
    make_scared_pair_rgb_dataset,
    pair_metadata,
    teacher_pair_cache_path,
    validate_pair_cache,
)


def _fake_rgb_dataset() -> SimpleNamespace:
    sequence = {
        "dataset_id": 1, "keyframe_id": "keyframe_1", "sequence_id": "seq",
        "sequence_length": 5,
        "frame_paths": ["frame_{:06d}.png".format(i) for i in range(5)],
    }
    clips = [ClipRecord(sequence, (i, i + 2), i) for i in range(3)]
    return SimpleNamespace(
        clips=clips, sample_stride=2, window_stride=1, pair_stride=2, pair_step=1,
        __len__=lambda self: len(clips),
    )


def _cache_arrays(shape=(448, 560), version=PAIR_CACHE_FORMAT_VERSION):
    h, w = shape
    point = np.zeros((h, w, 3), dtype=np.float16)
    scalar = np.ones((h, w), dtype=np.float16)
    arrays = {
        "frame_id_a": np.asarray(0), "frame_id_b": np.asarray(2),
        "frame_name_a": np.asarray("frame_000000.png"), "frame_name_b": np.asarray("frame_000002.png"),
        "pair_stride": np.asarray(2), "image_shape": np.asarray(shape),
        "teacher_variant": np.asarray("lora"), "depth_a": scalar, "depth_b": scalar,
        "xyz_local_a": point, "xyz_local_b": point, "xyz_global_a": point, "xyz_global_b": point,
        "pts3d_a_in_a": point, "pts3d_b_in_a": point,
        "confidence_a": scalar, "confidence_b": scalar,
        "valid_mask_a": scalar.astype(bool), "valid_mask_b": scalar.astype(bool),
        "intrinsics_a": np.eye(3), "intrinsics_b": np.eye(3),
        "extrinsics_a": np.eye(4)[:3], "extrinsics_b": np.eye(4)[:3],
        "coordinate_convention": np.asarray(PAIR_COORDINATE_CONVENTION),
        "cache_format_version": np.asarray(version),
        "lora_checkpoint": np.asarray("./checkpoints/teacher_lora/last.pt"),
        "metadata_json": np.asarray("{}"),
    }
    assert set(REQUIRED_PAIR_CACHE_KEYS).issubset(arrays)
    return arrays


def test_pair_dataset(monkeypatch) -> None:
    captured = {}

    def fake_make(config, split):
        captured.update(config)
        captured["split"] = split
        return SimpleNamespace(sample_stride=config["sample_stride"], window_stride=config["window_stride"])

    monkeypatch.setattr("datasets.scared_pair_dataset.make_scared_rgb_dataset", fake_make)
    result = make_scared_pair_rgb_dataset({"pair_mode": True, "pair_stride": 2, "pair_step": 1}, "train")
    assert captured["clip_length"] == 2
    assert captured["sample_stride"] == 2
    assert captured["window_stride"] == 1
    assert result.pair_stride == 2


def test_pair_stride_2_and_fixed_order() -> None:
    dataset = _fake_rgb_dataset()
    for index in range(3):
        metadata = pair_metadata(dataset, index)
        assert metadata["frame_index_b"] - metadata["frame_index_a"] == 2
        assert metadata["frame_id_b"] - metadata["frame_id_a"] == 2


def test_teacher_pair_cache_metadata(tmp_path) -> None:
    path = tmp_path / "cache.npz"
    np.savez_compressed(path, **_cache_arrays())
    metadata = pair_metadata(_fake_rgb_dataset(), 0)
    with np.load(path, allow_pickle=False) as cache:
        validate_pair_cache(cache, metadata, (448, 560))


def test_stale_8frame_cache_rejected(tmp_path) -> None:
    path = tmp_path / "stale.npz"
    np.savez_compressed(path, **_cache_arrays(version="legacy-8frame"))
    with np.load(path, allow_pickle=False) as cache, pytest.raises(RuntimeError, match="Eight-frame"):
        validate_pair_cache(cache)


def test_wrong_teacher_variant_rejected(tmp_path) -> None:
    arrays = _cache_arrays()
    arrays["teacher_variant"] = np.asarray("base")
    path = tmp_path / "base.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as cache, pytest.raises(RuntimeError, match="teacher variant"):
        validate_pair_cache(
            cache,
            expected_teacher_variant="lora",
            expected_lora_checkpoint="./checkpoints/teacher_lora/last.pt",
        )


def test_pair_cache_path_is_separate() -> None:
    metadata = pair_metadata(_fake_rgb_dataset(), 0)
    path = teacher_pair_cache_path("teacher_cache_vggtomast3r_pair2", metadata)
    assert "pair_000000_000002_stride_02.npz" in str(path)
    assert "teacher_cache_endodac" not in str(path)
