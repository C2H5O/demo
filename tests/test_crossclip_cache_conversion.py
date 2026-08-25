from __future__ import annotations

import importlib
import json
import zipfile

import numpy as np
import pytest

converter = importlib.import_module("cache.convert_crossclip_teacher_cache")


def _complete_arrays():
    shape = (2, 3, 4)
    depth = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    points = np.stack((depth, depth + 1.0, depth + 2.0), axis=-1)
    return {
        "cache_format_version": np.asarray("crossclip-v1"),
        "cache_stage": np.asarray("raw"),
        "sequence_id": np.asarray("dataset1/keyframe_1"),
        "clip_start": np.asarray(0, dtype=np.int64),
        "absolute_frame_ids": np.asarray([0, 1], dtype=np.int64),
        "depth": depth,
        "xyz_local": points,
        "xyz_global": points.copy(),
        "confidence": np.ones(shape, dtype=np.float32),
        "valid_mask": np.ones(shape, dtype=np.bool_),
        "highlight_mask": np.zeros(shape, dtype=np.bool_),
        "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        "extrinsics": np.repeat(np.eye(4, dtype=np.float32)[None, :3], 2, axis=0),
        "metadata_json": np.asarray(json.dumps({"clip_start": 0})),
        "extra_complete_field": np.asarray([np.nan, 3.0], dtype=np.float32),
    }


def _write_compressed(path, arrays=None):
    values = _complete_arrays() if arrays is None else arrays
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return values


def _compression_methods(path):
    with zipfile.ZipFile(path, "r") as archive:
        return [item.compress_type for item in archive.infolist() if not item.is_dir()]


def test_complete_cache_is_verified_and_atomically_replaced_in_place(tmp_path) -> None:
    path = tmp_path / "train" / "dataset1" / "clip_000000.npz"
    expected = _write_compressed(path)
    report = converter.convert_crossclip_teacher_cache_root(
        tmp_path, confirm_no_readers=True
    )
    assert report.converted == 1
    assert report.already_uncompressed == 0
    assert path.is_file()
    assert all(
        method == zipfile.ZIP_STORED for method in _compression_methods(path)
    )
    with np.load(path, allow_pickle=False) as actual:
        assert list(actual.files) == list(expected)
        for key, value in expected.items():
            if value.dtype.kind in "fc":
                assert np.array_equal(actual[key], value, equal_nan=True)
            else:
                assert np.array_equal(actual[key], value)
            assert actual[key].dtype == value.dtype
            assert actual[key].shape == value.shape
    assert not list(tmp_path.rglob("*.tmp.npz"))
    assert not (tmp_path / converter.LOCK_FILENAME).exists()


def test_resume_skips_an_already_uncompressed_cache(tmp_path) -> None:
    path = tmp_path / "clip_000000.npz"
    np.savez(path, **_complete_arrays())
    original = path.read_bytes()
    report = converter.convert_crossclip_teacher_cache_root(
        tmp_path, confirm_no_readers=True
    )
    assert report.converted == 0
    assert report.already_uncompressed == 1
    assert path.read_bytes() == original


def test_write_requires_explicit_no_reader_confirmation(tmp_path) -> None:
    _write_compressed(tmp_path / "clip_000000.npz")
    with pytest.raises(ValueError, match="confirm-no-readers"):
        converter.convert_crossclip_teacher_cache_root(tmp_path)
    assert not converter.is_uncompressed_npz(tmp_path / "clip_000000.npz")


def test_dry_run_does_not_write_or_lock(tmp_path) -> None:
    path = tmp_path / "clip_000000.npz"
    _write_compressed(path)
    original = path.read_bytes()
    report = converter.convert_crossclip_teacher_cache_root(tmp_path, dry_run=True)
    assert report.dry_run
    assert report.converted == 0
    assert path.read_bytes() == original
    assert not (tmp_path / converter.LOCK_FILENAME).exists()


def test_failed_verification_preserves_source_and_removes_owned_temp(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "clip_000000.npz"
    _write_compressed(path)
    original = path.read_bytes()

    def fail_verification(*args, **kwargs):
        raise RuntimeError("injected verification failure")

    monkeypatch.setattr(converter, "_verify_uncompressed_copy", fail_verification)
    with pytest.raises(RuntimeError, match="injected verification failure"):
        converter.convert_crossclip_teacher_cache_root(
            tmp_path, confirm_no_readers=True
        )
    assert path.read_bytes() == original
    assert not converter.is_uncompressed_npz(path)
    assert not list(tmp_path.rglob("*.tmp.npz"))
    assert not (tmp_path / converter.LOCK_FILENAME).exists()


def test_incomplete_npz_is_rejected_before_any_conversion(tmp_path) -> None:
    first = tmp_path / "a_complete.npz"
    second = tmp_path / "z_incomplete.npz"
    _write_compressed(first)
    np.savez_compressed(second, depth=np.ones((1,), dtype=np.float32))
    with pytest.raises(ValueError, match="incomplete cache"):
        converter.convert_crossclip_teacher_cache_root(
            tmp_path, confirm_no_readers=True
        )
    assert not converter.is_uncompressed_npz(first)
    assert not (tmp_path / converter.LOCK_FILENAME).exists()


def test_existing_lock_refuses_second_converter(tmp_path) -> None:
    path = tmp_path / "clip_000000.npz"
    _write_compressed(path)
    lock = tmp_path / converter.LOCK_FILENAME
    lock.write_text('{"pid": 123, "hostname": "worker"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="lock already exists"):
        converter.convert_crossclip_teacher_cache_root(
            tmp_path, confirm_no_readers=True
        )
    assert not converter.is_uncompressed_npz(path)
