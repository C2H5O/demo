from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import evaluation.hamlyn.data as hamlyn_data
from evaluation.hamlyn.config import (
    DEFAULT_ENDO3R_PYTHON,
    DEFAULT_ENDODAV_PYTHON,
    DEFAULT_OURS_PYTHON,
    _environment_integer,
)
from evaluation.hamlyn.data import SequenceRecord, inspect_sequence
from evaluation.hamlyn.endodav import load_resized_rgb
from inference.student_video import (
    KEYFRAMES,
    OVERLAP,
    SequenceFrames,
    _load_frame_worker,
    _window_frame_ids,
    infer_student_video,
)


def _write_rgb(path: Path, color: tuple[int, int, int], size=(11, 7)) -> None:
    Image.new("RGB", size, color=color).save(path)


def _rgb_paths(tmp_path: Path) -> list[Path]:
    paths = []
    for index, color in enumerate(((10, 20, 30), (40, 50, 60), (70, 80, 90))):
        path = tmp_path / "{:010d}.png".format(index)
        _write_rgb(path, color)
        paths.append(path)
    return paths


def test_parallel_loader_preserves_order_and_matches_one_worker(tmp_path: Path) -> None:
    paths = _rgb_paths(tmp_path)
    with SequenceFrames(
        paths, height=8, width=12, num_workers=1, frame_cache_size=8
    ) as serial:
        expected = serial.load_indices([2, 0, 1])
    with SequenceFrames(
        paths, height=8, width=12, num_workers=4, frame_cache_size=8
    ) as parallel:
        actual = parallel.load_indices([2, 0, 1])
        assert parallel.multiprocessing_context == "spawn"
        assert parallel.loader_backend == "ProcessPoolExecutor(spawn)"
    assert len(actual) == len(expected) == 3
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_parallel_resize_matches_legacy_getitem_exactly(tmp_path: Path) -> None:
    path = _rgb_paths(tmp_path)[0]
    serial = SequenceFrames([path], height=9, width=13, num_workers=1)
    expected = serial[0]
    serial.close()
    with SequenceFrames([path], height=9, width=13, num_workers=4) as parallel:
        actual = parallel.load_indices([0])[0]
    assert actual.shape == expected.shape == (3, 9, 13)
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_lru_cache_hit_preserves_tensor_without_redecode(tmp_path: Path) -> None:
    path = _rgb_paths(tmp_path)[0]
    with SequenceFrames(
        [path], height=8, width=12, num_workers=1, frame_cache_size=1
    ) as frames:
        first = frames.load_indices([0])[0].clone()
        path.unlink()
        cached = frames.load_indices([0])[0]
    assert torch.equal(cached, first)


def test_window_prefetch_keeps_ids_and_is_scheduled_before_forward() -> None:
    events = []
    source = []
    for index in range(55):
        value = torch.zeros(3, 2, 3)
        value[0].fill_(index)
        source.append(value)

    class Batch:
        def __init__(self, indices):
            self.indices = tuple(indices)

        def result(self):
            events.append(("result", self.indices))
            return [source[index] for index in self.indices]

    class Frames:
        num_workers = 4
        loader_backend = "test-spawn"
        frame_cache_size = 96

        def __len__(self):
            return len(source)

        def prefetch_indices(self, indices):
            values = tuple(indices)
            events.append(("prefetch", values))
            return Batch(values)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ids = []

        def forward(self, images, include_global_points=False):
            ids = tuple(int(value) for value in images[0, :, 0, 0, 0].tolist())
            self.ids.append(ids)
            events.append(("forward", ids))
            count = images.shape[1]
            return {
                "depth": torch.ones(1, count, 2, 3),
                "intrinsics": torch.eye(3).repeat(1, count, 1, 1),
            }

    model = Model().eval()
    stats = infer_student_video(model, Frames(), lambda *_: None, device="cpu")
    expected_windows = [tuple(values) for values in _window_frame_ids(55)]
    assert model.ids == expected_windows
    assert events[:4] == [
        ("prefetch", tuple(dict.fromkeys(expected_windows[0]))),
        ("result", tuple(dict.fromkeys(expected_windows[0]))),
        ("prefetch", tuple(dict.fromkeys(expected_windows[1]))),
        ("forward", expected_windows[0]),
    ]
    assert stats["prefetch_windows"] == 1
    assert stats["frame_cache_size"] == 96


def test_overlap_reference_ids_are_unchanged() -> None:
    windows = _window_frame_ids(80)
    assert len(windows) > 1
    assert windows[1][:OVERLAP] == [windows[0][index] for index in KEYFRAMES]


def test_executor_shutdowns_after_decode_exception(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_text("not an image", encoding="utf-8")
    frames = SequenceFrames([bad], num_workers=2, frame_cache_size=1)
    with pytest.raises(RuntimeError, match="Failed to decode"):
        with frames:
            frames.load_indices([0])
    assert frames.closed is True
    assert frames._executor is None


def test_executor_shutdowns_after_inference(tmp_path: Path) -> None:
    paths = _rgb_paths(tmp_path)

    class Model(nn.Module):
        def forward(self, images, include_global_points=False):
            count, height, width = images.shape[1], images.shape[-2], images.shape[-1]
            return {
                "depth": torch.ones(1, count, height, width),
                "intrinsics": torch.eye(3).repeat(1, count, 1, 1),
            }

    frames = SequenceFrames(
        paths, height=8, width=12, num_workers=2, frame_cache_size=8
    )
    with frames:
        stats = infer_student_video(Model().eval(), frames, lambda *_: None, device="cpu")
    assert stats["output_frame_count"] == len(paths)
    assert frames.closed is True
    assert frames._executor is None


def test_worker_has_no_cuda_or_model_references() -> None:
    referenced_names = {name.casefold() for name in _load_frame_worker.__code__.co_names}
    assert "cuda" not in referenced_names
    assert "model" not in referenced_names
    assert "to" not in referenced_names


def test_endodav_parallel_resize_preserves_order_and_values(tmp_path: Path) -> None:
    paths = _rgb_paths(tmp_path)
    expected = load_resized_rgb(paths, num_workers=1)
    actual = load_resized_rgb(paths, num_workers=4)
    assert actual.shape == expected.shape == (3, 224, 280, 3)
    assert actual.dtype == expected.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


def test_preflight_decodes_only_first_rgb_and_gt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rgb_by_id = {}
    depth_by_id = {}
    for identifier in range(3):
        rgb_path = tmp_path / "{:010d}.jpg".format(identifier)
        depth_path = tmp_path / "depth_{:010d}.png".format(identifier)
        _write_rgb(rgb_path, (identifier, identifier, identifier))
        assert cv2.imwrite(
            str(depth_path), np.full((5, 7), 100 + identifier, dtype=np.uint16)
        )
        rgb_by_id[identifier] = rgb_path
        depth_by_id[identifier] = depth_path
    record = SequenceRecord(
        sequence_id=1,
        rgb_directory=tmp_path,
        depth_directory=tmp_path / "depth01",
        rgb_by_id=rgb_by_id,
        depth_by_id=depth_by_id,
    )
    calls = {"rgb": 0, "gt": 0}
    original_open = hamlyn_data.Image.open
    original_read_gt = hamlyn_data.read_hamlyn_gt_uint16

    def counted_open(*args, **kwargs):
        calls["rgb"] += 1
        return original_open(*args, **kwargs)

    def counted_read_gt(*args, **kwargs):
        calls["gt"] += 1
        return original_read_gt(*args, **kwargs)

    monkeypatch.setattr(hamlyn_data.Image, "open", counted_open)
    monkeypatch.setattr(hamlyn_data, "read_hamlyn_gt_uint16", counted_read_gt)
    result = inspect_sequence(record)
    assert calls == {"rgb": 1, "gt": 1}
    assert result["rgb_frame_count"] == result["gt_frame_count"] == 3
    assert result["matched_frame_count"] == 3
    assert result["sample_rgb_hw"] == [7, 11]
    assert result["sample_gt_hw"] == [5, 7]
    assert result["sample_gt_dtype"] == "uint16"
    assert "gt_raw_min" not in result
    assert "valid_pixel_count" not in result

def test_resize_worker_environment_defaults_and_overrides() -> None:
    assert _environment_integer({}, "HAMLYN_RESIZE_WORKERS", 4, 1) == 4
    assert _environment_integer(
        {"HAMLYN_RESIZE_WORKERS": "8"}, "HAMLYN_RESIZE_WORKERS", 4, 1
    ) == 8
    assert _environment_integer(
        {"HAMLYN_FRAME_CACHE_SIZE": "0"}, "HAMLYN_FRAME_CACHE_SIZE", 96, 0
    ) == 0
    with pytest.raises(ValueError, match=">= 1"):
        _environment_integer(
            {"HAMLYN_RESIZE_WORKERS": "0"}, "HAMLYN_RESIZE_WORKERS", 4, 1
        )


def test_eval_script_has_independent_server_python_defaults() -> None:
    assert len({DEFAULT_OURS_PYTHON, DEFAULT_ENDODAV_PYTHON, DEFAULT_ENDO3R_PYTHON}) == 3
    assert DEFAULT_OURS_PYTHON.endswith("/envs/vggtomast3r/bin/python")
    assert DEFAULT_ENDODAV_PYTHON.endswith("/envs/endodav/bin/python")
    assert DEFAULT_ENDO3R_PYTHON.endswith("/envs/endo3r/bin/python")
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "eval_all.bash"
    ).read_text(encoding="utf-8")
    assert "envs/vggtomast3r/bin/python" in script
    assert "envs/endodav/bin/python" in script
    assert "envs/endo3r/bin/python" in script
    assert 'ENDODAV_PYTHON="${ENDODAV_PYTHON:-${OURS_PYTHON}}"' not in script
    assert 'HAMLYN_RESIZE_WORKERS="${HAMLYN_RESIZE_WORKERS:-4}"' in script
