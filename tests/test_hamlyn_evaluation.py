from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from evaluation.hamlyn.common import (
    fit_sequence_disparity_scale_shift,
    macro_mean,
    resize_predicted_depth,
)
from evaluation.hamlyn.constants import (
    EVALUATION_RESOLUTION_HW,
    HAMLYN_GT_SCALE,
    HAMLYN_MAX_DEPTH,
    HAMLYN_MIN_DEPTH,
    HAMLYN_SEQUENCE_IDS,
    INFERENCE_RESOLUTIONS_HW,
    METHOD_ORDER,
)
from evaluation.hamlyn.data import (
    DiscoveryError,
    discover_sequences,
    frame_id,
    index_by_frame_id,
)
from evaluation.hamlyn.endo3r import _validate_raft_checkpoint
from evaluation.hamlyn.gt import load_hamlyn_gt_depth
from utils.config import load_config


EXPECTED_SEQUENCES = (
    1,
    4,
    5,
    6,
    8,
    9,
    11,
    12,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
)


def test_fixed_sequence_list_contains_exactly_22_sequences_including_9() -> None:
    assert HAMLYN_SEQUENCE_IDS == EXPECTED_SEQUENCES
    assert len(HAMLYN_SEQUENCE_IDS) == 22
    assert 9 in HAMLYN_SEQUENCE_IDS


def test_discovery_finds_the_exact_prepared_22_sequence_layout(tmp_path: Path) -> None:
    for sequence_id in HAMLYN_SEQUENCE_IDS:
        directory = tmp_path / "prepared" / "rectified{:02d}".format(sequence_id)
        color = directory / "color"
        depth = directory / "depth"
        color.mkdir(parents=True)
        depth.mkdir(parents=True)
        assert cv2.imwrite(
            str(color / "frame_10.png"), np.zeros((2, 3, 3), dtype=np.uint8)
        )
        assert cv2.imwrite(
            str(depth / "depth_10.png"), np.full((2, 3), 100, dtype=np.uint16)
        )
    records = discover_sequences(tmp_path)
    assert tuple(record.sequence_id for record in records) == HAMLYN_SEQUENCE_IDS
    assert all(record.frame_ids == (10,) for record in records)


def test_discovery_prefers_native_camera01_layout_when_both_views_exist(
    tmp_path: Path,
) -> None:
    for sequence_id in HAMLYN_SEQUENCE_IDS:
        directory = tmp_path / "rectified{:02d}".format(sequence_id)
        for camera in ("01", "02"):
            image = directory / "image{}".format(camera)
            depth = directory / "depth{}".format(camera)
            image.mkdir(parents=True)
            depth.mkdir(parents=True)
            assert cv2.imwrite(
                str(image / "0000000000.jpg"), np.zeros((2, 3, 3), dtype=np.uint8)
            )
            assert cv2.imwrite(
                str(depth / "0000000000.png"), np.full((2, 3), 100, dtype=np.uint16)
            )
    records = discover_sequences(tmp_path)
    assert all(record.rgb_directory.name == "image01" for record in records)
    assert all(record.depth_directory.name == "depth01" for record in records)
    assert all(record.frame_ids == (0,) for record in records)


def test_numeric_frame_id_matching_is_not_lexicographic(tmp_path: Path) -> None:
    paths = [
        tmp_path / "frame_10.png",
        tmp_path / "frame_2.png",
        tmp_path / "000001.png",
    ]
    indexed = index_by_frame_id(paths, "RGB")
    assert sorted(indexed) == [1, 2, 10]
    assert frame_id(indexed[10]) == 10


def test_duplicate_numeric_frame_ids_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(DiscoveryError, match="Duplicate RGB frame ID 1"):
        index_by_frame_id(
            [tmp_path / "left_0001.png", tmp_path / "color_1.png"], "RGB"
        )


def test_uint16_mm_gt_conversion_and_nearest_resize(tmp_path: Path) -> None:
    source = np.array([[2, 100], [200, 299]], dtype=np.uint16)
    path = tmp_path / "depth_1.png"
    assert cv2.imwrite(str(path), source)
    depth = load_hamlyn_gt_depth(path)
    assert depth.shape == EVALUATION_RESOLUTION_HW
    assert depth.dtype == np.float32
    assert sorted(np.unique(depth).tolist()) == pytest.approx(
        [0.002, 0.100, 0.200, 0.299]
    )
    assert HAMLYN_GT_SCALE == 0.001


def test_non_uint16_gt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "depth_1.png"
    assert cv2.imwrite(str(path), np.full((2, 2), 10, dtype=np.uint8))
    with pytest.raises(ValueError, match="uint16"):
        load_hamlyn_gt_depth(path)


def test_endo3r_depth_is_bilinearly_resized_from_256x320_to_224x280() -> None:
    rows = np.linspace(0.01, 0.2, 256, dtype=np.float32)[:, None]
    depth = np.broadcast_to(rows, (256, 320)).copy()
    resized = resize_predicted_depth(depth)
    assert resized.shape == (224, 280)
    assert resized.dtype == np.float32
    assert resized[0, 0] < resized[-1, 0]


def test_sequence_level_disparity_scale_shift_recovery() -> None:
    predicted = [
        np.linspace(1.0, 3.0, 12, dtype=np.float32).reshape(3, 4),
        np.linspace(2.0, 4.0, 12, dtype=np.float32).reshape(3, 4),
    ]
    scale, shift = 2.0, 3.0
    ground_truth = [1.0 / (scale * value + shift) for value in predicted]
    recovered_scale, recovered_shift, count = fit_sequence_disparity_scale_shift(
        predicted, ground_truth
    )
    assert recovered_scale == pytest.approx(scale, rel=1e-6)
    assert recovered_shift == pytest.approx(shift, rel=1e-6)
    assert count == 24


def test_macro_mean_is_not_frame_count_weighted() -> None:
    names = ("abs_relative_difference", "rmse_linear", "delta1_acc")
    results = [
        {"frame_count": 1, "metrics": {name: 0.0 for name in names}},
        {"frame_count": 1000, "metrics": {name: 1.0 for name in names}},
    ]
    assert macro_mean(results) == {
        "abs_relative_difference": 0.5,
        "rmse_linear": 0.5,
        "delta1_acc": 0.5,
    }


def test_method_and_common_resolutions_are_locked() -> None:
    assert INFERENCE_RESOLUTIONS_HW["ours"] == (224, 280)
    assert INFERENCE_RESOLUTIONS_HW["da3"] == (224, 280)
    assert INFERENCE_RESOLUTIONS_HW["endodav"] == (224, 280)
    assert INFERENCE_RESOLUTIONS_HW["endo3r"] == (256, 320)
    assert EVALUATION_RESOLUTION_HW == (224, 280)
    assert METHOD_ORDER == ("ours", "da3", "endodav", "endo3r")


def test_hamlyn_depth_range_is_independent_from_scared() -> None:
    assert HAMLYN_MIN_DEPTH == 0.001
    assert HAMLYN_MAX_DEPTH == 0.300


def test_tae_is_disabled_in_formal_config() -> None:
    config = load_config(Path("configs/hamlyn_eval.yaml"))
    assert config["evaluation"]["tae"] == {"enabled": False}
    assert config["evaluation"]["resolution_hw"] == [224, 280]


def test_endo3r_uses_endo3r_and_raft_checkpoints() -> None:
    config = load_config(Path("configs/hamlyn_eval.yaml"))
    paths = config["paths"]
    assert paths["endo3r_checkpoint"].endswith("/checkpoints/endo3r.pth")
    assert paths["endo3r_raft_checkpoint"].endswith(
        "/checkpoints/raft-things.pth"
    )
    assert "endo3r_dust3r_checkpoint" not in paths


def test_endo3r_raft_checkpoint_must_match_official_relative_path(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "Endo3R"
    checkpoint = repository / "checkpoints" / "raft-things.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    runtime = SimpleNamespace(
        endo3r_repository=repository.resolve(),
        endo3r_raft_checkpoint=checkpoint.resolve(),
    )
    _validate_raft_checkpoint(runtime)
    runtime.endo3r_raft_checkpoint = (tmp_path / "elsewhere" / "raft-things.pth").resolve()
    with pytest.raises(RuntimeError, match="loads RAFT"):
        _validate_raft_checkpoint(runtime)
