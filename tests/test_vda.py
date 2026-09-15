from pathlib import Path

import cv2
import numpy as np
import pytest

from endodaveval.vda import (
    evaluate_sequence_arrays,
    load_ground_truth,
    prediction_depth_to_evaluation_disparity,
)


def test_prediction_is_reciprocal_before_bilinear_resize() -> None:
    depth = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    actual = prediction_depth_to_evaluation_disparity(depth)
    expected = cv2.resize(1.0 / depth, (320, 256), interpolation=cv2.INTER_LINEAR)
    wrong = 1.0 / cv2.resize(depth, (320, 256), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, wrong)


def test_ground_truth_is_mm_to_m_then_nearest(tmp_path: Path) -> None:
    path = tmp_path / "depth.npy"
    np.save(path, np.array([[1000.0, 2000.0], [3000.0, 4000.0]], np.float32))
    actual = load_ground_truth(path, 0.001, 0)
    expected = cv2.resize(
        np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
        (320, 256),
        interpolation=cv2.INTER_NEAREST,
    )
    np.testing.assert_allclose(actual, expected)


def test_sequence_global_float64_scale_shift_and_spatial_metrics() -> None:
    y, x = np.mgrid[:256, :320]
    gt0 = (1.0 + x / 640.0 + y / 512.0).astype(np.float32)
    gt1 = (1.2 + x / 700.0 + y / 600.0).astype(np.float32)
    gt = {0: gt0, 1: gt1}
    gt_disp = {identifier: 1.0 / value for identifier, value in gt.items()}
    prediction = {identifier: (value - 0.07) / 1.8 for identifier, value in gt_disp.items()}
    result, aligned = evaluate_sequence_arrays(prediction, gt, [0, 1])
    assert result["alignment"]["scale"] == pytest.approx(1.8, rel=1e-5)
    assert result["alignment"]["shift"] == pytest.approx(0.07, rel=1e-5)
    assert result["metrics"]["abs_relative_difference"] == pytest.approx(0.0, abs=1e-5)
    assert result["metrics"]["rmse_linear"] == pytest.approx(0.0, abs=1e-5)
    assert result["metrics"]["delta1_acc"] == pytest.approx(1.0)
    assert set(aligned) == {0, 1}
    assert result["evaluation_shape_hxw"] == [256, 320]

\n