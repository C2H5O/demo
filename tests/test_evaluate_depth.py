from __future__ import annotations

import sys
import numpy as np
import pytest


class _FakeCV2:
    INTER_NEAREST = 0

    @staticmethod
    def resize(array, size, interpolation=None):
        del size, interpolation
        return array


def test_endo3r_uses_one_scale_for_the_complete_scene(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", _FakeCV2())
    from evaluation.evaluate_depth import depth_evaluation

    gt_depths = [
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([[4.0, 8.0]], dtype=np.float32),
    ]
    pred_depths = [
        np.asarray([[2.0, 4.0]], dtype=np.float32),
        np.asarray([[8.0, 16.0]], dtype=np.float32),
    ]

    errors, mean_errors, ratio = depth_evaluation(gt_depths, pred_depths)

    assert ratio == pytest.approx(0.5)
    assert errors.shape == (2, 7)
    np.testing.assert_allclose(mean_errors[:4], 0.0, atol=1e-12)
    np.testing.assert_allclose(mean_errors[4:], 1.0, atol=1e-12)


def test_endo3r_scene_scale_is_not_per_frame(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", _FakeCV2())
    from evaluation.evaluate_depth import depth_evaluation

    gt_depths = [
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([[1.0, 2.0]], dtype=np.float32),
    ]
    pred_depths = [
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([[10.0, 20.0]], dtype=np.float32),
    ]

    _, mean_errors, ratio = depth_evaluation(gt_depths, pred_depths)

    assert ratio != pytest.approx(1.0)
    assert ratio != pytest.approx(0.1)
    assert mean_errors[0] > 0.0
