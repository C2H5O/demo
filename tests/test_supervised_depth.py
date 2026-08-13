from __future__ import annotations

import torch
import numpy as np

from datasets.ground_truth import load_clip_ground_truth
from datasets.scared_clip_dataset import _resize_map
from losses.distillation_loss import ScaredDistillationLoss
from losses.supervised_depth_loss import SupervisedDepthLoss


def test_median_aligned_supervised_depth_is_scale_invariant() -> None:
    prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    ground_truth = prediction.detach() * 3.0
    loss, logs = SupervisedDepthLoss()(prediction, ground_truth)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
    loss.backward()
    assert prediction.grad is not None
    assert logs["supervised_depth_scale"].item() == 3.0


def test_empty_supervised_mask_returns_differentiable_zero() -> None:
    prediction = torch.ones((1, 2, 3, 4), requires_grad=True)
    ground_truth = torch.zeros_like(prediction)

    loss, logs = SupervisedDepthLoss()(prediction, ground_truth)

    assert loss.item() == 0.0
    assert logs["supervised_depth_valid_fraction"].item() == 0.0
    loss.backward()
    torch.testing.assert_close(
        prediction.grad, torch.zeros_like(prediction)
    )


def test_distillation_uses_separate_supervised_depth_bounds() -> None:
    loss_function = ScaredDistillationLoss(
        {
            "min_depth": 0.1,
            "max_depth": 150.0,
            "supervised_depth_min_depth": 1e-4,
            "supervised_depth_max_depth": 100.0,
        }
    )

    assert loss_function.supervised_depth.config.min_depth == 1e-4
    assert loss_function.supervised_depth.config.max_depth == 100.0


def test_teacher_targets_resize_to_native_student_output() -> None:
    point_map = torch.rand(2, 448, 560, 3)
    confidence = torch.rand(2, 448, 560)
    valid = torch.ones(2, 448, 560)

    assert _resize_map(point_map, 256, 320, "bilinear").shape == (2, 256, 320, 3)
    assert _resize_map(confidence, 256, 320, "bilinear").shape == (2, 256, 320)
    assert _resize_map(valid, 256, 320, "nearest").shape == (2, 256, 320)


def test_student_distillation_combines_cache_and_ground_truth() -> None:
    torch.manual_seed(0)
    batch, frames, height, width = 1, 2, 6, 8
    target_local = torch.randn(batch, frames, height, width, 3)
    target_local[..., 2] = target_local[..., 2].abs() + 1.0
    target_global = target_local + 0.1
    prediction_local = (target_local + 0.01 * torch.randn_like(target_local)).requires_grad_()
    prediction_global = (target_global + 0.01 * torch.randn_like(target_global)).requires_grad_()
    confidence = torch.full((batch, frames, height, width), 0.8)
    prediction = {
        "xyz_local": prediction_local,
        "xyz_global": prediction_global,
        "conf_local": confidence.clone().requires_grad_(),
        "conf_global": confidence.clone().requires_grad_(),
    }
    target = {
        "xyz_local": target_local,
        "xyz_global": target_global,
        "conf_local": confidence,
        "conf_global": confidence,
    }
    valid = torch.ones(batch, frames, height, width, dtype=torch.bool)
    # RGB remains at the 448x560-equivalent input grid while predictions and
    # supervision use the smaller native DPT output grid.
    images = torch.rand(batch, frames, 3, height * 2, width * 2)
    ground_truth = target_local[..., 2] * 2.0
    loss_function = ScaredDistillationLoss(
        {
            "lambda_supervised_depth": 0.1,
            "min_depth": 0.1,
            "max_depth": 150.0,
        }
    )
    loss, logs = loss_function(
        prediction,
        target,
        images,
        valid,
        ground_truth,
        valid,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction_local.grad is not None
    assert "loss_supervised_depth" in logs


def test_npy_ground_truth_is_aligned_by_frame_id(tmp_path) -> None:
    np.save(tmp_path / "depth_0007.npy", np.full((3, 4), 2.0, np.float32))
    np.save(tmp_path / "depth_0012.npy", np.full((3, 4), 5.0, np.float32))
    depth, valid = load_clip_ground_truth(
        ["left_0012.png", "left_0007.png"],
        [str(tmp_path)],
        (6, 8),
        scale=2.0,
    )
    assert depth.shape == (2, 6, 8)
    assert valid.all()
    torch.testing.assert_close(depth[0], torch.full((6, 8), 10.0))
    torch.testing.assert_close(depth[1], torch.full((6, 8), 4.0))
