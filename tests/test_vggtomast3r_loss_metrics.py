from __future__ import annotations

import pytest
import torch

from evaluation.vggtomast3r_metrics import patch_boundary_artifact
from losses.vggtomast3r_loss import VggToMast3RLoss
from utils.config import load_config


def _loss_inputs():
    pred_ref = torch.ones(1, 4, 5, 3, requires_grad=True)
    pred_other = torch.ones(1, 4, 5, 3, requires_grad=True)
    target = {
        "pts3d_ref": torch.ones_like(pred_ref),
        "pts3d_other_local": torch.ones_like(pred_other),
        "confidence_ref": torch.ones(1, 4, 5),
        "confidence_other": torch.ones(1, 4, 5),
        "valid_mask_ref": torch.ones(1, 4, 5, dtype=torch.bool),
        "valid_mask_other": torch.ones(1, 4, 5, dtype=torch.bool),
    }
    target["pts3d_ref"][..., 2] = 2.0
    target["pts3d_other_local"][..., 2] = 2.0
    prediction = {"pts3d_ref": pred_ref, "pts3d_other_local": pred_other}
    return prediction, target


def test_gt_mask_and_only_two_losses() -> None:
    prediction, target = _loss_inputs()
    gt = torch.full((1, 4, 5), 2.0)
    mask = torch.ones_like(gt, dtype=torch.bool)
    mask[:, 0] = False
    function = VggToMast3RLoss({"lambda_point": 1.0, "lambda_supervised_depth": 0.1})
    total, logs = function(prediction, target, gt, mask)
    total.backward()
    assert torch.isfinite(total)
    assert set(key for key in logs if key.startswith("loss_")) == {
        "loss_total", "loss_teacher_point_raw", "loss_teacher_point_weighted",
        "loss_scared_depth_raw", "loss_scared_depth_weighted",
    }


def test_patch_artifact_metric_detects_patch_boundary_jump() -> None:
    depth = torch.ones(28, 28)
    depth[:, 14:] += 5.0
    result = patch_boundary_artifact(depth, patch_size=14)
    assert result["patch_boundary_gradient"] > result["non_boundary_gradient"]
    assert result["patch_artifact_ratio"] > 1.0


def test_v1_supervised_depth_anchors_absolute_scale() -> None:
    config = load_config("configs/vggtomast3r_v1.yaml")
    assert config["loss"]["supervised_depth_scale_alignment"] == "none"

    prediction, target = _loss_inputs()
    gt = torch.full((1, 4, 5), 2.0)
    mask = torch.ones_like(gt, dtype=torch.bool)
    function = VggToMast3RLoss(config["loss"])
    _, base_logs = function(prediction, target, gt, mask)

    scaled_prediction = {
        name: value * 10.0 for name, value in prediction.items()
    }
    _, scaled_logs = function(scaled_prediction, target, gt, mask)

    # The normalized teacher point term remains scale invariant. The existing
    # supervised-depth term is not: it now anchors global student scale.
    assert abs(
        scaled_logs["loss_teacher_point_raw"]
        - base_logs["loss_teacher_point_raw"]
    ) < 1e-6
    assert scaled_logs["loss_scared_depth_raw"] > base_logs["loss_scared_depth_raw"]
    assert scaled_logs["supervised_depth_scale"] == 1.0


def test_v1_rejects_fully_scale_invariant_objective() -> None:
    with pytest.raises(ValueError, match="output scale unconstrained"):
        VggToMast3RLoss(
            {
                "lambda_point": 1.0,
                "lambda_supervised_depth": 0.1,
                "supervised_depth_scale_alignment": "median",
            }
        )
