from copy import deepcopy
from pathlib import Path

import pytest
import torch

from losses.direct_teacher_distillation_loss import (
    DirectTeacherDistillationLoss,
    compute_clip_shared_affine_disparity_distillation_loss,
)
from trainers.direct_teacher_distillation_trainer import _check_resume_contract
from utils.checkpoint import DIRECT_TEACHER_DISTILLATION_PROTOCOL
from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _affine_depth_pair(batch=2, frames=16, height=2, width=3):
    student_disparity = torch.linspace(
        0.2, 1.4, batch * frames * height * width
    ).reshape(batch, frames, height, width)
    scale = torch.tensor([1.5, 0.7])[:batch]
    shift = torch.tensor([0.1, 0.3])[:batch]
    teacher_disparity = (
        scale[:, None, None, None] * student_disparity
        + shift[:, None, None, None]
    )
    return student_disparity.reciprocal(), teacher_disparity.reciprocal(), scale, shift


def test_one_affine_pair_per_sample_matches_complete_clip():
    student, teacher, expected_scale, expected_shift = _affine_depth_pair()
    loss, diagnostics = compute_clip_shared_affine_disparity_distillation_loss(
        student,
        teacher,
        torch.ones_like(teacher),
        torch.ones_like(teacher, dtype=torch.bool),
    )
    assert diagnostics["scale"].shape == (2,)
    assert diagnostics["shift"].shape == (2,)
    assert diagnostics["valid_frames"].shape == (2, 16)
    torch.testing.assert_close(diagnostics["scale"], expected_scale, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(diagnostics["shift"], expected_shift, rtol=1e-5, atol=1e-5)
    assert loss.item() < 1e-10


def test_affine_solver_is_detached_but_student_depth_keeps_gradient():
    depth_head = torch.nn.Parameter(torch.tensor(0.9))
    lora_adapter = torch.nn.Parameter(torch.tensor(0.15))
    pattern = torch.linspace(1.0, 2.0, 16 * 2 * 3).reshape(1, 16, 2, 3)
    student = (depth_head + lora_adapter) * pattern
    teacher = pattern * 1.4 + torch.linspace(0.0, 0.2, 16).view(1, 16, 1, 1)
    loss, diagnostics = compute_clip_shared_affine_disparity_distillation_loss(
        student,
        teacher,
        torch.ones_like(teacher),
        torch.ones_like(teacher, dtype=torch.bool),
    )
    assert not diagnostics["scale"].requires_grad
    assert not diagnostics["shift"].requires_grad
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert depth_head.grad is not None and depth_head.grad.abs().item() > 0.0
    assert lora_adapter.grad is not None and lora_adapter.grad.abs().item() > 0.0


def test_residual_reduction_is_frame_balanced_after_clip_fit():
    student = torch.ones(1, 16, 1, 4)
    teacher_disparity = torch.full_like(student, 4.0)
    teacher_disparity[:, 0] = 2.0
    teacher = teacher_disparity.reciprocal()
    valid = torch.ones_like(teacher, dtype=torch.bool)
    valid[:, 0, :, 1:] = False
    loss, diagnostics = compute_clip_shared_affine_disparity_distillation_loss(
        student, teacher, torch.ones_like(teacher), valid
    )
    assert diagnostics["fallback"].item()
    # Smooth-L1(1)=0.5 and Smooth-L1(3)=2.5; all 16 frames receive equal weight.
    assert loss.item() == pytest.approx((0.5 + 15 * 2.5) / 16)


def test_batch_reduction_gives_each_supervised_clip_equal_weight():
    student = torch.ones(2, 16, 1, 1)
    teacher_disparity = torch.full_like(student, 4.0)
    teacher_disparity[0] = 2.0
    teacher = teacher_disparity.reciprocal()
    valid = torch.ones_like(teacher, dtype=torch.bool)
    valid[0, 1:] = False
    loss, _ = compute_clip_shared_affine_disparity_distillation_loss(
        student, teacher, torch.ones_like(teacher), valid
    )
    # Clip losses are Smooth-L1(1)=0.5 and Smooth-L1(3)=2.5.
    assert loss.item() == pytest.approx(1.5)


def test_confidence_and_invalid_depth_follow_safe_baseline_e_pipeline():
    student, teacher, _, _ = _affine_depth_pair(batch=1)
    confidence = torch.zeros_like(teacher)
    student = student.clone().requires_grad_()
    student.data[0, 0, 0, 0] = float("nan")
    teacher[0, 1, 0, 0] = 0.0
    loss, diagnostics = compute_clip_shared_affine_disparity_distillation_loss(
        student, teacher, confidence, torch.ones_like(teacher, dtype=torch.bool)
    )
    assert diagnostics["confidence_fallback"].item()
    assert not diagnostics["fallback"].item()
    assert torch.isfinite(loss)
    assert torch.isfinite(diagnostics["scale"]).all()
    assert torch.isfinite(diagnostics["shift"]).all()
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_degenerate_clip_uses_finite_identity_affine_fallback():
    student = torch.full((2, 16, 2, 3), 2.0, requires_grad=True)
    teacher = torch.full_like(student, 3.0)
    loss, diagnostics = compute_clip_shared_affine_disparity_distillation_loss(
        student,
        teacher,
        torch.ones_like(teacher),
        torch.ones_like(teacher, dtype=torch.bool),
    )
    assert diagnostics["fallback"].tolist() == [True, True]
    torch.testing.assert_close(diagnostics["scale"], torch.ones(2))
    torch.testing.assert_close(diagnostics["shift"], torch.zeros(2))
    assert torch.isfinite(loss)


def test_smallest_positive_depth_cannot_overflow_disparity_loss():
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    student = torch.ones(1, 16, 1, 2, requires_grad=True)
    teacher = torch.ones_like(student)
    student.data[0, 0, 0, 0] = tiny
    teacher[0, 1, 0, 0] = tiny
    loss, diagnostics = compute_clip_shared_affine_disparity_distillation_loss(
        student,
        teacher,
        torch.ones_like(teacher),
        torch.ones_like(teacher, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(diagnostics["student_disparity"]).all()
    assert torch.isfinite(diagnostics["teacher_disparity"]).all()


def test_baseline_i_config_inherits_baseline_e_except_depth_ablation_outputs():
    baseline_e = load_config(ROOT / "configs/baselines/E.yaml")
    baseline_i = load_config(ROOT / "configs/baselines/I.yaml")
    assert baseline_i["attention_distill"] == baseline_e["attention_distill"]
    assert baseline_i["student"] == baseline_e["student"]
    assert baseline_i["teacher"] == baseline_e["teacher"]
    assert baseline_i["dataset"] == baseline_e["dataset"]
    for key in (
        "lambda_depth", "lambda_camera", "camera", "lambda_highlight",
        "lambda_smooth", "eps", "use_confidence_weight", "highlight_mode",
        "highlight_cone_full_angle_degrees", "highlight_softness",
    ):
        assert baseline_i["loss"][key] == baseline_e["loss"][key]
    assert baseline_i["loss"]["depth_mode"] == "clip_shared_affine_disparity"
    assert baseline_i["loss"]["depth_robust_loss"] == "smooth_l1"
    assert baseline_i["loss"]["affine_detach"] is True

    expected = deepcopy(baseline_e)
    expected["experiment"] = baseline_i["experiment"]
    expected["loss"].update(
        depth_mode="clip_shared_affine_disparity",
        depth_robust_loss="smooth_l1",
        affine_detach=True,
    )
    expected["training"] = baseline_i["training"]
    expected["vda_evaluation"] = baseline_i["vda_evaluation"]
    expected["visualization"] = baseline_i["visualization"]
    assert baseline_i == expected


def test_baseline_i_cannot_resume_a_baseline_e_checkpoint():
    baseline_e = load_config(ROOT / "configs/baselines/E.yaml")
    baseline_i = load_config(ROOT / "configs/baselines/I.yaml")
    checkpoint = {
        "objective_protocol": DIRECT_TEACHER_DISTILLATION_PROTOCOL,
        "config": baseline_e,
    }
    with pytest.raises(ValueError, match="loss settings differ"):
        _check_resume_contract(checkpoint, baseline_i, object())


def test_full_loss_routes_affine_depth_and_emits_required_diagnostics(monkeypatch):
    config = load_config(ROOT / "configs/baselines/I.yaml")["loss"]
    loss_module = DirectTeacherDistillationLoss(config)
    student, teacher, _, _ = _affine_depth_pair(batch=1, height=3, width=4)
    student = student.requires_grad_()
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(1, 16, 1, 1)
    extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(1, 16, 1, 1)[..., :3, :]
    points = torch.stack((torch.zeros_like(student), torch.zeros_like(student), student), dim=-1)
    batch = {
        "absolute_frame_ids": torch.arange(16).view(1, 16),
        "clip_start": torch.tensor([0]),
        "highlight_masks": torch.zeros(1, 16, 1, 3, 4, dtype=torch.bool),
        "clean_images": torch.full((1, 16, 3, 3, 4), 0.5),
        "teacher": {
            "depth": teacher,
            "confidence": torch.zeros_like(teacher),
            "valid_mask": torch.ones_like(teacher, dtype=torch.bool),
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "absolute_frame_ids": torch.arange(16).view(1, 16),
            "clip_start": torch.tensor([0]),
        },
    }
    total, logs = loss_module(
        {
            "depth": student,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "xyz_local": points,
        },
        batch,
    )
    assert torch.isfinite(total)
    for name in (
        "loss/depth_raw", "loss/depth_weighted",
        "stats/depth_affine_scale_mean", "stats/depth_affine_scale_min",
        "stats/depth_affine_scale_max", "stats/depth_affine_shift_mean",
        "stats/depth_affine_shift_min", "stats/depth_affine_shift_max",
        "stats/depth_affine_fallback_ratio", "stats/depth_valid_ratio",
        "stats/depth_valid_frames_mean", "stats/student_disparity_mean",
        "stats/teacher_disparity_mean", "stats/aligned_student_disparity_mean",
    ):
        assert name in logs and torch.isfinite(torch.tensor(logs[name]))
    assert logs["stats/depth_confidence_fallback_ratio"] == pytest.approx(1.0)
    assert logs["stats/depth_affine_fallback_ratio"] == pytest.approx(0.0)
