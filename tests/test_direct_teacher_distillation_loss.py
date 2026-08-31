from __future__ import annotations

import torch

from losses.direct_teacher_distillation_loss import (
    CameraLossWeights,
    DirectTeacherDistillationLoss,
    compute_camera_distillation_loss,
    compute_direct_depth_distillation_loss,
    compute_highlight_aware_smoothness_loss,
    compute_highlight_surface_loss,
)


def _camera(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(batch, 16, 1, 1)
    intrinsics[..., 0, 0] = 500.0
    intrinsics[..., 1, 1] = 400.0
    intrinsics[..., 0, 2] = 280.0
    intrinsics[..., 1, 2] = 224.0
    extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(batch, 16, 1, 1)
    extrinsics[..., 0, 3] = torch.arange(16).float() * 0.1
    return intrinsics, extrinsics[..., :3, :]


def test_depth_exact_match_is_zero() -> None:
    teacher = torch.rand(2, 16, 3, 4) + 1.0
    loss, _ = compute_direct_depth_distillation_loss(
        teacher.clone(), teacher, torch.ones_like(teacher), torch.ones_like(teacher).bool()
    )
    assert loss.item() < 1e-8


def test_lower_confidence_reduces_pixel_contribution() -> None:
    teacher = torch.ones(1, 1, 1, 2)
    student = torch.tensor([[[[2.0, 11.0]]]])
    valid = torch.ones_like(teacher, dtype=torch.bool)
    uniform, _ = compute_direct_depth_distillation_loss(
        student, teacher, torch.ones_like(teacher), valid
    )
    weighted, _ = compute_direct_depth_distillation_loss(
        student, teacher, torch.tensor([[[[1.0, 0.01]]]]), valid
    )
    assert weighted < uniform
    assert weighted.item() < 1.1


def test_zero_confidence_falls_back_to_uniform_without_nan() -> None:
    teacher = torch.ones(1, 1, 1, 2)
    student = torch.tensor([[[[2.0, 11.0]]]], requires_grad=True)
    loss, diagnostics = compute_direct_depth_distillation_loss(
        student, teacher, torch.zeros_like(teacher), torch.ones_like(teacher).bool()
    )
    assert diagnostics["fallback"].item()
    assert torch.isfinite(loss)
    assert loss.item() == 5.5
    loss.backward()
    assert student.grad is not None


def test_camera_exact_match_is_near_zero() -> None:
    intrinsics, extrinsics = _camera()
    total, diagnostics = compute_camera_distillation_loss(
        intrinsics,
        extrinsics,
        intrinsics.clone(),
        extrinsics.clone(),
        (448, 560),
        CameraLossWeights(),
    )
    assert diagnostics["rotation"].item() < 0.002
    assert diagnostics["translation_direction"].item() < 1e-7
    assert diagnostics["translation_magnitude"].item() < 1e-7
    assert diagnostics["intrinsics"].item() < 1e-7
    assert total.item() < 0.002


def test_relative_camera_loss_is_invariant_to_absolute_world_gauge() -> None:
    intrinsics, teacher = _camera()
    gauge = torch.eye(4)
    gauge[:3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    gauge[:3, 3] = torch.tensor([3.0, -2.0, 5.0])
    teacher_h = torch.eye(4).view(1, 1, 4, 4).repeat(1, 16, 1, 1)
    teacher_h[..., :3, :] = teacher
    student = torch.matmul(teacher_h, torch.linalg.inv(gauge))[..., :3, :]
    total, diagnostics = compute_camera_distillation_loss(
        intrinsics,
        student,
        intrinsics,
        teacher,
        (448, 560),
        CameraLossWeights(),
    )
    assert diagnostics["rotation"].item() < 0.002
    assert diagnostics["translation_direction"].item() < 1e-6
    assert diagnostics["translation_magnitude"].item() < 1e-6
    assert total.item() < 0.002


def test_full_loss_gradients_reach_student_but_not_teacher() -> None:
    depth = torch.full((1, 16, 3, 4), 2.0, requires_grad=True)
    teacher_depth = torch.full((1, 16, 3, 4), 3.0, requires_grad=True)
    intrinsics, extrinsics = _camera()
    student_intrinsics = (intrinsics * 1.01).detach().requires_grad_()
    student_extrinsics = (extrinsics * 1.01).detach().requires_grad_()
    teacher_intrinsics = intrinsics.detach().requires_grad_()
    teacher_extrinsics = extrinsics.detach().requires_grad_()
    teacher_confidence = torch.ones_like(teacher_depth, requires_grad=True)
    rows, columns = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    points = torch.stack(
        (
            columns.float().view(1, 1, 3, 4).expand_as(depth) * depth,
            rows.float().view(1, 1, 3, 4).expand_as(depth) * depth,
            depth,
        ),
        dim=-1,
    )
    batch = {
        "absolute_frame_ids": torch.arange(16).view(1, 16),
        "clip_start": torch.tensor([0]),
        "highlight_masks": torch.zeros(1, 16, 1, 3, 4, dtype=torch.bool),
        "clean_images": torch.full((1, 16, 3, 3, 4), 0.5),
        "teacher": {
            "depth": teacher_depth,
            "confidence": teacher_confidence,
            "valid_mask": torch.ones_like(teacher_depth, dtype=torch.bool),
            "intrinsics": teacher_intrinsics,
            "extrinsics": teacher_extrinsics,
            "absolute_frame_ids": torch.arange(16).view(1, 16),
            "clip_start": torch.tensor([0]),
        },
    }
    loss_fn = DirectTeacherDistillationLoss(
        {
            "mode": "direct_teacher_distillation",
            "lambda_depth": 1.0,
            "lambda_camera": 0.1,
            "camera": {
                "lambda_rotation": 1.0,
                "lambda_translation_direction": 1.0,
                "lambda_translation_magnitude": 1.0,
                "lambda_intrinsics": 1.0,
            },
            "lambda_highlight": 0.01,
            "lambda_smooth": 0.1,
            "eps": 1e-6,
            "use_confidence_weight": True,
        }
    )
    total, logs = loss_fn(
        {
            "depth": depth,
            "intrinsics": student_intrinsics,
            "extrinsics": student_extrinsics,
            "xyz_local": points,
        },
        batch,
    )
    total.backward()
    assert depth.grad is not None
    assert student_intrinsics.grad is not None
    assert student_extrinsics.grad is not None
    assert teacher_depth.grad is None
    assert teacher_confidence.grad is None
    assert teacher_intrinsics.grad is None
    assert teacher_extrinsics.grad is None
    assert "loss/depth_raw" in logs and "loss/depth_weighted" in logs
    assert "loss/camera" in logs and "loss/camera_weighted" in logs


def test_highlight_and_smoothness_regularizers_still_backpropagate() -> None:
    points = torch.randn(1, 2, 5, 6, 3)
    points[..., 2].abs_().add_(1.0)
    points.requires_grad_()
    highlight = torch.zeros(1, 2, 1, 5, 6, dtype=torch.bool)
    highlight[:, :, :, 2, 2:4] = True
    clean = torch.full((1, 2, 3, 5, 6), 0.5)
    highlight_loss = compute_highlight_surface_loss(points, highlight)
    smooth_loss = compute_highlight_aware_smoothness_loss(points, clean, highlight)
    (highlight_loss + smooth_loss).backward()
    assert torch.isfinite(highlight_loss)
    assert torch.isfinite(smooth_loss)
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()


def test_depth_loss_reaches_toy_depth_head_and_lora_adapter() -> None:
    depth_head = torch.nn.Parameter(torch.tensor(1.0))
    lora_adapter = torch.nn.Parameter(torch.tensor(0.25))
    student = (depth_head + lora_adapter).expand(1, 1, 2, 2)
    teacher = torch.full_like(student, 2.0)
    loss, _ = compute_direct_depth_distillation_loss(
        student,
        teacher,
        torch.ones_like(teacher),
        torch.ones_like(teacher, dtype=torch.bool),
    )
    loss.backward()
    assert depth_head.grad is not None and depth_head.grad.abs().item() > 0.0
    assert lora_adapter.grad is not None and lora_adapter.grad.abs().item() > 0.0
