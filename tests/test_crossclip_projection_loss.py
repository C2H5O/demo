from __future__ import annotations

import torch

from losses.crossclip_projection_loss import (
    CrossClipProjectionLoss,
    CrossClipProjectionLossConfig,
    compute_cross_clip_projection_loss,
    compute_highlight_aware_smoothness_loss,
    compute_highlight_surface_loss,
)
from utils.crossclip_geometry import (
    project_student_points_to_teacher,
    resize_crop_intrinsics,
)


def _identity_plane(batch=1, frames=1, height=3, width=4, depth=2.0):
    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    points = torch.stack(
        (columns * depth, rows * depth, torch.full_like(columns, depth)), dim=-1
    )
    return points.view(1, 1, height, width, 3).repeat(batch, frames, 1, 1, 1)


def _teacher_side(depth, exists=True):
    batch, frames, height, width = depth.shape
    return {
        "exists": torch.full((batch,), exists, dtype=torch.bool),
        "depth": depth,
        "confidence": torch.ones_like(depth),
        "valid_mask": torch.ones_like(depth, dtype=torch.bool),
        "intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(batch, frames, 1, 1),
    }


def test_identity_camera_projection_returns_corresponding_pixels() -> None:
    points = _identity_plane()
    teacher = torch.full((1, 1, 3, 4), 2.0)
    result = project_student_points_to_teacher(
        points,
        teacher,
        torch.eye(3).view(1, 1, 3, 3),
        torch.ones_like(teacher, dtype=torch.bool),
    )
    assert result["valid_mask"].all()
    torch.testing.assert_close(result["sampled_teacher_depth"], teacher)
    rows, columns = torch.meshgrid(
        torch.arange(3), torch.arange(4), indexing="ij"
    )
    torch.testing.assert_close(
        result["grid"][0, 0, ..., 0], columns.float() * 2.0 / 3.0 - 1.0
    )
    torch.testing.assert_close(
        result["grid"][0, 0, ..., 1], rows.float() - 1.0
    )


def test_projection_loss_zero_for_equal_depth_and_positive_for_ten_percent_scale() -> None:
    teacher_depth = torch.full((1, 15, 3, 4), 2.0)
    points = _identity_plane(frames=15)
    mask = torch.zeros(1, 15, 1, 3, 4, dtype=torch.bool)
    config = CrossClipProjectionLossConfig(use_confidence_weight=False)
    equal, _, _ = compute_cross_clip_projection_loss(
        points, _teacher_side(teacher_depth), mask, config
    )
    scaled, _, _ = compute_cross_clip_projection_loss(
        points * 1.1, _teacher_side(teacher_depth), mask, config
    )
    assert equal.item() < 1e-7
    assert scaled.item() > 0.04


def test_invalid_projection_points_are_masked_without_nan() -> None:
    points = _identity_plane(height=2, width=3)
    points[0, 0, 0, 0] = torch.tensor([float("nan"), 0.0, 1.0])
    points[0, 0, 0, 1, 2] = 0.0
    points[0, 0, 0, 2, 0] = 1000.0
    teacher = torch.ones(1, 1, 2, 3) * 2.0
    result = project_student_points_to_teacher(
        points,
        teacher,
        torch.eye(3).view(1, 1, 3, 3),
        torch.ones_like(teacher, dtype=torch.bool),
    )
    assert not result["valid_mask"][0, 0, 0].any()
    assert torch.isfinite(result["sampled_teacher_depth"]).all()


def test_empty_highlight_mask_is_zero_and_smoothness_still_works() -> None:
    points = _identity_plane(frames=16, height=3, width=4)
    empty = torch.zeros(1, 16, 1, 3, 4, dtype=torch.bool)
    clean = torch.full((1, 16, 3, 3, 4), 0.5)
    highlight = compute_highlight_surface_loss(points, empty)
    smooth = compute_highlight_aware_smoothness_loss(points, clean, empty)
    assert highlight.item() == 0.0
    assert torch.isfinite(smooth)
    assert smooth.item() < 1e-7


def test_total_loss_contains_exactly_three_terms_and_no_neighbor_projection() -> None:
    points = _identity_plane(frames=16, height=3, width=4)
    batch = {
        "absolute_frame_ids": torch.arange(16).view(1, 16),
        "highlight_masks": torch.zeros(1, 16, 1, 3, 4, dtype=torch.bool),
        "clean_images": torch.full((1, 16, 3, 3, 4), 0.5),
        "teacher_left": {
            **_teacher_side(torch.zeros(1, 15, 3, 4), exists=False),
            "absolute_frame_ids": torch.full((1, 15), -1),
        },
        "teacher_right": {
            **_teacher_side(torch.zeros(1, 15, 3, 4), exists=False),
            "absolute_frame_ids": torch.full((1, 15), -1),
        },
    }
    loss_fn = CrossClipProjectionLoss(
        {
            "mode": "crossclip_projection_highlight_smooth",
            "lambda_projection": 1.0,
            "lambda_highlight": 0.01,
            "lambda_smooth": 0.1,
            "projection_eps": 1e-6,
            "projection_ignore_highlight": False,
            "use_confidence_weight": True,
        }
    )
    total, logs = loss_fn({"pts3d_local": points}, batch)
    assert logs["loss/projection"] == 0.0
    assert set(name for name in logs if name.startswith("loss/")) == {
        "loss/total",
        "loss/proj_left",
        "loss/proj_right",
        "loss/projection",
        "loss/highlight",
        "loss/smooth",
    }
    expected = logs["loss/projection"] + 0.01 * logs["loss/highlight"] + 0.1 * logs["loss/smooth"]
    assert abs(float(total) - expected) < 1e-7


def test_resize_crop_intrinsics_updates_focal_and_principal_point() -> None:
    intrinsics = torch.tensor([[100.0, 0.0, 50.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]])
    updated = resize_crop_intrinsics(
        intrinsics, (100, 200), (50, 100), crop_left=5, crop_top=3
    )
    torch.testing.assert_close(
        updated,
        torch.tensor([[50.0, 0.0, 20.0], [0.0, 40.0, 17.0], [0.0, 0.0, 1.0]]),
    )
