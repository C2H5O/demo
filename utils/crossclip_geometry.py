"""Differentiable Student-to-Teacher image-plane projection helpers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def resize_crop_intrinsics(
    intrinsics: torch.Tensor,
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int],
    crop_left: float = 0.0,
    crop_top: float = 0.0,
) -> torch.Tensor:
    """Update pinhole K for deterministic resize followed by crop."""
    original_height, original_width = original_size
    resized_height, resized_width = resized_size
    if min(original_height, original_width, resized_height, resized_width) <= 0:
        raise ValueError("Image sizes must be positive")
    output = intrinsics.clone()
    sx = float(resized_width) / float(original_width)
    sy = float(resized_height) / float(original_height)
    output[..., 0, 0] *= sx
    output[..., 1, 1] *= sy
    output[..., 0, 2] = output[..., 0, 2] * sx - float(crop_left)
    output[..., 1, 2] = output[..., 1, 2] * sy - float(crop_top)
    return output


def project_student_points_to_teacher(
    student_points: torch.Tensor,
    teacher_depth: torch.Tensor,
    teacher_intrinsics: torch.Tensor,
    teacher_valid_mask: torch.Tensor,
    teacher_confidence: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Project student local XYZ and bilinearly sample teacher depth.

    Shapes:
      student_points: ``[B,T,H,W,3]``
      teacher_depth/confidence/valid: ``[B,T,H,W]``
      teacher_intrinsics: ``[B,T,3,3]``

    The sampling grid is computed from student X/Y/Z, so gradients flow through
    all three student coordinates. Teacher tensors are treated as frozen data.
    """
    if student_points.ndim != 5 or student_points.shape[-1] != 3:
        raise ValueError("student_points must have shape [B,T,H,W,3]")
    batch, frames, height, width, _ = student_points.shape
    expected_map = (batch, frames, height, width)
    if tuple(teacher_depth.shape) != expected_map:
        raise ValueError("teacher_depth shape does not match student points")
    if tuple(teacher_valid_mask.shape) != expected_map:
        raise ValueError("teacher_valid_mask shape does not match student points")
    if tuple(teacher_intrinsics.shape) != (batch, frames, 3, 3):
        raise ValueError("teacher_intrinsics must have shape [B,T,3,3]")
    if teacher_confidence is not None and tuple(teacher_confidence.shape) != expected_map:
        raise ValueError("teacher_confidence shape does not match student points")

    finite_student = torch.isfinite(student_points).all(dim=-1)
    safe_points = torch.nan_to_num(student_points, nan=0.0, posinf=0.0, neginf=0.0)
    student_depth = safe_points[..., 2]
    projected = torch.einsum(
        "btij,bthwj->bthwi", teacher_intrinsics.to(safe_points), safe_points
    )
    projected_z = projected[..., 2]
    safe_projected_z = projected_z.clamp_min(eps)
    u = projected[..., 0] / safe_projected_z
    v = projected[..., 1] / safe_projected_z
    inside = (
        finite_student
        & torch.isfinite(u)
        & torch.isfinite(v)
        & (student_depth > eps)
        & (projected_z > eps)
        & (u >= 0.0)
        & (u <= width - 1.0)
        & (v >= 0.0)
        & (v <= height - 1.0)
    )
    normalized_u = 2.0 * u / max(width - 1, 1) - 1.0
    normalized_v = 2.0 * v / max(height - 1, 1) - 1.0
    grid = torch.stack((normalized_u, normalized_v), dim=-1)
    grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0)
    flat_grid = grid.reshape(batch * frames, height, width, 2)

    depth_input = torch.nan_to_num(
        teacher_depth.detach(), nan=0.0, posinf=0.0, neginf=0.0
    ).reshape(batch * frames, 1, height, width)
    sampled_depth = F.grid_sample(
        depth_input,
        flat_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch, frames, height, width)
    valid_input = teacher_valid_mask.detach().float().reshape(
        batch * frames, 1, height, width
    )
    sampled_valid = F.grid_sample(
        valid_input,
        flat_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch, frames, height, width) > 0.5
    valid = (
        inside
        & sampled_valid
        & torch.isfinite(sampled_depth)
        & (sampled_depth > eps)
    )
    if teacher_confidence is None:
        sampled_confidence = torch.ones_like(sampled_depth)
    else:
        confidence_input = torch.nan_to_num(
            teacher_confidence.detach(), nan=0.0, posinf=0.0, neginf=0.0
        ).reshape(batch * frames, 1, height, width)
        sampled_confidence = F.grid_sample(
            confidence_input,
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(batch, frames, height, width).clamp_min(0.0)
    return {
        "student_projected_depth": student_depth,
        "sampled_teacher_depth": sampled_depth,
        "sampled_teacher_confidence": sampled_confidence,
        "valid_mask": valid,
        "grid": grid,
    }


__all__ = ["project_student_points_to_teacher", "resize_crop_intrinsics"]
