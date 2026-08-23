"""The exclusive three-term objective for cross-clip teacher projection training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.teacher_self_supervised.geometry_warp import surface_normals
from utils.crossclip_geometry import project_student_points_to_teacher


@dataclass(frozen=True)
class CrossClipProjectionLossConfig:
    mode: str = "crossclip_projection_highlight_smooth"
    lambda_projection: float = 1.0
    lambda_highlight: float = 0.01
    lambda_smooth: float = 0.1
    projection_eps: float = 1e-6
    projection_ignore_highlight: bool = False
    use_confidence_weight: bool = True

    def validate(self) -> None:
        if self.mode != "crossclip_projection_highlight_smooth":
            raise ValueError("Unsupported cross-clip loss mode {!r}".format(self.mode))
        for name in ("lambda_projection", "lambda_highlight", "lambda_smooth"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError("{} cannot be negative".format(name))
        if self.projection_eps <= 0.0:
            raise ValueError("projection_eps must be positive")


def _samplewise_masked_mean(
    values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    weighted_mask = weights * mask.to(weights.dtype)
    numerator = (torch.where(mask, values, torch.zeros_like(values)) * weighted_mask).flatten(1).sum(1)
    denominator = weighted_mask.flatten(1).sum(1)
    return numerator / denominator.clamp_min(1.0), denominator


def compute_cross_clip_projection_loss(
    student_points: torch.Tensor,
    teacher_side: Dict[str, torch.Tensor],
    student_highlight_mask: torch.Tensor,
    config: CrossClipProjectionLossConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one 15-frame student overlap into its matching teacher clip."""
    projected = project_student_points_to_teacher(
        student_points,
        teacher_side["depth"],
        teacher_side["intrinsics"],
        teacher_side["valid_mask"],
        teacher_side["confidence"] if config.use_confidence_weight else None,
        eps=config.projection_eps,
    )
    valid = projected["valid_mask"]
    exists = teacher_side["exists"].bool()
    valid = valid & exists[:, None, None, None]
    if config.projection_ignore_highlight:
        valid = valid & ~student_highlight_mask[:, :, 0].bool()
    student_depth = projected["student_projected_depth"]
    teacher_depth = projected["sampled_teacher_depth"]
    residual = (student_depth - teacher_depth).abs() / (
        student_depth + teacher_depth + config.projection_eps
    )
    weights = (
        projected["sampled_teacher_confidence"]
        if config.use_confidence_weight
        else torch.ones_like(residual)
    )
    per_sample, valid_weight = _samplewise_masked_mean(residual, weights, valid)
    potential = float(student_points.shape[1] * student_points.shape[2] * student_points.shape[3])
    valid_ratio = valid.flatten(1).float().sum(1) / potential
    return per_sample, valid_weight, valid_ratio


def compute_highlight_surface_loss(
    student_points: torch.Tensor,
    highlight_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """PC-Depth-inspired camera-facing normal loss over highlight pixels."""
    if student_points.ndim != 5 or highlight_mask.ndim != 5:
        raise ValueError("Expected point [B,T,H,W,3] and highlight [B,T,1,H,W]")
    batch, frames, height, width, _ = student_points.shape
    finite = torch.isfinite(student_points).all(dim=-1) & (student_points[..., 2] > eps)
    safe = torch.nan_to_num(student_points, nan=0.0, posinf=0.0, neginf=0.0)
    points = safe.reshape(batch * frames, height, width, 3).permute(0, 3, 1, 2)
    normals = surface_normals(points)
    viewing = F.normalize(-points, dim=1, eps=eps)
    cosine = (viewing * normals).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    loss_map = (1.0 - cosine).square()
    valid = finite.reshape(batch * frames, 1, height, width)
    neighbor_valid = torch.zeros_like(valid)
    neighbor_valid[:, :, 1:-1, 1:-1] = (
        valid[:, :, 1:-1, 1:-1]
        & valid[:, :, 1:-1, :-2]
        & valid[:, :, 1:-1, 2:]
        & valid[:, :, :-2, 1:-1]
        & valid[:, :, 2:, 1:-1]
    )
    mask = highlight_mask.reshape(batch * frames, 1, height, width).bool() & neighbor_valid
    if not torch.any(mask):
        return torch.nan_to_num(student_points).sum() * 0.0
    return torch.where(mask, loss_map, torch.zeros_like(loss_map)).sum() / mask.sum().clamp_min(1)


def compute_highlight_aware_smoothness_loss(
    student_points: torch.Tensor,
    clean_images: torch.Tensor,
    highlight_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean-normalized inverse-depth smoothness with highlight edges excluded."""
    if student_points.ndim != 5 or clean_images.ndim != 5 or highlight_mask.ndim != 5:
        raise ValueError("Invalid temporal tensor rank for smoothness")
    depth = student_points[..., 2]
    valid = torch.isfinite(depth) & (depth > eps)
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))
    inverse = torch.where(valid, safe_depth.reciprocal(), torch.zeros_like(depth))
    count = valid.flatten(2).sum(2).clamp_min(1).to(inverse.dtype)
    mean_inverse = inverse.flatten(2).sum(2) / count
    normalized = inverse / mean_inverse[:, :, None, None].clamp_min(eps)
    depth_x = (normalized[:, :, :, 1:] - normalized[:, :, :, :-1]).abs()
    depth_y = (normalized[:, :, 1:, :] - normalized[:, :, :-1, :]).abs()
    image_x = (clean_images[:, :, :, :, 1:] - clean_images[:, :, :, :, :-1]).abs().mean(2)
    image_y = (clean_images[:, :, :, 1:, :] - clean_images[:, :, :, :-1, :]).abs().mean(2)
    highlight = highlight_mask[:, :, 0].bool()
    valid_x = (
        valid[:, :, :, 1:]
        & valid[:, :, :, :-1]
        & ~highlight[:, :, :, 1:]
        & ~highlight[:, :, :, :-1]
    )
    valid_y = (
        valid[:, :, 1:, :]
        & valid[:, :, :-1, :]
        & ~highlight[:, :, 1:, :]
        & ~highlight[:, :, :-1, :]
    )
    weighted_x = depth_x * torch.exp(-image_x)
    weighted_y = depth_y * torch.exp(-image_y)
    numerator = torch.where(valid_x, weighted_x, torch.zeros_like(weighted_x)).sum()
    numerator = numerator + torch.where(valid_y, weighted_y, torch.zeros_like(weighted_y)).sum()
    denominator = valid_x.sum() + valid_y.sum()
    if int(denominator.detach().cpu()) == 0:
        return torch.nan_to_num(student_points).sum() * 0.0
    return numerator / denominator.to(numerator.dtype)


class CrossClipProjectionLoss(nn.Module):
    """Total = projection + highlight + smoothness, with no fourth loss."""

    def __init__(
        self,
        config: Mapping[str, Any] | CrossClipProjectionLossConfig,
    ) -> None:
        super().__init__()
        self.config = (
            CrossClipProjectionLossConfig(**dict(config))
            if isinstance(config, Mapping)
            else config
        )
        self.config.validate()

    @staticmethod
    def _assert_absolute_mapping(
        batch: Dict[str, Any], side_name: str, student_slice: slice
    ) -> None:
        side = batch[side_name]
        exists = side["exists"].bool()
        if not torch.any(exists):
            return
        student_ids = batch["absolute_frame_ids"][:, student_slice]
        if not torch.equal(student_ids[exists].cpu(), side["absolute_frame_ids"][exists].cpu()):
            raise RuntimeError("{} absolute frame mapping mismatch".format(side_name))

    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        points = prediction["pts3d_local"]
        if points.ndim != 5 or points.shape[1] != 16 or points.shape[-1] != 3:
            raise ValueError("Student points must have shape [B,16,H,W,3]")
        self._assert_absolute_mapping(batch, "teacher_left", slice(0, 15))
        self._assert_absolute_mapping(batch, "teacher_right", slice(1, 16))
        highlight = batch["highlight_masks"].bool()
        left_per, _, left_ratio = compute_cross_clip_projection_loss(
            points[:, 0:15],
            batch["teacher_left"],
            highlight[:, 0:15],
            self.config,
        )
        right_per, _, right_ratio = compute_cross_clip_projection_loss(
            points[:, 1:16],
            batch["teacher_right"],
            highlight[:, 1:16],
            self.config,
        )
        left_exists = batch["teacher_left"]["exists"].to(points.dtype)
        right_exists = batch["teacher_right"]["exists"].to(points.dtype)
        side_count = left_exists + right_exists
        projection_per = (left_per * left_exists + right_per * right_exists) / side_count.clamp_min(1.0)
        projection = projection_per.mean()
        left = (left_per * left_exists).sum() / left_exists.sum().clamp_min(1.0)
        right = (right_per * right_exists).sum() / right_exists.sum().clamp_min(1.0)
        left_valid_ratio = (left_ratio * left_exists).sum() / left_exists.sum().clamp_min(1.0)
        right_valid_ratio = (right_ratio * right_exists).sum() / right_exists.sum().clamp_min(1.0)
        highlight_loss = compute_highlight_surface_loss(
            points, highlight, self.config.projection_eps
        )
        smooth = compute_highlight_aware_smoothness_loss(
            points,
            batch["clean_images"].float(),
            highlight,
            self.config.projection_eps,
        )
        total = (
            self.config.lambda_projection * projection
            + self.config.lambda_highlight * highlight_loss
            + self.config.lambda_smooth * smooth
        )
        tensors = {
            "loss/total": total,
            "loss/proj_left": left,
            "loss/proj_right": right,
            "loss/projection": projection,
            "loss/highlight": highlight_loss,
            "loss/smooth": smooth,
            "stats/proj_left_valid_ratio": left_valid_ratio,
            "stats/proj_right_valid_ratio": right_valid_ratio,
        }
        return total, {
            name: float(value.detach().cpu()) for name, value in tensors.items()
        }


__all__ = [
    "CrossClipProjectionLoss",
    "CrossClipProjectionLossConfig",
    "compute_cross_clip_projection_loss",
    "compute_highlight_aware_smoothness_loss",
    "compute_highlight_surface_loss",
]
