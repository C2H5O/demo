"""Same-frame VGGT-Omega depth/camera distillation with unchanged regularizers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CameraLossWeights:
    lambda_rotation: float = 1.0
    lambda_translation_direction: float = 1.0
    lambda_translation_magnitude: float = 1.0
    lambda_intrinsics: float = 1.0

    def validate(self) -> None:
        for name in self.__dataclass_fields__:
            if float(getattr(self, name)) < 0.0:
                raise ValueError("loss.camera.{} cannot be negative".format(name))


@dataclass(frozen=True)
class DirectTeacherDistillationLossConfig:
    mode: str = "direct_teacher_distillation"
    lambda_depth: float = 1.0
    lambda_camera: float = 0.1
    camera: CameraLossWeights = CameraLossWeights()
    lambda_highlight: float = 0.01
    lambda_smooth: float = 0.1
    eps: float = 1e-6
    use_confidence_weight: bool = True

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any]
    ) -> "DirectTeacherDistillationLossConfig":
        values = dict(config)
        required = {
            "mode", "lambda_depth", "lambda_camera", "camera",
            "lambda_highlight", "lambda_smooth", "eps", "use_confidence_weight",
        }
        missing = sorted(required - set(values))
        if missing:
            raise ValueError("Loss config is missing explicit fields {}".format(missing))
        camera_values = dict(values["camera"])
        required_camera = set(CameraLossWeights.__dataclass_fields__)
        missing_camera = sorted(required_camera - set(camera_values))
        if missing_camera:
            raise ValueError(
                "Loss camera config is missing explicit weights {}".format(missing_camera)
            )
        values["camera"] = CameraLossWeights(**camera_values)
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if self.mode != "direct_teacher_distillation":
            raise ValueError("Unsupported loss mode {!r}".format(self.mode))
        for name in (
            "lambda_depth", "lambda_camera", "lambda_highlight", "lambda_smooth"
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError("loss.{} cannot be negative".format(name))
        self.camera.validate()
        if self.eps <= 0.0:
            raise ValueError("loss.eps must be positive")


def _differentiable_zero(anchor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(anchor.reshape(-1)[0]) * 0.0


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor, anchor: torch.Tensor
) -> torch.Tensor:
    del anchor
    numerator = torch.where(mask, values, torch.zeros_like(values)).sum()
    denominator = mask.sum().to(values.dtype)
    return numerator / denominator.clamp_min(1.0)


def compute_direct_depth_distillation_loss(
    student_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
    teacher_confidence: torch.Tensor,
    teacher_valid_mask: torch.Tensor,
    eps: float = 1e-6,
    use_confidence_weight: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Confidence-weighted per-sample L1 depth on identical pixels."""
    if student_depth.ndim != 4:
        raise ValueError("Student depth must have shape [B,T,H,W]")
    expected = tuple(student_depth.shape)
    for name, value in (
        ("teacher depth", teacher_depth),
        ("teacher confidence", teacher_confidence),
        ("teacher valid mask", teacher_valid_mask),
    ):
        if tuple(value.shape) != expected:
            raise ValueError("{} shape {} != {}".format(name, tuple(value.shape), expected))
    teacher_depth = teacher_depth.detach()
    teacher_confidence = teacher_confidence.detach()
    teacher_valid_mask = teacher_valid_mask.detach().bool()
    valid = (
        teacher_valid_mask
        & torch.isfinite(teacher_depth)
        & (teacher_depth > 0.0)
        & torch.isfinite(student_depth)
        & (student_depth > 0.0)
    )
    confidence = torch.nan_to_num(
        teacher_confidence.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    if not use_confidence_weight:
        confidence = torch.ones_like(confidence)
    flat_valid = valid.flatten(1)
    confidence_sum = torch.where(valid, confidence, torch.zeros_like(confidence)).flatten(1).sum(1)
    valid_count = flat_valid.sum(1)
    fallback = (confidence_sum <= eps) & (valid_count > 0)
    effective_weight = torch.where(
        fallback[:, None, None, None], torch.ones_like(confidence), confidence
    )
    weighted_mask = effective_weight * valid.to(effective_weight.dtype)
    residual = (student_depth.float() - teacher_depth.float()).abs()
    numerator = torch.where(valid, residual, torch.zeros_like(residual)).mul(
        weighted_mask
    ).flatten(1).sum(1)
    denominator = weighted_mask.flatten(1).sum(1)
    per_sample = numerator / denominator.clamp_min(eps)
    supervised = valid_count > 0
    loss = _masked_mean(per_sample, supervised, student_depth)
    diagnostics = {
        "valid": valid,
        "effective_weight": effective_weight,
        "valid_weight_sum": denominator,
        "fallback": fallback,
    }
    return loss, diagnostics


def _homogeneous_w2c(extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.ndim != 4 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError("Extrinsics must have shape [B,T,3,4]")
    bottom = extrinsics.new_zeros(extrinsics.shape[:2] + (1, 4))
    bottom[..., 0, 3] = 1.0
    return torch.cat((extrinsics, bottom), dim=-2)


def relative_w2c_from_reference(extrinsics: torch.Tensor) -> torch.Tensor:
    """Return T(i<-0) = E_w2c(i) @ inverse(E_w2c(0))."""
    homogeneous = _homogeneous_w2c(extrinsics)
    reference_inverse = torch.linalg.inv(homogeneous[:, 0:1])
    return torch.matmul(homogeneous[:, 1:], reference_inverse)


def compute_camera_distillation_loss(
    student_intrinsics: torch.Tensor,
    student_extrinsics: torch.Tensor,
    teacher_intrinsics: torch.Tensor,
    teacher_extrinsics: torch.Tensor,
    image_shape: tuple[int, int],
    weights: CameraLossWeights,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Distill relative W2C motion and per-frame normalized focal lengths."""
    teacher_intrinsics = teacher_intrinsics.detach()
    teacher_extrinsics = teacher_extrinsics.detach()
    if tuple(student_intrinsics.shape) != tuple(teacher_intrinsics.shape):
        raise ValueError("Student and teacher intrinsics shapes differ")
    if tuple(student_extrinsics.shape) != tuple(teacher_extrinsics.shape):
        raise ValueError("Student and teacher extrinsics shapes differ")
    if student_intrinsics.shape[1:] != (16, 3, 3):
        raise ValueError("Intrinsics must have shape [B,16,3,3]")
    student_relative = relative_w2c_from_reference(student_extrinsics.float())
    teacher_relative = relative_w2c_from_reference(teacher_extrinsics.float())
    student_rotation = student_relative[..., :3, :3]
    teacher_rotation = teacher_relative[..., :3, :3]
    rotation_error = torch.matmul(
        teacher_rotation.transpose(-1, -2), student_rotation
    )
    cosine_rotation = (
        (rotation_error.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    ).clamp(-1.0 + eps, 1.0 - eps)
    rotation = torch.acos(cosine_rotation).mean()

    student_translation = student_relative[..., :3, 3]
    teacher_translation = teacher_relative[..., :3, 3]
    student_magnitude = torch.linalg.vector_norm(student_translation, dim=-1)
    teacher_magnitude = torch.linalg.vector_norm(teacher_translation, dim=-1)
    translation_valid = torch.isfinite(teacher_magnitude) & (teacher_magnitude > eps)
    student_direction = F.normalize(student_translation, dim=-1, eps=eps)
    teacher_direction = F.normalize(teacher_translation, dim=-1, eps=eps)
    direction_cosine_map = (student_direction * teacher_direction).sum(-1).clamp(-1.0, 1.0)
    translation_direction = _masked_mean(
        1.0 - direction_cosine_map, translation_valid, student_extrinsics
    )
    magnitude_error = (
        torch.log(student_magnitude + eps) - torch.log(teacher_magnitude + eps)
    ).abs()
    translation_magnitude = _masked_mean(
        magnitude_error, translation_valid, student_extrinsics
    )

    height, width = image_shape
    student_fx = student_intrinsics[..., 0, 0].float()
    student_fy = student_intrinsics[..., 1, 1].float()
    teacher_fx = teacher_intrinsics[..., 0, 0].float()
    teacher_fy = teacher_intrinsics[..., 1, 1].float()
    intrinsics = (
        (student_fx / float(width) - teacher_fx / float(width)).abs()
        + (student_fy / float(height) - teacher_fy / float(height)).abs()
    ).mean()
    total = (
        weights.lambda_rotation * rotation
        + weights.lambda_translation_direction * translation_direction
        + weights.lambda_translation_magnitude * translation_magnitude
        + weights.lambda_intrinsics * intrinsics
    )
    valid_direction_cosine = _masked_mean(
        direction_cosine_map, translation_valid, student_extrinsics
    )
    direction_angle = _masked_mean(
        torch.acos(direction_cosine_map.clamp(-1.0 + eps, 1.0 - eps)),
        translation_valid,
        student_extrinsics,
    )
    diagnostics = {
        "rotation": rotation,
        "translation_direction": translation_direction,
        "translation_magnitude": translation_magnitude,
        "intrinsics": intrinsics,
        "rotation_degrees": rotation * (180.0 / math.pi),
        "translation_direction_cosine": valid_direction_cosine,
        "translation_direction_degrees": direction_angle * (180.0 / math.pi),
        "student_translation_magnitude_mean": _masked_mean(
            student_magnitude, translation_valid, student_extrinsics
        ),
        "teacher_translation_magnitude_mean": _masked_mean(
            teacher_magnitude, translation_valid, student_extrinsics
        ),
        "student_fx_mean": student_fx.mean(),
        "student_fy_mean": student_fy.mean(),
        "teacher_fx_mean": teacher_fx.mean(),
        "teacher_fy_mean": teacher_fy.mean(),
        "translation_valid_ratio": translation_valid.float().mean(),
    }
    return total, diagnostics


def surface_normals(points: torch.Tensor) -> torch.Tensor:
    """Compute camera-oriented normals for ``[B,3,H,W]`` point maps."""
    if points.ndim != 4 or points.shape[1] != 3:
        raise ValueError("points must have shape [B,3,H,W]")
    horizontal = points[:, :, 1:-1, 2:] - points[:, :, 1:-1, :-2]
    vertical = points[:, :, 2:, 1:-1] - points[:, :, :-2, 1:-1]
    normals = F.normalize(torch.cross(horizontal, vertical, dim=1), dim=1, eps=1e-6)
    normals = F.pad(normals, (1, 1, 1, 1))
    view = F.normalize(-points, dim=1, eps=1e-6)
    return torch.where((normals * view).sum(dim=1, keepdim=True) < 0, -normals, normals)


def compute_highlight_surface_loss(
    student_points: torch.Tensor,
    highlight_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Unchanged PC-Depth-inspired camera-facing normal loss."""
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
        return _differentiable_zero(student_points)
    return torch.where(mask, loss_map, torch.zeros_like(loss_map)).sum() / mask.sum().clamp_min(1)


def compute_highlight_aware_smoothness_loss(
    student_points: torch.Tensor,
    clean_images: torch.Tensor,
    highlight_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Unchanged mean-normalized inverse-depth, edge-aware smoothness."""
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
    valid_x = valid[:, :, :, 1:] & valid[:, :, :, :-1] & ~highlight[:, :, :, 1:] & ~highlight[:, :, :, :-1]
    valid_y = valid[:, :, 1:, :] & valid[:, :, :-1, :] & ~highlight[:, :, 1:, :] & ~highlight[:, :, :-1, :]
    weighted_x = depth_x * torch.exp(-image_x)
    weighted_y = depth_y * torch.exp(-image_y)
    numerator = torch.where(valid_x, weighted_x, torch.zeros_like(weighted_x)).sum()
    numerator = numerator + torch.where(valid_y, weighted_y, torch.zeros_like(weighted_y)).sum()
    denominator = valid_x.sum() + valid_y.sum()
    if int(denominator.detach().cpu()) == 0:
        return _differentiable_zero(student_points)
    return numerator / denominator.to(numerator.dtype)


def _positive_stats(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    selected = value.detach()[mask.detach()]
    if selected.numel() == 0:
        zero = value.detach().new_zeros(())
        return zero, zero, zero
    selected = selected.float()
    return selected.min(), selected.max(), selected.mean()


class DirectTeacherDistillationLoss(nn.Module):
    def __init__(
        self,
        config: Mapping[str, Any] | DirectTeacherDistillationLossConfig,
    ) -> None:
        super().__init__()
        self.config = (
            DirectTeacherDistillationLossConfig.from_mapping(config)
            if isinstance(config, Mapping)
            else config
        )
        self.config.validate()

    @staticmethod
    def _assert_mapping(batch: Dict[str, Any]) -> None:
        teacher = batch["teacher"]
        if not torch.equal(
            batch["absolute_frame_ids"].detach().cpu(),
            teacher["absolute_frame_ids"].detach().cpu(),
        ):
            raise RuntimeError("Student and teacher absolute_frame_ids differ")
        if not torch.equal(
            batch["clip_start"].detach().cpu(), teacher["clip_start"].detach().cpu()
        ):
            raise RuntimeError("Student and teacher clip_start differ")

    def forward(
        self, prediction: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._assert_mapping(batch)
        student_depth = prediction["depth"]
        points = prediction["xyz_local"]
        teacher = batch["teacher"]
        if tuple(points.shape) != tuple(student_depth.shape) + (3,):
            raise ValueError("Student xyz_local does not match depth")
        depth, depth_diagnostics = compute_direct_depth_distillation_loss(
            student_depth,
            teacher["depth"],
            teacher["confidence"],
            teacher["valid_mask"],
            eps=self.config.eps,
            use_confidence_weight=self.config.use_confidence_weight,
        )
        camera, camera_diagnostics = compute_camera_distillation_loss(
            prediction["intrinsics"],
            prediction["extrinsics"],
            teacher["intrinsics"],
            teacher["extrinsics"],
            tuple(int(value) for value in student_depth.shape[-2:]),
            self.config.camera,
            eps=self.config.eps,
        )
        highlight = compute_highlight_surface_loss(
            points, batch["highlight_masks"].bool(), self.config.eps
        )
        smooth = compute_highlight_aware_smoothness_loss(
            points,
            batch["clean_images"].float(),
            batch["highlight_masks"].bool(),
            self.config.eps,
        )
        depth_weighted = self.config.lambda_depth * depth
        camera_weighted = self.config.lambda_camera * camera
        highlight_weighted = self.config.lambda_highlight * highlight
        smooth_weighted = self.config.lambda_smooth * smooth
        total = depth_weighted + camera_weighted + highlight_weighted + smooth_weighted

        valid = depth_diagnostics["valid"]
        teacher_valid = (
            teacher["valid_mask"].bool()
            & torch.isfinite(teacher["depth"])
            & (teacher["depth"] > 0.0)
        )
        student_valid = torch.isfinite(student_depth) & (student_depth > 0.0)
        student_min, student_max, student_mean = _positive_stats(student_depth, student_valid)
        teacher_min, teacher_max, teacher_mean = _positive_stats(teacher["depth"], teacher_valid)
        valid_confidence = teacher["confidence"].detach()[valid]
        confidence_valid_mean = (
            valid_confidence.float().mean()
            if valid_confidence.numel()
            else teacher["confidence"].new_zeros(())
        )
        camera_weights = self.config.camera
        tensors = {
            "loss/total": total,
            "loss/depth_raw": depth,
            "loss/depth_weighted": depth_weighted,
            "loss/camera": camera,
            "loss/camera_weighted": camera_weighted,
            "loss/camera_rotation": camera_diagnostics["rotation"],
            "loss/camera_translation_direction": camera_diagnostics["translation_direction"],
            "loss/camera_translation_magnitude": camera_diagnostics["translation_magnitude"],
            "loss/camera_intrinsics": camera_diagnostics["intrinsics"],
            "loss/camera_rotation_weighted": self.config.lambda_camera * camera_weights.lambda_rotation * camera_diagnostics["rotation"],
            "loss/camera_translation_direction_weighted": self.config.lambda_camera * camera_weights.lambda_translation_direction * camera_diagnostics["translation_direction"],
            "loss/camera_translation_magnitude_weighted": self.config.lambda_camera * camera_weights.lambda_translation_magnitude * camera_diagnostics["translation_magnitude"],
            "loss/camera_intrinsics_weighted": self.config.lambda_camera * camera_weights.lambda_intrinsics * camera_diagnostics["intrinsics"],
            "loss/highlight": highlight,
            "loss/highlight_weighted": highlight_weighted,
            "loss/smooth": smooth,
            "loss/smooth_weighted": smooth_weighted,
            "stats/depth_valid_ratio": valid.float().mean(),
            "stats/depth_confidence_fallback_ratio": depth_diagnostics["fallback"].float().mean(),
            "stats/teacher_confidence_mean": torch.nan_to_num(teacher["confidence"].detach().float()).mean(),
            "stats/teacher_confidence_valid_mean": confidence_valid_mean,
            "stats/student_depth_min": student_min,
            "stats/student_depth_max": student_max,
            "stats/student_depth_mean": student_mean,
            "stats/teacher_depth_min": teacher_min,
            "stats/teacher_depth_max": teacher_max,
            "stats/teacher_depth_mean": teacher_mean,
            "stats/camera_rotation_error_degrees": camera_diagnostics["rotation_degrees"],
            "stats/camera_translation_direction_cosine": camera_diagnostics["translation_direction_cosine"],
            "stats/camera_translation_direction_error_degrees": camera_diagnostics["translation_direction_degrees"],
            "stats/student_relative_translation_magnitude_mean": camera_diagnostics["student_translation_magnitude_mean"],
            "stats/teacher_relative_translation_magnitude_mean": camera_diagnostics["teacher_translation_magnitude_mean"],
            "stats/camera_translation_valid_ratio": camera_diagnostics["translation_valid_ratio"],
            "stats/student_fx_mean": camera_diagnostics["student_fx_mean"],
            "stats/student_fy_mean": camera_diagnostics["student_fy_mean"],
            "stats/teacher_fx_mean": camera_diagnostics["teacher_fx_mean"],
            "stats/teacher_fy_mean": camera_diagnostics["teacher_fy_mean"],
        }
        return total, {name: float(value.detach().cpu()) for name, value in tensors.items()}


__all__ = [
    "CameraLossWeights",
    "DirectTeacherDistillationLoss",
    "DirectTeacherDistillationLossConfig",
    "compute_camera_distillation_loss",
    "compute_direct_depth_distillation_loss",
    "compute_highlight_aware_smoothness_loss",
    "compute_highlight_surface_loss",
    "relative_w2c_from_reference",
]
