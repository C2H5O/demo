"""Geometry and confidence distillation objective for SCARED point maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.supervised_depth_loss import SupervisedDepthLoss


@dataclass
class ScaredDistillationLossConfig:
    lambda_point: float = 1.0
    lambda_geo: float = 0.25
    lambda_smooth: float = 0.03
    lambda_conf: float = 0.1
    lambda_supervised_depth: float = 0.0
    lambda_global: float = 1.0
    lambda_local: float = 1.0
    alpha_dist: float = 1.0
    alpha_normal: float = 0.5
    charbonnier_eps: float = 1e-3
    confidence_floor: float = 0.02
    point_scale_mode: str = "avg_distance"
    min_depth: float = 0.1
    max_depth: float = 150.0
    normalize_mode: str = "imagenet"
    supervised_depth_scale_alignment: str = "median"
    supervised_depth_loss: str = "log_l1"
    supervised_depth_min_depth: Optional[float] = None
    supervised_depth_max_depth: Optional[float] = None


def _weighted_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _masked_stats(values: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    selected = values.detach().float()[mask]
    if selected.numel() == 0:
        zero = values.detach().new_zeros((), dtype=torch.float32)
        return zero, zero
    return selected.mean(), selected.std(unbiased=False)


def _teacher_weight(confidence: Optional[torch.Tensor], valid: Optional[torch.Tensor], floor: float) -> Optional[torch.Tensor]:
    weight = None
    if confidence is not None:
        weight = confidence.detach().float().clamp(0.0, 1.0)
        if floor > 0:
            weight = torch.where(weight > 0, weight.clamp_min(floor), weight)
    if valid is not None:
        valid_float = valid.detach().float()
        weight = valid_float if weight is None else weight * valid_float
    return weight


def _charbonnier_xyz(prediction: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
    difference = prediction.float() - target.float()
    return torch.sqrt(difference.square().sum(dim=-1) + epsilon * epsilon)


def _neighbor_differences(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    horizontal = points[:, :, :, 1:, :] - points[:, :, :, :-1, :]
    vertical = points[:, :, 1:, :, :] - points[:, :, :-1, :, :]
    return horizontal, vertical


def _neighbor_weights(weight: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if weight is None:
        return None, None
    horizontal = weight[:, :, :, 1:] * weight[:, :, :, :-1]
    vertical = weight[:, :, 1:, :] * weight[:, :, :-1, :]
    return horizontal, vertical


def _surface_normals(points: torch.Tensor) -> torch.Tensor:
    horizontal = points[:, :, 1:-1, 2:, :] - points[:, :, 1:-1, :-2, :]
    vertical = points[:, :, 2:, 1:-1, :] - points[:, :, :-2, 1:-1, :]
    return F.normalize(torch.cross(horizontal, vertical, dim=-1), dim=-1, eps=1e-6)


def _surface_normal_weights(weight: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if weight is None:
        return None
    horizontal = torch.minimum(weight[:, :, 1:-1, 2:], weight[:, :, 1:-1, :-2])
    vertical = torch.minimum(weight[:, :, 2:, 1:-1], weight[:, :, :-2, 1:-1])
    return torch.minimum(horizontal, vertical)


def _average_distance_scale(
    points: torch.Tensor,
    valid: torch.Tensor,
    dimensions: Tuple[int, ...],
) -> torch.Tensor:
    """Return a differentiable mean point-distance scale over selected axes."""
    points = points.float()
    point_finite = torch.isfinite(points).all(dim=-1)
    if torch.any(valid & ~point_finite):
        raise ValueError("Point map contains NaN or Inf at a valid teacher pixel")
    safe_points = torch.where(valid.unsqueeze(-1), points, torch.zeros_like(points))
    safe_distances = safe_points.norm(dim=-1)
    count = valid.sum(dim=dimensions, keepdim=True)
    scale = safe_distances.sum(dim=dimensions, keepdim=True) / count.clamp_min(1).to(safe_distances.dtype)
    scale = torch.where(count > 0, scale, torch.ones_like(scale))
    return scale.clamp_min(1e-8).unsqueeze(-1)


def _normalize_point_maps(
    prediction: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    valid: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Normalize global maps per clip and local maps independently per frame."""
    pred_global_scale = _average_distance_scale(prediction["xyz_global"], valid, (1, 2, 3))
    target_global_scale = _average_distance_scale(target["xyz_global"], valid, (1, 2, 3))
    pred_local_scale = _average_distance_scale(prediction["xyz_local"], valid, (2, 3))
    target_local_scale = _average_distance_scale(target["xyz_local"], valid, (2, 3))
    valid_xyz = valid.unsqueeze(-1)

    normalized_prediction = dict(prediction)
    normalized_target = dict(target)
    pred_global = torch.where(
        valid_xyz,
        prediction["xyz_global"].float(),
        torch.zeros_like(prediction["xyz_global"], dtype=torch.float32),
    )
    target_global = torch.where(
        valid_xyz,
        target["xyz_global"].float(),
        torch.zeros_like(target["xyz_global"], dtype=torch.float32),
    )
    pred_local = torch.where(
        valid_xyz,
        prediction["xyz_local"].float(),
        torch.zeros_like(prediction["xyz_local"], dtype=torch.float32),
    )
    target_local = torch.where(
        valid_xyz,
        target["xyz_local"].float(),
        torch.zeros_like(target["xyz_local"], dtype=torch.float32),
    )
    normalized_prediction["xyz_global"] = pred_global / pred_global_scale
    normalized_target["xyz_global"] = target_global / target_global_scale
    normalized_prediction["xyz_local"] = pred_local / pred_local_scale
    normalized_target["xyz_local"] = target_local / target_local_scale
    scales = {
        "pred_global_scale": pred_global_scale,
        "teacher_global_scale": target_global_scale,
        "pred_local_scale": pred_local_scale,
        "teacher_local_scale": target_local_scale,
    }
    return normalized_prediction, normalized_target, scales


def _zero_one_images(images: torch.Tensor, normalize_mode: str) -> torch.Tensor:
    images = images.float()
    if normalize_mode in ("none", "zero_one"):
        return images.clamp(0.0, 1.0)
    if normalize_mode == "minus_one_one":
        return ((images + 1.0) * 0.5).clamp(0.0, 1.0)
    if normalize_mode == "imagenet":
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
        return (images * std + mean).clamp(0.0, 1.0)
    raise ValueError("Unknown normalize_mode {!r}".format(normalize_mode))


class ScaredDistillationLoss(nn.Module):
    """Point, structure, smoothness, and student confidence distillation loss."""

    def __init__(self, config: Union[ScaredDistillationLossConfig, Dict[str, Any]]) -> None:
        super().__init__()
        self.config = ScaredDistillationLossConfig(**config) if isinstance(config, dict) else config
        if self.config.point_scale_mode != "avg_distance":
            raise ValueError("point_scale_mode must be 'avg_distance'")
        if self.config.min_depth < 0 or self.config.max_depth <= self.config.min_depth:
            raise ValueError(
                "Expected 0 <= min_depth < max_depth, got {} and {}".format(
                    self.config.min_depth, self.config.max_depth
                )
            )
        supervised_min_depth = (
            self.config.min_depth
            if self.config.supervised_depth_min_depth is None
            else self.config.supervised_depth_min_depth
        )
        supervised_max_depth = (
            self.config.max_depth
            if self.config.supervised_depth_max_depth is None
            else self.config.supervised_depth_max_depth
        )
        if (
            supervised_min_depth < 0
            or supervised_max_depth <= supervised_min_depth
        ):
            raise ValueError(
                "Expected valid supervised depth bounds, got {} and {}".format(
                    supervised_min_depth, supervised_max_depth
                )
            )
        self.supervised_depth = SupervisedDepthLoss(
            {
                "min_depth": supervised_min_depth,
                "max_depth": supervised_max_depth,
                "scale_alignment": self.config.supervised_depth_scale_alignment,
                "loss": self.config.supervised_depth_loss,
            }
        )

    def _effective_valid_mask(
        self,
        target: Dict[str, torch.Tensor],
        valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Combine cache validity with finite XYZ and the teacher local-Z range."""
        local = target["xyz_local"]
        global_points = target["xyz_global"]
        expected_shape = local.shape[:-1]
        if local.ndim != 5 or local.shape[-1] != 3 or global_points.shape != local.shape:
            raise ValueError(
                "Teacher xyz_global/xyz_local must both have shape [B,T,H,W,3], got {} and {}".format(
                    tuple(global_points.shape), tuple(local.shape)
                )
            )
        if valid is None:
            effective = torch.ones(expected_shape, dtype=torch.bool, device=local.device)
        else:
            if tuple(valid.shape) != tuple(expected_shape):
                raise ValueError(
                    "valid_mask shape mismatch: expected {}, got {}".format(
                        tuple(expected_shape), tuple(valid.shape)
                    )
                )
            effective = valid.detach().bool()
        depth = local[..., 2]
        effective = effective & torch.isfinite(local).all(dim=-1)
        effective = effective & torch.isfinite(global_points).all(dim=-1)
        effective = effective & (depth >= self.config.min_depth) & (depth <= self.config.max_depth)
        for name in ("conf_global", "conf_local"):
            if name in target:
                confidence = target[name]
                if confidence.ndim == effective.ndim + 1 and confidence.shape[-1] == 1:
                    confidence = confidence.squeeze(-1)
                if confidence.shape != effective.shape:
                    raise ValueError(
                        "Teacher {} shape mismatch: expected {}, got {}".format(
                            name, tuple(effective.shape), tuple(confidence.shape)
                        )
                    )
                effective = effective & torch.isfinite(confidence)
        if not torch.any(effective):
            raise ValueError(
                "No valid teacher points remain in depth range [{}, {}]".format(
                    self.config.min_depth, self.config.max_depth
                )
            )
        return effective

    def _point_loss(
        self,
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        global_weight = _teacher_weight(target.get("conf_global"), valid, self.config.confidence_floor)
        local_weight = _teacher_weight(target.get("conf_local"), valid, self.config.confidence_floor)
        global_loss = _weighted_mean(
            _charbonnier_xyz(prediction["xyz_global"], target["xyz_global"], self.config.charbonnier_eps),
            global_weight,
        )
        local_loss = _weighted_mean(
            _charbonnier_xyz(prediction["xyz_local"], target["xyz_local"], self.config.charbonnier_eps),
            local_weight,
        )
        total = self.config.lambda_global * global_loss + self.config.lambda_local * local_loss
        return total, {"point_global": global_loss, "point_local": local_loss}

    def _one_geometry_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = prediction.float()
        target = target.float()
        prediction_x, prediction_y = _neighbor_differences(prediction)
        target_x, target_y = _neighbor_differences(target)
        weight_x, weight_y = _neighbor_weights(weight)
        distance = 0.5 * (
            _weighted_mean((prediction_x.norm(dim=-1) - target_x.norm(dim=-1)).abs(), weight_x)
            + _weighted_mean((prediction_y.norm(dim=-1) - target_y.norm(dim=-1)).abs(), weight_y)
        )
        prediction_normal = _surface_normals(prediction)
        target_normal = _surface_normals(target)
        normal_map = 1.0 - (prediction_normal * target_normal).sum(dim=-1).clamp(-1.0, 1.0)
        normal_weight = _surface_normal_weights(weight)
        normal = _weighted_mean(normal_map, normal_weight)
        return distance, normal

    def _geometry_loss(
        self,
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        global_weight = _teacher_weight(target.get("conf_global"), valid, self.config.confidence_floor)
        local_weight = _teacher_weight(target.get("conf_local"), valid, self.config.confidence_floor)
        global_dist, global_normal = self._one_geometry_loss(
            prediction["xyz_global"], target["xyz_global"], global_weight
        )
        local_dist, local_normal = self._one_geometry_loss(
            prediction["xyz_local"], target["xyz_local"], local_weight
        )
        distance = 0.5 * (global_dist + local_dist)
        normal = 0.5 * (global_normal + local_normal)
        total = self.config.alpha_dist * distance + self.config.alpha_normal * normal
        return total, {"geo_dist": distance, "geo_normal": normal}

    def _smoothness_loss(
        self,
        prediction: Dict[str, torch.Tensor],
        images: torch.Tensor,
        valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        points = prediction["xyz_local"].float()
        horizontal, vertical = _neighbor_differences(points)
        rgb = _zero_one_images(images, self.config.normalize_mode).mean(dim=2)
        image_x = (rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).abs()
        image_y = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs()
        edge_x, edge_y = torch.exp(-10.0 * image_x), torch.exp(-10.0 * image_y)
        if valid is not None:
            valid_x, valid_y = _neighbor_weights(valid.float())
            edge_x = edge_x * valid_x
            edge_y = edge_y * valid_y
        return 0.5 * (
            _weighted_mean(horizontal.abs().sum(dim=-1), edge_x)
            + _weighted_mean(vertical.abs().sum(dim=-1), edge_y)
        )

    @staticmethod
    def _prepare_confidence_pair(
        student_conf: torch.Tensor,
        teacher_conf: torch.Tensor,
        valid: Optional[torch.Tensor],
        name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        student_conf = student_conf.float()
        teacher_conf = teacher_conf.detach().float().clamp(0.0, 1.0)

        if teacher_conf.ndim == student_conf.ndim + 1 and teacher_conf.shape[-1] == 1:
            teacher_conf = teacher_conf.squeeze(-1)
        if student_conf.shape != teacher_conf.shape:
            raise ValueError(
                "{} confidence shape mismatch: student={}, teacher={}".format(
                    name, tuple(student_conf.shape), tuple(teacher_conf.shape)
                )
            )
        if not torch.isfinite(student_conf).all():
            raise ValueError("Student {} confidence contains NaN or Inf".format(name))
        if ((student_conf < 0.0) | (student_conf > 1.0)).any():
            raise ValueError(
                "Student {} confidence must already be sigmoid-normalized to [0, 1]".format(name)
            )

        if valid is None:
            effective_valid = torch.ones_like(student_conf, dtype=torch.bool)
        else:
            effective_valid = valid.detach()
            if effective_valid.ndim == student_conf.ndim + 1 and effective_valid.shape[-1] == 1:
                effective_valid = effective_valid.squeeze(-1)
            if effective_valid.shape != student_conf.shape:
                raise ValueError(
                    "valid_mask shape mismatch for {} confidence: valid={}, confidence={}".format(
                        name, tuple(effective_valid.shape), tuple(student_conf.shape)
                    )
                )
            effective_valid = effective_valid.bool()

        teacher_finite = torch.isfinite(teacher_conf)
        effective_valid = effective_valid & teacher_finite
        # Invalid teacher entries are excluded by the mask; a finite placeholder
        # only prevents NaNs from propagating through the elementwise loss map.
        teacher_conf = torch.where(
            teacher_finite,
            teacher_conf,
            torch.zeros_like(teacher_conf),
        )
        return student_conf, teacher_conf, effective_valid

    def _confidence_loss(
        self,
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        required_keys = ("conf_global", "conf_local")
        for source_name, source in (("prediction", prediction), ("target", target)):
            missing = [key for key in required_keys if key not in source]
            if missing:
                raise KeyError(
                    "{} is missing required confidence key(s): {}".format(
                        source_name, ", ".join(missing)
                    )
                )

        student_global, teacher_global, global_valid = self._prepare_confidence_pair(
            prediction["conf_global"], target["conf_global"], valid, "global"
        )
        student_local, teacher_local, local_valid = self._prepare_confidence_pair(
            prediction["conf_local"], target["conf_local"], valid, "local"
        )
        global_loss_map = F.smooth_l1_loss(student_global, teacher_global, reduction="none")
        local_loss_map = F.smooth_l1_loss(student_local, teacher_local, reduction="none")
        global_loss = _weighted_mean(global_loss_map, global_valid)
        local_loss = _weighted_mean(local_loss_map, local_valid)
        confidence = 0.5 * (global_loss + local_loss)

        student_global_mean, student_global_std = _masked_stats(student_global, global_valid)
        student_local_mean, student_local_std = _masked_stats(student_local, local_valid)
        teacher_global_mean, teacher_global_std = _masked_stats(teacher_global, global_valid)
        teacher_local_mean, teacher_local_std = _masked_stats(teacher_local, local_valid)
        return confidence, {
            "conf_global": global_loss,
            "conf_local": local_loss,
            "student_conf_global_mean": student_global_mean,
            "student_conf_global_std": student_global_std,
            "student_conf_local_mean": student_local_mean,
            "student_conf_local_std": student_local_std,
            "teacher_conf_global_mean": teacher_global_mean,
            "teacher_conf_global_std": teacher_global_std,
            "teacher_conf_local_mean": teacher_local_mean,
            "teacher_conf_local_std": teacher_local_std,
        }

    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        images: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        ground_truth_depth: Optional[torch.Tensor] = None,
        ground_truth_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        effective_valid = self._effective_valid_mask(target, valid_mask)
        normalized_prediction, normalized_target, scales = _normalize_point_maps(
            prediction, target, effective_valid
        )
        point, point_parts = self._point_loss(normalized_prediction, normalized_target, effective_valid)
        geometry, geometry_parts = self._geometry_loss(
            normalized_prediction, normalized_target, effective_valid
        )
        smoothness = self._smoothness_loss(normalized_prediction, images, effective_valid)
        confidence, confidence_parts = self._confidence_loss(prediction, target, effective_valid)
        supervised = point.new_zeros(())
        supervised_parts: Dict[str, torch.Tensor] = {}
        if ground_truth_depth is not None:
            supervised, supervised_parts = self.supervised_depth(
                prediction["xyz_local"][..., 2],
                ground_truth_depth,
                ground_truth_valid_mask,
            )
        elif self.config.lambda_supervised_depth > 0:
            raise ValueError(
                "lambda_supervised_depth > 0 but the batch has no SCARED ground truth"
            )
        weighted_confidence = self.config.lambda_conf * confidence
        weighted_supervised = self.config.lambda_supervised_depth * supervised
        total = (
            self.config.lambda_point * point
            + self.config.lambda_geo * geometry
            + self.config.lambda_smooth * smoothness
            + weighted_confidence
            + weighted_supervised
        )
        tensors = {
            "loss_total": total,
            "loss_point": point,
            "loss_geo": geometry,
            "loss_smooth": smoothness,
            "loss_conf": confidence,
            "weighted_conf": weighted_confidence,
            "loss_supervised_depth": supervised,
            "weighted_supervised_depth": weighted_supervised,
            "valid_depth_fraction": effective_valid.float().mean(),
            **{name: value.mean() for name, value in scales.items()},
            **point_parts,
            **geometry_parts,
            **confidence_parts,
            **supervised_parts,
        }
        logs = {name: float(value.detach().cpu()) for name, value in tensors.items()}
        return total, logs
