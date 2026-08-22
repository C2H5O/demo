"""Minimal V1 objective: per-frame local teacher points plus reference GT depth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from losses.supervised_depth_loss import SupervisedDepthLoss


@dataclass(frozen=True)
class VggToMast3RLossConfig:
    lambda_point: float = 1.0
    lambda_supervised_depth: float = 0.1
    charbonnier_eps: float = 1e-3
    confidence_floor: float = 0.02
    point_scale_mode: str = "avg_distance"
    min_depth: float = 0.1
    max_depth: float = 150.0
    supervised_depth_scale_alignment: str = "none"
    supervised_depth_loss: str = "log_l1"
    supervised_depth_min_depth: float = 1e-4
    supervised_depth_max_depth: float = 100.0
    all_other_losses: str = "disabled"


def _joint_scale(points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(points).all(dim=-1)
    if torch.any(valid & ~finite):
        raise ValueError("Point map has non-finite values at valid pixels")
    safe = torch.where(valid.unsqueeze(-1), points.float(), torch.zeros_like(points.float()))
    count = valid.sum(dim=(1, 2, 3), keepdim=True)
    scale = safe.norm(dim=-1).sum(dim=(1, 2, 3), keepdim=True) / count.clamp_min(1)
    return scale.clamp_min(1e-8).unsqueeze(-1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


class VggToMast3RLoss(nn.Module):
    def __init__(self, config: Dict[str, Any] | VggToMast3RLossConfig) -> None:
        super().__init__()
        self.config = VggToMast3RLossConfig(**config) if isinstance(config, dict) else config
        if self.config.point_scale_mode != "avg_distance":
            raise ValueError("V1 reuses point_scale_mode=avg_distance")
        if self.config.all_other_losses != "disabled":
            raise ValueError("V1 requires all_other_losses=disabled")
        if self.config.lambda_supervised_depth <= 0:
            raise ValueError(
                "V1 requires positive metric-depth supervision to anchor "
                "the scale-invariant teacher point objective"
            )
        if self.config.supervised_depth_scale_alignment != "none":
            raise ValueError(
                "V1 supervised depth must use scale_alignment=none; aligning "
                "both loss terms leaves MASt3R output scale unconstrained"
            )
        self.supervised_depth = SupervisedDepthLoss(
            {
                "min_depth": self.config.supervised_depth_min_depth,
                "max_depth": self.config.supervised_depth_max_depth,
                "scale_alignment": self.config.supervised_depth_scale_alignment,
                "loss": self.config.supervised_depth_loss,
            }
        )

    def _teacher_point_loss(
        self, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred = torch.stack((prediction["pts3d_ref"], prediction["pts3d_other_local"]), dim=1)
        teacher = torch.stack((target["pts3d_ref"], target["pts3d_other_local"]), dim=1)
        valid = torch.stack((target["valid_mask_ref"], target["valid_mask_other"]), dim=1).bool()
        confidence = torch.stack((target["confidence_ref"], target["confidence_other"]), dim=1).float()
        if pred.shape != teacher.shape or pred.ndim != 5 or pred.shape[-1] != 3:
            raise ValueError("Student/teacher pair point shapes differ: {} vs {}".format(tuple(pred.shape), tuple(teacher.shape)))
        if confidence.shape != valid.shape or valid.shape != pred.shape[:-1]:
            raise ValueError("Pair confidence/mask shape mismatch")
        teacher_depth = teacher[..., 2]
        valid = (
            valid
            & torch.isfinite(teacher).all(dim=-1)
            & torch.isfinite(confidence)
            & (teacher_depth >= self.config.min_depth)
            & (teacher_depth <= self.config.max_depth)
        )
        if not torch.any(valid):
            raise ValueError("No valid teacher pair points remain")
        pred_scale = _joint_scale(pred, valid)
        teacher_scale = _joint_scale(teacher, valid)
        normalized_pred = pred.float() / pred_scale
        normalized_teacher = teacher.float() / teacher_scale
        distance = torch.sqrt(
            (normalized_pred - normalized_teacher).square().sum(dim=-1)
            + self.config.charbonnier_eps**2
        )
        weight = confidence.detach().clamp(0.0, 1.0)
        if self.config.confidence_floor > 0:
            weight = torch.where(weight > 0, weight.clamp_min(self.config.confidence_floor), weight)
        weight = weight * valid.float()
        ref = _weighted_mean(distance[:, 0], weight[:, 0])
        other = _weighted_mean(distance[:, 1], weight[:, 1])
        point = 0.5 * (ref + other)
        return point, {
            "point_ref": ref,
            "point_other_local": other,
            "student_pair_scale": pred_scale.mean(),
            "teacher_pair_scale": teacher_scale.mean(),
            "teacher_valid_fraction": valid.float().mean(),
        }

    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        ground_truth_depth_ref: torch.Tensor | None = None,
        ground_truth_valid_mask_ref: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        point, point_parts = self._teacher_point_loss(prediction, target)
        supervised = point.new_zeros(())
        supervised_parts: Dict[str, torch.Tensor] = {}
        if ground_truth_depth_ref is not None:
            supervised, supervised_parts = self.supervised_depth(
                prediction["pts3d_ref"][..., 2].unsqueeze(1),
                ground_truth_depth_ref.unsqueeze(1),
                None if ground_truth_valid_mask_ref is None else ground_truth_valid_mask_ref.unsqueeze(1),
            )
        elif self.config.lambda_supervised_depth > 0:
            raise ValueError("lambda_supervised_depth > 0 but reference GT is absent")
        weighted_point = self.config.lambda_point * point
        weighted_depth = self.config.lambda_supervised_depth * supervised
        total = weighted_point + weighted_depth
        values = {
            "loss_total": total,
            "loss_teacher_point_raw": point,
            "loss_teacher_point_weighted": weighted_point,
            "loss_scared_depth_raw": supervised,
            "loss_scared_depth_weighted": weighted_depth,
            **point_parts,
            **supervised_parts,
        }
        return total, {name: float(value.detach().cpu()) for name, value in values.items()}


__all__ = ["VggToMast3RLoss", "VggToMast3RLossConfig"]
