"""Scale-aligned SCARED ground-truth depth supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SupervisedDepthLossConfig:
    min_depth: float = 0.1
    max_depth: float = 150.0
    scale_alignment: str = "median"
    loss: str = "log_l1"


class SupervisedDepthLoss(nn.Module):
    def __init__(
        self,
        config: SupervisedDepthLossConfig | Dict | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = SupervisedDepthLossConfig()
        elif isinstance(config, dict):
            config = SupervisedDepthLossConfig(**config)
        self.config = config
        if self.config.scale_alignment not in {"median", "none"}:
            raise ValueError("scale_alignment must be median or none")
        if self.config.loss not in {"log_l1", "smooth_l1"}:
            raise ValueError("loss must be log_l1 or smooth_l1")

    def forward(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                "Prediction/GT depth shapes differ: {} vs {}".format(
                    tuple(prediction.shape), tuple(ground_truth.shape)
                )
            )
        valid = (
            torch.isfinite(prediction)
            & torch.isfinite(ground_truth)
            & (ground_truth >= self.config.min_depth)
            & (ground_truth <= self.config.max_depth)
            & (prediction > 0)
        )
        if valid_mask is not None:
            valid = valid & valid_mask.bool()
        if not torch.any(valid):
            zero = prediction.float().sum() * 0.0
            return zero, {
                "supervised_depth_scale": prediction.new_tensor(1.0),
                "supervised_depth_valid_fraction": valid.float().mean(),
            }

        aligned = prediction.float()
        scales = []
        if self.config.scale_alignment == "median":
            aligned = aligned.clone()
            for batch_index in range(prediction.shape[0]):
                for frame_index in range(prediction.shape[1]):
                    frame_valid = valid[batch_index, frame_index]
                    if not torch.any(frame_valid):
                        scales.append(prediction.new_tensor(1.0))
                        continue
                    pred_values = prediction[batch_index, frame_index][frame_valid]
                    gt_values = ground_truth[batch_index, frame_index][frame_valid]
                    scale = (
                        gt_values.detach().median()
                        / pred_values.detach().median().clamp_min(1e-7)
                    )
                    aligned[batch_index, frame_index] = (
                        aligned[batch_index, frame_index] * scale
                    )
                    scales.append(scale)
        else:
            scales.append(prediction.new_tensor(1.0))

        if self.config.loss == "log_l1":
            loss_map = (
                torch.log(aligned.clamp_min(1e-7))
                - torch.log(ground_truth.float().clamp_min(1e-7))
            ).abs()
        else:
            loss_map = F.smooth_l1_loss(
                aligned, ground_truth.float(), reduction="none"
            )
        loss = loss_map[valid].mean()
        return loss, {
            "supervised_depth_scale": torch.stack(scales).mean(),
            "supervised_depth_valid_fraction": valid.float().mean(),
        }
