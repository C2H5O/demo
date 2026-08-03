from losses.distillation_loss import (
    ScaredDistillationLoss,
    ScaredDistillationLossConfig,
)
from losses.supervised_depth_loss import SupervisedDepthLoss

DistillationLoss = ScaredDistillationLoss

__all__ = [
    "DistillationLoss",
    "ScaredDistillationLoss",
    "ScaredDistillationLossConfig",
    "SupervisedDepthLoss",
]
