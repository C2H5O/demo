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
from losses.vggtomast3r_loss import VggToMast3RLoss, VggToMast3RLossConfig

__all__ += ["VggToMast3RLoss", "VggToMast3RLossConfig"]
