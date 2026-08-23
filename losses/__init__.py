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
from losses.crossclip_projection_loss import (
    CrossClipProjectionLoss,
    CrossClipProjectionLossConfig,
)

__all__ += [
    "CrossClipProjectionLoss",
    "CrossClipProjectionLossConfig",
    "VggToMast3RLoss",
    "VggToMast3RLossConfig",
]
