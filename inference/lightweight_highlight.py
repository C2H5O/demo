"""Pure Torch inference-only frame-ranking proxy, not PC-Depth segmentation."""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from inference.kv_sampling import resolve_lightweight_highlight_options


@torch.no_grad()
def compute_lightweight_highlight_scores(frames: torch.Tensor,
                                         options: Mapping | None = None) -> torch.Tensor:
    """Score [N,3,H,W] RGB in [0,1]; return float32 [N] on the same device.

    Batch spatial pooling precedes high-brightness/low-saturation thresholding.
    No per-frame Python work, image host copies, OpenCV or scalar synchronization.
    Input normalization is owned by the existing sequence RGB loader.
    """
    config = resolve_lightweight_highlight_options(options)
    if frames.ndim != 4 or frames.shape[1] != 3 or not frames.is_floating_point():
        raise ValueError("Lightweight highlight input must be floating RGB [N,3,H,W] in [0,1]")
    factor = config["downsample_factor"]
    if min(frames.shape[-2:]) < factor:
        raise ValueError("downsample_factor exceeds the input spatial dimensions")
    if frames.shape[0] == 0:
        return frames.new_empty((0,), dtype=torch.float32)
    # Explicit FP32 keeps the proxy independent of model AMP dtype.
    rgb = F.avg_pool2d(frames.float(), kernel_size=factor, stride=factor)
    value = rgb.amax(dim=1)
    min_rgb = rgb.amin(dim=1)
    saturation = (value - min_rgb) / value.clamp_min(1e-6)
    highlight = ((value >= config["brightness_threshold"])
                 & (saturation <= config["saturation_threshold"]))
    return highlight.float().mean(dim=(-2, -1))
