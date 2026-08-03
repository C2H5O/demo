from __future__ import annotations

import torch


def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    """Convert RGB image tensor in [0, 1] to HSV.

    Accepts [..., 3, H, W] and returns the same leading shape.
    """
    r, g, b = image.unbind(dim=-3)
    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
    delta = maxc - minc

    hue = torch.zeros_like(maxc)
    nonzero = delta > 1e-8
    hue = torch.where((maxc == r) & nonzero, ((g - b) / delta.clamp_min(1e-8)) % 6, hue)
    hue = torch.where((maxc == g) & nonzero, ((b - r) / delta.clamp_min(1e-8)) + 2, hue)
    hue = torch.where((maxc == b) & nonzero, ((r - g) / delta.clamp_min(1e-8)) + 4, hue)
    hue = hue / 6.0

    sat = torch.where(maxc > 1e-8, delta / maxc.clamp_min(1e-8), torch.zeros_like(maxc))
    val = maxc
    return torch.stack([hue, sat, val], dim=-3)


def specular_mask(
    image: torch.Tensor,
    value_threshold: float = 0.86,
    saturation_threshold: float = 0.28,
) -> torch.Tensor:
    """Detect likely specular pixels using high value and low saturation."""
    hsv = rgb_to_hsv(image.clamp(0, 1))
    sat = hsv[..., 1, :, :]
    val = hsv[..., 2, :, :]
    return (val > value_threshold) & (sat < saturation_threshold)


def frame_specular_ratio(mask: torch.Tensor) -> torch.Tensor:
    """Return per-frame specular area ratio for mask [..., H, W]."""
    return mask.float().flatten(-2).mean(dim=-1)

