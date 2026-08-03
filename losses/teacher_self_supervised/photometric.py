"""PC-Depth/EndoDAC-style photometric and light-alignment primitives."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.teacher_self_supervised.geometry_warp import surface_normals


class SSIM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(3, 1)
        self.pad = nn.ReflectionPad2d(1)
        self.c1 = 0.01**2
        self.c2 = 0.03**2

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first, second = self.pad(first), self.pad(second)
        mean_first, mean_second = self.pool(first), self.pool(second)
        variance_first = self.pool(first.square()) - mean_first.square()
        variance_second = self.pool(second.square()) - mean_second.square()
        covariance = self.pool(first * second) - mean_first * mean_second
        numerator = (2 * mean_first * mean_second + self.c1) * (
            2 * covariance + self.c2
        )
        denominator = (
            mean_first.square() + mean_second.square() + self.c1
        ) * (variance_first + variance_second + self.c2)
        return ((1.0 - numerator / denominator.clamp_min(1e-7)) * 0.5).clamp(
            0.0, 1.0
        )


def photometric_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    ssim: SSIM,
    ssim_weight: float,
) -> torch.Tensor:
    l1 = (target - prediction).abs().mean(dim=1, keepdim=True)
    if ssim_weight <= 0:
        return l1
    structural = ssim(target, prediction).mean(dim=1, keepdim=True)
    return (1.0 - ssim_weight) * l1 + ssim_weight * structural


def light_align_warped_source(
    warped_source: torch.Tensor,
    target_image: torch.Tensor,
    target_points: torch.Tensor,
    source_points: torch.Tensor,
    target_to_source: torch.Tensor,
    valid_mask: torch.Tensor,
    excluded_mask: torch.Tensor,
    mu: float,
    gamma: float,
    minimum_cosine: float,
) -> Dict[str, torch.Tensor]:
    """Apply PC-Depth's spatial light-source correction to a warped view."""
    eps = 1e-7
    target_normals = surface_normals(target_points)
    target_view = F.normalize(-target_points, dim=1, eps=eps)
    source_view = F.normalize(-source_points, dim=1, eps=eps)
    source_normals = target_to_source[:, :3, :3] @ target_normals.flatten(2)
    source_normals = source_normals.reshape_as(target_normals)

    theta_target = (target_view * target_normals).sum(dim=1, keepdim=True)
    theta_source = (source_view * source_normals).sum(dim=1, keepdim=True)
    light = target_points.new_tensor((0.0, 0.0, -1.0)).view(1, 3, 1, 1)
    phi_target = (light * target_view).sum(dim=1, keepdim=True)
    phi_source = (light * source_view).sum(dim=1, keepdim=True)
    radius_target = target_points.square().sum(dim=1, keepdim=True)
    radius_source = source_points.square().sum(dim=1, keepdim=True)
    radiance_target = (
        torch.exp(-mu * (1.0 - phi_target))
        * theta_target.clamp_min(eps)
        / radius_target.clamp_min(eps)
    )
    radiance_source = (
        torch.exp(-mu * (1.0 - phi_source))
        * theta_source.clamp_min(eps)
        / radius_source.clamp_min(eps)
    )
    stable = (
        valid_mask
        & ~excluded_mask.bool()
        & (theta_target >= minimum_cosine)
        & (theta_source >= minimum_cosine)
        & torch.isfinite(radiance_target)
        & torch.isfinite(radiance_source)
    )
    ratio = radiance_target / radiance_source.clamp_min(eps)
    ratio = torch.where(stable, ratio, torch.ones_like(ratio))
    source_gamma = warped_source.clamp_min(0.0).pow(gamma)
    target_gamma = target_image.clamp_min(0.0).pow(gamma)
    source_scaled = source_gamma * ratio
    stable_rgb = stable.expand_as(source_scaled)
    numerator = (target_gamma * stable_rgb).flatten(1).sum(dim=1)
    denominator = (source_scaled * stable_rgb).flatten(1).sum(dim=1)
    gain = (numerator / denominator.clamp_min(eps)).view(-1, 1, 1, 1)
    gain = torch.where(
        stable.flatten(1).any(dim=1).view(-1, 1, 1, 1),
        gain,
        torch.ones_like(gain),
    )
    correction = (ratio * gain).clamp_min(eps).pow(1.0 / gamma).detach()
    refined = (warped_source * correction).clamp(0.0, 1.0)
    return {
        "refined_image": refined,
        "correction": correction,
        "radiance_target": radiance_target,
        "stable_mask": stable,
        "target_normals": target_normals,
    }
