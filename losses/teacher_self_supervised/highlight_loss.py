"""Surface-normal highlight loss adapted from PC-Depth."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.teacher_self_supervised.geometry_warp import masked_mean, surface_normals


def highlight_surface_loss(
    local_points: torch.Tensor,
    highlight_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    normals = surface_normals(local_points)
    view = F.normalize(-local_points, dim=1, eps=1e-6)
    cosine = (view * normals).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    loss_map = (1.0 - cosine).square()
    mask = valid_mask.bool() & highlight_mask.bool()
    if not torch.any(mask):
        return loss_map.sum() * 0.0
    return masked_mean(loss_map, mask)


def edge_aware_depth_smoothness(
    depth: torch.Tensor,
    image: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mean_depth = depth.mean(dim=(2, 3), keepdim=True).clamp_min(1e-7)
    normalized = depth / mean_depth
    depth_x = (normalized[:, :, :, 1:] - normalized[:, :, :, :-1]).abs()
    depth_y = (normalized[:, :, 1:, :] - normalized[:, :, :-1, :]).abs()
    image_x = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(
        dim=1, keepdim=True
    )
    image_y = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(
        dim=1, keepdim=True
    )
    weight_x, weight_y = torch.exp(-image_x), torch.exp(-image_y)
    if valid_mask is not None:
        valid_x = valid_mask[:, :, :, 1:] & valid_mask[:, :, :, :-1]
        valid_y = valid_mask[:, :, 1:, :] & valid_mask[:, :, :-1, :]
    else:
        valid_x = torch.ones_like(depth_x, dtype=torch.bool)
        valid_y = torch.ones_like(depth_y, dtype=torch.bool)
    return masked_mean(depth_x * weight_x, valid_x) + masked_mean(
        depth_y * weight_y, valid_y
    )
