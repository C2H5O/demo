"""Evaluation-only diagnostics for DUNE patch-boundary depth artifacts."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def patch_boundary_artifact(
    depth: torch.Tensor,
    patch_size: int = 14,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    if depth.ndim != 2:
        raise ValueError("depth must be [H,W]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    finite = torch.isfinite(depth)
    valid = finite if valid_mask is None else finite & valid_mask.bool()
    dx = (depth[:, 1:] - depth[:, :-1]).abs()
    dy = (depth[1:, :] - depth[:-1, :]).abs()
    valid_x = valid[:, 1:] & valid[:, :-1]
    valid_y = valid[1:, :] & valid[:-1, :]
    x_positions = torch.arange(1, depth.shape[1], device=depth.device)
    y_positions = torch.arange(1, depth.shape[0], device=depth.device)
    boundary_x = (x_positions % patch_size == 0).unsqueeze(0).expand_as(dx)
    boundary_y = (y_positions % patch_size == 0).unsqueeze(1).expand_as(dy)
    boundary_values = torch.cat((dx[valid_x & boundary_x], dy[valid_y & boundary_y]))
    non_boundary_values = torch.cat((dx[valid_x & ~boundary_x], dy[valid_y & ~boundary_y]))
    if boundary_values.numel() == 0 or non_boundary_values.numel() == 0:
        raise ValueError("No valid boundary/non-boundary gradients")
    boundary = boundary_values.float().mean()
    non_boundary = non_boundary_values.float().mean()
    return {
        "patch_boundary_gradient": float(boundary.cpu()),
        "non_boundary_gradient": float(non_boundary.cpu()),
        "patch_artifact_ratio": float((boundary / non_boundary.clamp_min(1e-12)).cpu()),
    }


__all__ = ["patch_boundary_artifact"]
