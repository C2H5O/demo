"""Differentiable depth/camera geometry for the DA3 student."""

from __future__ import annotations

import torch


WORLD_TO_CAMERA = "world_to_camera: X_camera = R @ X_world + t"


def _as_homogeneous(extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError("extrinsics must end in [3,4] or [4,4]")
    bottom = extrinsics.new_zeros(extrinsics.shape[:-2] + (1, 4))
    bottom[..., 0, 3] = 1.0
    return torch.cat((extrinsics, bottom), dim=-2)


def depth_intrinsics_to_local_points(
    depth: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """Back-project ``[B,T,H,W]`` depth with pixel-space K to camera XYZ."""
    if depth.ndim != 4:
        raise ValueError("depth must have shape [B,T,H,W]")
    batch, frames, height, width = depth.shape
    if tuple(intrinsics.shape) != (batch, frames, 3, 3):
        raise ValueError("intrinsics must have shape [B,T,3,3]")
    if not torch.isfinite(depth).all() or not torch.isfinite(intrinsics).all():
        raise FloatingPointError("depth/intrinsics contain non-finite values")
    if torch.any(depth <= 0):
        raise ValueError("DA3 depth must be strictly positive")
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    if torch.any(fx <= 0) or torch.any(fy <= 0):
        raise ValueError("DA3 focal lengths must be positive")
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = (xs.view(1, 1, height, width) - cx[..., None, None]) / fx[..., None, None] * depth
    y = (ys.view(1, 1, height, width) - cy[..., None, None]) / fy[..., None, None] * depth
    points = torch.stack((x, y, depth), dim=-1)
    if not torch.equal(points[..., 2], depth):
        raise RuntimeError("depth != xyz_local[...,2]")
    return points


def local_to_global_points(
    xyz_local: torch.Tensor, extrinsics_w2c: torch.Tensor
) -> torch.Tensor:
    """Transform camera XYZ to world XYZ using differentiable inverse W2C poses."""
    if xyz_local.ndim != 5 or xyz_local.shape[-1] != 3:
        raise ValueError("xyz_local must have shape [B,T,H,W,3]")
    batch, frames, height, width, _ = xyz_local.shape
    if extrinsics_w2c.shape[:2] != (batch, frames):
        raise ValueError("extrinsics leading dimensions must match xyz_local")
    w2c = _as_homogeneous(extrinsics_w2c)
    c2w = torch.linalg.inv(w2c)
    ones = torch.ones_like(xyz_local[..., :1])
    local_h = torch.cat((xyz_local, ones), dim=-1)
    global_h = torch.einsum("btij,bthwj->bthwi", c2w, local_h)
    xyz_global = global_h[..., :3] / global_h[..., 3:].clamp_min(1.0e-8)
    if tuple(xyz_global.shape) != (batch, frames, height, width, 3):
        raise RuntimeError("Unexpected global point shape")
    return xyz_global


__all__ = [
    "WORLD_TO_CAMERA",
    "depth_intrinsics_to_local_points",
    "local_to_global_points",
]
