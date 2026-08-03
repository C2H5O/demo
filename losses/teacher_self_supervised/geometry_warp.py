"""Differentiable camera geometry for temporal self-supervision."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def relative_camera_transform(
    target_extrinsics: torch.Tensor,
    source_extrinsics: torch.Tensor,
) -> torch.Tensor:
    """Return target-camera to source-camera transforms.

    VGGT-Omega extrinsics follow ``X_camera = R X_world + t``.
    """
    if target_extrinsics.shape != source_extrinsics.shape:
        raise ValueError("Target/source extrinsics must have identical shapes")
    if target_extrinsics.ndim != 3 or target_extrinsics.shape[-2:] != (3, 4):
        raise ValueError("Extrinsics must have shape [B,3,4]")
    target_rotation = target_extrinsics[:, :3, :3]
    target_translation = target_extrinsics[:, :3, 3:]
    source_rotation = source_extrinsics[:, :3, :3]
    source_translation = source_extrinsics[:, :3, 3:]
    rotation = source_rotation @ target_rotation.transpose(-1, -2)
    translation = source_translation - rotation @ target_translation
    return torch.cat((rotation, translation), dim=-1)


def _pixel_grid(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack((columns, rows, torch.ones_like(columns)), dim=0)
    return pixels.reshape(1, 3, -1).expand(batch, -1, -1)


def warp_source_to_target(
    source_image: torch.Tensor,
    target_depth: torch.Tensor,
    source_depth: torch.Tensor,
    target_intrinsics: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_to_source: torch.Tensor,
    source_mask: torch.Tensor | None = None,
    padding_mode: str = "zeros",
) -> Dict[str, torch.Tensor]:
    """Sample a source view into the target frame using target depth."""
    if target_depth.ndim == 3:
        target_depth = target_depth.unsqueeze(1)
    if source_depth.ndim == 3:
        source_depth = source_depth.unsqueeze(1)
    batch, _, height, width = target_depth.shape
    if source_image.shape[0] != batch or source_image.shape[-2:] != (height, width):
        raise ValueError("Source image and target depth spatial shapes must match")

    pixels = _pixel_grid(
        batch, height, width, target_depth.device, target_depth.dtype
    )
    rays = torch.linalg.solve(target_intrinsics.to(target_depth.dtype), pixels)
    target_points = rays * target_depth.reshape(batch, 1, -1)
    rotation = target_to_source[:, :3, :3].to(target_depth.dtype)
    translation = target_to_source[:, :3, 3:].to(target_depth.dtype)
    source_points = rotation @ target_points + translation
    projected = source_intrinsics.to(target_depth.dtype) @ source_points
    projected_z = projected[:, 2:3]
    x = projected[:, 0:1] / projected_z.clamp_min(1e-7)
    y = projected[:, 1:2] / projected_z.clamp_min(1e-7)

    valid = (
        torch.isfinite(x)
        & torch.isfinite(y)
        & torch.isfinite(projected_z)
        & (projected_z > 1e-7)
        & (x >= 0.0)
        & (x <= width - 1.0)
        & (y >= 0.0)
        & (y <= height - 1.0)
        & torch.isfinite(target_depth.reshape(batch, 1, -1))
        & (target_depth.reshape(batch, 1, -1) > 1e-7)
    )
    normalized_x = 2.0 * x / max(width - 1, 1) - 1.0
    normalized_y = 2.0 * y / max(height - 1, 1) - 1.0
    grid = torch.cat((normalized_x, normalized_y), dim=1)
    grid = grid.transpose(1, 2).reshape(batch, height, width, 2)

    warped_image = F.grid_sample(
        source_image,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    sampled_depth = F.grid_sample(
        source_depth,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    valid = valid.reshape(batch, 1, height, width)
    valid = (
        valid
        & torch.isfinite(sampled_depth)
        & (sampled_depth > 1e-7)
    )
    result = {
        "warped_image": warped_image,
        "sampled_source_depth": sampled_depth,
        "computed_source_depth": projected_z.reshape(batch, 1, height, width),
        "valid_mask": valid,
        "grid": grid,
        "target_points": target_points.reshape(batch, 3, height, width),
        "source_points": source_points.reshape(batch, 3, height, width),
    }
    if source_mask is not None:
        result["warped_source_mask"] = F.grid_sample(
            source_mask.float(),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
    return result


def surface_normals(points: torch.Tensor) -> torch.Tensor:
    """Compute camera-oriented normals for ``[B,3,H,W]`` point maps."""
    if points.ndim != 4 or points.shape[1] != 3:
        raise ValueError("points must have shape [B,3,H,W]")
    horizontal = points[:, :, 1:-1, 2:] - points[:, :, 1:-1, :-2]
    vertical = points[:, :, 2:, 1:-1] - points[:, :, :-2, 1:-1]
    normals = F.normalize(
        torch.cross(horizontal, vertical, dim=1), dim=1, eps=1e-6
    )
    normals = F.pad(normals, (1, 1, 1, 1))
    view = F.normalize(-points, dim=1, eps=1e-6)
    flip = (normals * view).sum(dim=1, keepdim=True) < 0
    return torch.where(flip, -normals, normals)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype).expand_as(values)
    count = mask.sum()
    return (values * mask).sum() / count.clamp_min(1.0)
