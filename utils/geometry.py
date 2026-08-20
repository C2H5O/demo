"""Camera unprojection utilities used by the SCARED teacher exporter."""

from __future__ import annotations

from typing import Tuple

import torch


CAMERA_FROM_WORLD_CONVENTION = "camera-from-world: X_cam = R @ X_world + t"


def _validate_extrinsics(extrinsics: torch.Tensor) -> None:
    if extrinsics.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError(
            "extrinsics must end in [3,4] or [4,4], got {}".format(
                tuple(extrinsics.shape)
            )
        )


def world_to_camera(
    points_world: torch.Tensor, camera_from_world: torch.Tensor
) -> torch.Tensor:
    """Transform ``[...,3]`` world points with ``X_cam = R X_world + t``."""
    _validate_extrinsics(camera_from_world)
    rotation = camera_from_world[..., :3, :3].to(points_world)
    translation = camera_from_world[..., :3, 3].to(points_world)
    for _ in range(points_world.ndim - translation.ndim):
        rotation = rotation.unsqueeze(-3)
        translation = translation.unsqueeze(-2)
    return (
        torch.matmul(points_world.unsqueeze(-2), rotation.transpose(-1, -2))
        .squeeze(-2)
        .add(translation)
    )


def camera_to_world(
    points_camera: torch.Tensor, camera_from_world: torch.Tensor
) -> torch.Tensor:
    """Invert a camera-from-world pose for arbitrary point-map leading dims."""
    _validate_extrinsics(camera_from_world)
    rotation = camera_from_world[..., :3, :3].to(points_camera)
    translation = camera_from_world[..., :3, 3].to(points_camera)
    for _ in range(points_camera.ndim - translation.ndim):
        rotation = rotation.unsqueeze(-3)
        translation = translation.unsqueeze(-2)
    return torch.matmul((points_camera - translation).unsqueeze(-2), rotation).squeeze(-2)


def camera_to_camera(
    points_source: torch.Tensor,
    source_camera_from_world: torch.Tensor,
    target_camera_from_world: torch.Tensor,
) -> torch.Tensor:
    """Move source-camera points into the target camera via world coordinates."""
    return world_to_camera(
        camera_to_world(points_source, source_camera_from_world),
        target_camera_from_world,
    )


def unproject_depth_to_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return camera-local and world point maps from camera-from-world poses."""
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 4:
        raise ValueError("depth must have shape [B,T,H,W], got {}".format(tuple(depth.shape)))
    batch, frames, height, width = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
    pixels = pixels.view(1, 1, height, width, 3).expand(batch, frames, -1, -1, -1)
    rays = torch.einsum("btij,bthwj->bthwi", torch.linalg.inv(intrinsics).to(depth.dtype), pixels)
    local_points = rays * depth.unsqueeze(-1)

    rotation = extrinsics[..., :3, :3].to(depth.dtype)
    translation = extrinsics[..., :3, 3].to(depth.dtype)
    # VGGT-Omega extrinsics are camera-from-world: X_cam = R X_world + t.
    global_points = torch.einsum(
        "btji,bthwj->bthwi",
        rotation,
        local_points - translation[:, :, None, None, :],
    )
    return local_points, global_points


def normalize_teacher_confidence(confidence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Map VGGT-Omega's unbounded confidence to robust per-frame [0,1] weights."""
    if confidence.ndim == 5 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    output = torch.zeros_like(confidence, dtype=torch.float32)
    log_confidence = torch.log(confidence.float().clamp_min(1.0))
    for batch_index in range(log_confidence.shape[0]):
        for frame_index in range(log_confidence.shape[1]):
            values = log_confidence[batch_index, frame_index]
            frame_valid = valid[batch_index, frame_index] & torch.isfinite(values)
            finite_values = values[frame_valid]
            if finite_values.numel() == 0:
                continue
            low = torch.quantile(finite_values, 0.05)
            high = torch.quantile(finite_values, 0.95)
            scale = (high - low).clamp_min(1e-6)
            normalized = ((values - low) / scale).clamp(0.0, 1.0)
            output[batch_index, frame_index] = torch.where(frame_valid, normalized, torch.zeros_like(normalized))
    return output
