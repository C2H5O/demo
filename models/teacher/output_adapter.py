"""Normalize VGGT-Omega depth, pose, confidence, and point-map outputs."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from utils.geometry import normalize_teacher_confidence, unproject_depth_to_points


def adapt_teacher_outputs(
    predictions: Dict[str, torch.Tensor],
    image_shape: Tuple[int, int],
    min_depth: float = 0.1,
    max_depth: float = 150.0,
) -> Dict[str, torch.Tensor]:
    required = ("pose_enc", "depth", "depth_conf")
    missing = [name for name in required if name not in predictions]
    if missing:
        raise KeyError("VGGT-Omega outputs are missing {}".format(missing))
    pose_module = importlib.import_module("vggt_omega.utils.pose_enc")
    extrinsics, intrinsics = pose_module.encoding_to_camera(
        predictions["pose_enc"], image_shape
    )
    depth = predictions["depth"]
    raw_confidence = predictions["depth_conf"]
    depth_scalar = depth[..., 0] if depth.ndim == 5 and depth.shape[-1] == 1 else depth
    confidence_scalar = (
        raw_confidence[..., 0]
        if raw_confidence.ndim == 5 and raw_confidence.shape[-1] == 1
        else raw_confidence
    )
    valid = (
        torch.isfinite(depth_scalar)
        & torch.isfinite(confidence_scalar)
        & (depth_scalar >= min_depth)
        & (depth_scalar <= max_depth)
    )
    local_points, global_points = unproject_depth_to_points(
        depth, intrinsics, extrinsics
    )
    valid = (
        valid
        & torch.isfinite(local_points).all(dim=-1)
        & torch.isfinite(global_points).all(dim=-1)
    )
    confidence = normalize_teacher_confidence(raw_confidence, valid)
    return {
        "depth": depth_scalar,
        "xyz_local": local_points,
        "xyz_global": global_points,
        "conf_local": confidence,
        "conf_global": confidence,
        "valid_mask": valid,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "pose_enc": predictions["pose_enc"],
    }


def adapt_teacher_depth_outputs(
    predictions: Dict[str, torch.Tensor], image_shape: Tuple[int, int]
) -> Dict[str, torch.Tensor]:
    """Decode only the dense depth/cameras needed by online VDA evaluation."""
    required = ("pose_enc", "depth")
    missing = [name for name in required if name not in predictions]
    if missing:
        raise KeyError("VGGT-Omega outputs are missing {}".format(missing))
    pose_module = importlib.import_module("vggt_omega.utils.pose_enc")
    extrinsics, intrinsics = pose_module.encoding_to_camera(
        predictions["pose_enc"].float(), image_shape
    )
    depth = predictions["depth"].float()
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 4:
        raise ValueError("VGGT-Omega depth must be [B,F,H,W] or [B,F,H,W,1]")
    return {
        "depth": depth,
        "intrinsics": intrinsics.float(),
        "extrinsics": extrinsics.float(),
    }


def adapt_teacher_distillation_outputs(
    predictions: Dict[str, torch.Tensor],
    image_shape: Tuple[int, int],
    supervision_shape: Tuple[int, int],
    min_depth: float = 0.1,
    max_depth: float = 150.0,
) -> Dict[str, torch.Tensor]:
    """Keep only online supervision used by the unchanged distillation loss."""
    required = ("pose_enc", "depth", "depth_conf")
    missing = [name for name in required if name not in predictions]
    if missing:
        raise KeyError("VGGT-Omega outputs are missing {}".format(missing))
    input_height, input_width = (int(value) for value in image_shape)
    output_height, output_width = (int(value) for value in supervision_shape)
    if min(input_height, input_width, output_height, output_width) <= 0:
        raise ValueError("Teacher input and supervision dimensions must be positive")

    pose_module = importlib.import_module("vggt_omega.utils.pose_enc")
    extrinsics, intrinsics = pose_module.encoding_to_camera(
        predictions["pose_enc"].float(), (input_height, input_width)
    )
    depth = predictions["depth"].float()
    raw_confidence = predictions["depth_conf"].float()
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if raw_confidence.ndim == 5 and raw_confidence.shape[-1] == 1:
        raw_confidence = raw_confidence[..., 0]
    if depth.ndim != 4 or tuple(raw_confidence.shape) != tuple(depth.shape):
        raise ValueError("VGGT-Omega depth/confidence must have matching [B,F,H,W] shapes")
    if tuple(depth.shape[-2:]) != (input_height, input_width):
        raise ValueError(
            "VGGT-Omega dense output shape {} does not match Teacher input {}".format(
                tuple(depth.shape[-2:]), (input_height, input_width)
            )
        )

    valid = (
        torch.isfinite(depth)
        & torch.isfinite(raw_confidence)
        & (depth >= float(min_depth))
        & (depth <= float(max_depth))
    )
    confidence = normalize_teacher_confidence(raw_confidence, valid)
    batch, frames = depth.shape[:2]
    flat_valid = valid.reshape(batch * frames, 1, input_height, input_width).float()

    def resize_valid_aware(value: torch.Tensor) -> torch.Tensor:
        flat = value.reshape(batch * frames, 1, input_height, input_width).float()
        weight = F.interpolate(
            flat_valid,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
        )
        sampled = F.interpolate(
            flat * flat_valid,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
        ) / weight.clamp_min(1.0e-6)
        sampled = torch.where(weight > 1.0e-6, sampled, torch.zeros_like(sampled))
        return sampled.reshape(batch, frames, output_height, output_width)

    if (input_height, input_width) == (output_height, output_width):
        output_valid = valid
        output_depth = torch.where(valid, depth, torch.zeros_like(depth))
        output_confidence = torch.where(
            valid, confidence, torch.zeros_like(confidence)
        )
    else:
        output_valid = F.interpolate(
            flat_valid,
            size=(output_height, output_width),
            mode="nearest",
        ).reshape(batch, frames, output_height, output_width).gt(0.5)
        output_depth = torch.where(
            output_valid, resize_valid_aware(depth), torch.zeros_like(output_valid, dtype=torch.float32)
        )
        output_confidence = torch.where(
            output_valid,
            resize_valid_aware(confidence),
            torch.zeros_like(output_valid, dtype=torch.float32),
        )

    intrinsics = intrinsics.float().clone()
    sx = float(output_width) / float(input_width)
    sy = float(output_height) / float(input_height)
    intrinsics[..., 0, 0] *= sx
    intrinsics[..., 1, 1] *= sy
    intrinsics[..., 0, 2] *= sx
    intrinsics[..., 1, 2] *= sy
    return {
        "depth": output_depth.detach(),
        "confidence": output_confidence.detach(),
        "valid_mask": output_valid.detach(),
        "intrinsics": intrinsics.detach(),
        "extrinsics": extrinsics.float().detach(),
    }


__all__ = [
    "adapt_teacher_depth_outputs",
    "adapt_teacher_distillation_outputs",
    "adapt_teacher_outputs",
]
