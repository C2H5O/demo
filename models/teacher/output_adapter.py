"""Normalize VGGT-Omega depth, pose, confidence, and point-map outputs."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

import torch

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
