"""Original/inpainted teacher prediction consistency."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from losses.teacher_self_supervised.geometry_warp import masked_mean


def original_inpainted_consistency(
    original: Dict[str, torch.Tensor],
    inpainted: Dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    highlight_mask: torch.Tensor,
    include_highlights: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    valid = (
        valid_mask.bool()
        & inpainted["valid_mask"].bool()
        & torch.isfinite(original["depth"])
        & torch.isfinite(inpainted["depth"])
    )
    if not include_highlights:
        valid = valid & ~highlight_mask[:, :, 0].bool()
    original_depth = original["depth"].float()
    target_depth = inpainted["depth"].detach().float()
    valid_float = valid.float()
    original_scale = (original_depth * valid_float).sum(
        dim=(2, 3), keepdim=True
    ) / valid.sum(
        dim=(2, 3), keepdim=True
    ).clamp_min(1)
    target_scale = (target_depth * valid_float).sum(
        dim=(2, 3), keepdim=True
    ) / valid.sum(
        dim=(2, 3), keepdim=True
    ).clamp_min(1)
    depth_map = (
        original_depth / original_scale.clamp_min(1e-7)
        - target_depth / target_scale.clamp_min(1e-7)
    ).abs()
    depth_loss = masked_mean(depth_map, valid)

    original_rotation = original["extrinsics"][..., :3, :3]
    target_rotation = inpainted["extrinsics"][..., :3, :3].detach()
    relative = original_rotation @ target_rotation.transpose(-1, -2)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rotation_cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    rotation_loss = (1.0 - rotation_cosine).mean()
    original_translation = original["extrinsics"][..., :3, 3]
    target_translation = inpainted["extrinsics"][..., :3, 3].detach()
    translation_loss = F.smooth_l1_loss(
        original_translation, target_translation
    )
    confidence_loss = 0.5 * (
        F.smooth_l1_loss(
            original["conf_local"].float(), inpainted["conf_local"].detach().float()
        )
        + F.smooth_l1_loss(
            original["conf_global"].float(), inpainted["conf_global"].detach().float()
        )
    )
    pose_loss = rotation_loss + translation_loss
    return depth_loss + pose_loss + confidence_loss, {
        "inpaint_depth": depth_loss,
        "inpaint_pose": pose_loss,
        "inpaint_confidence": confidence_loss,
    }
