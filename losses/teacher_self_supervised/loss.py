"""Complete temporal self-supervised objective for VGGT-Omega LoRA adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.teacher_self_supervised.consistency_loss import (
    original_inpainted_consistency,
)
from losses.teacher_self_supervised.geometry_warp import (
    masked_mean,
    relative_camera_transform,
    warp_source_to_target,
)
from losses.teacher_self_supervised.highlight_loss import (
    edge_aware_depth_smoothness,
    highlight_surface_loss,
)
from losses.teacher_self_supervised.photometric import (
    SSIM,
    light_align_warped_source,
    photometric_error,
)


@dataclass(frozen=True)
class TeacherSelfSupervisedLossConfig:
    photometric_weight: float = 1.0
    geometry_weight: float = 0.1
    highlight_weight: float = 0.01
    smoothness_weight: float = 0.01
    inpaint_consistency_weight: float = 0.1
    ssim_weight: float = 0.85
    auto_mask: bool = True
    dynamic_weighting: bool = True
    minimum_reprojection: bool = True
    mask_highlights: bool = True
    light_alignment: bool = True
    light_mu: float = 3.069096
    light_gamma: float = 2.2
    light_minimum_cosine: float = 0.4
    temporal_offsets: Tuple[int, ...] = (-1, 1)
    consistency_include_highlights: bool = True


def _resize_temporal(
    tensor: torch.Tensor,
    size: Tuple[int, int],
    mode: str,
) -> torch.Tensor:
    if tensor.shape[-2:] == size:
        return tensor
    batch, frames, channels = tensor.shape[:3]
    flattened = tensor.reshape(batch * frames, channels, *tensor.shape[-2:])
    if mode == "nearest":
        resized = F.interpolate(flattened, size=size, mode=mode)
    else:
        resized = F.interpolate(
            flattened, size=size, mode=mode, align_corners=False
        )
    return resized.reshape(batch, frames, channels, *size)


class TeacherSelfSupervisedLoss(nn.Module):
    def __init__(
        self,
        config: TeacherSelfSupervisedLossConfig | Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = TeacherSelfSupervisedLossConfig()
        elif isinstance(config, dict):
            value = dict(config)
            if "temporal_offsets" in value:
                value["temporal_offsets"] = tuple(int(x) for x in value["temporal_offsets"])
            config = TeacherSelfSupervisedLossConfig(**value)
        self.config = config
        self.ssim = SSIM()
        if not self.config.temporal_offsets:
            raise ValueError("At least one temporal offset is required")

    def _pair_maps(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        images: torch.Tensor,
        highlight_masks: torch.Tensor,
        target_index: int,
        source_index: int,
    ) -> Dict[str, torch.Tensor]:
        target_depth = teacher_outputs["depth"][:, target_index].unsqueeze(1)
        source_depth = teacher_outputs["depth"][:, source_index].unsqueeze(1)
        target_extrinsics = teacher_outputs["extrinsics"][:, target_index]
        source_extrinsics = teacher_outputs["extrinsics"][:, source_index]
        transform = relative_camera_transform(target_extrinsics, source_extrinsics)
        warped = warp_source_to_target(
            source_image=images[:, source_index],
            target_depth=target_depth,
            source_depth=source_depth,
            target_intrinsics=teacher_outputs["intrinsics"][:, target_index],
            source_intrinsics=teacher_outputs["intrinsics"][:, source_index],
            target_to_source=transform,
            source_mask=highlight_masks[:, source_index],
        )
        target_highlight = highlight_masks[:, target_index].bool()
        source_highlight = warped["warped_source_mask"].bool()
        excluded = target_highlight | source_highlight
        prediction = warped["warped_image"]
        light = None
        if self.config.light_alignment:
            light = light_align_warped_source(
                prediction,
                images[:, target_index],
                warped["target_points"],
                warped["source_points"],
                transform,
                warped["valid_mask"],
                excluded,
                self.config.light_mu,
                self.config.light_gamma,
                self.config.light_minimum_cosine,
            )
            prediction = light["refined_image"]

        geometry = (
            warped["computed_source_depth"] - warped["sampled_source_depth"]
        ).abs() / (
            warped["computed_source_depth"].abs()
            + warped["sampled_source_depth"].abs()
            + 1e-7
        )
        photo = photometric_error(
            images[:, target_index], prediction, self.ssim, self.config.ssim_weight
        )
        valid = (
            warped["valid_mask"]
            & teacher_outputs["valid_mask"][:, target_index].unsqueeze(1)
        )
        if self.config.auto_mask:
            identity = (
                images[:, target_index] - images[:, source_index]
            ).abs().mean(dim=1, keepdim=True)
            valid = valid & (photo.detach() < identity.detach())
        if self.config.mask_highlights:
            valid = valid & ~excluded
        if self.config.dynamic_weighting:
            photo = photo * (1.0 - geometry.detach().clamp(0.0, 1.0))
        result = {
            "photometric": photo,
            "geometry": geometry,
            "valid": valid,
            "warped": prediction,
        }
        if light is not None:
            result["light_correction"] = light["correction"]
        return result

    def _temporal_losses(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        images: torch.Tensor,
        highlight_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        frames = images.shape[1]
        photo_terms, geometry_terms = [], []
        pair_count = 0
        for target_index in range(frames):
            pairs = []
            for offset in self.config.temporal_offsets:
                source_index = target_index + offset
                if 0 <= source_index < frames:
                    pairs.append(
                        self._pair_maps(
                            teacher_outputs,
                            images,
                            highlight_masks,
                            target_index,
                            source_index,
                        )
                    )
                    pair_count += 1
            if not pairs:
                continue
            if self.config.minimum_reprojection:
                photo_stack = torch.cat(
                    [pair["photometric"] for pair in pairs], dim=1
                )
                geometry_stack = torch.cat(
                    [pair["geometry"] for pair in pairs], dim=1
                )
                valid_stack = torch.cat([pair["valid"] for pair in pairs], dim=1)
                candidates = torch.where(
                    valid_stack,
                    photo_stack,
                    torch.full_like(photo_stack, torch.inf),
                )
                selected_photo, indices = candidates.min(dim=1, keepdim=True)
                selected_geometry = torch.gather(
                    geometry_stack, 1, indices
                )
                selected_valid = valid_stack.any(dim=1, keepdim=True)
                selected_photo = torch.where(
                    selected_valid, selected_photo, torch.zeros_like(selected_photo)
                )
                selected_geometry = torch.where(
                    selected_valid,
                    selected_geometry,
                    torch.zeros_like(selected_geometry),
                )
                photo_terms.append(masked_mean(selected_photo, selected_valid))
                geometry_terms.append(
                    masked_mean(selected_geometry, selected_valid)
                )
            else:
                photo_terms.extend(
                    masked_mean(pair["photometric"], pair["valid"]) for pair in pairs
                )
                geometry_terms.extend(
                    masked_mean(pair["geometry"], pair["valid"]) for pair in pairs
                )
        if not photo_terms:
            raise ValueError("The clip has no valid temporal source/target pairs")
        return (
            torch.stack(photo_terms).mean(),
            torch.stack(geometry_terms).mean(),
            pair_count,
        )

    def forward(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        images: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
        highlight_masks: torch.Tensor | None = None,
        inpainted_images: torch.Tensor | None = None,
        inpainted_teacher_outputs: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del intrinsics
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,T,3,H,W]")
        depth = teacher_outputs["depth"]
        if depth.ndim != 4:
            raise ValueError("teacher depth must have shape [B,T,H,W]")
        size = tuple(depth.shape[-2:])
        images = _resize_temporal(images.float(), size, "bilinear").clamp(0.0, 1.0)
        if highlight_masks is None:
            highlight_masks = images.new_zeros(
                images.shape[0], images.shape[1], 1, *size
            )
        else:
            highlight_masks = _resize_temporal(
                highlight_masks.float(), size, "nearest"
            )
        if inpainted_images is None:
            inpainted_images = images
        else:
            inpainted_images = _resize_temporal(
                inpainted_images.float(), size, "bilinear"
            ).clamp(0.0, 1.0)

        photometric, geometry, pair_count = self._temporal_losses(
            teacher_outputs, images, highlight_masks
        )
        highlight_terms, smoothness_terms = [], []
        for frame_index in range(images.shape[1]):
            points = teacher_outputs["xyz_local"][:, frame_index].permute(
                0, 3, 1, 2
            )
            valid = teacher_outputs["valid_mask"][:, frame_index].unsqueeze(1)
            highlight_terms.append(
                highlight_surface_loss(
                    points, highlight_masks[:, frame_index], valid
                )
            )
            smoothness_terms.append(
                edge_aware_depth_smoothness(
                    depth[:, frame_index].unsqueeze(1),
                    inpainted_images[:, frame_index],
                    valid,
                )
            )
        highlight = torch.stack(highlight_terms).mean()
        smoothness = torch.stack(smoothness_terms).mean()

        consistency = depth.sum() * 0.0
        consistency_parts: Dict[str, torch.Tensor] = {}
        if inpainted_teacher_outputs is not None:
            consistency, consistency_parts = original_inpainted_consistency(
                teacher_outputs,
                inpainted_teacher_outputs,
                teacher_outputs["valid_mask"],
                highlight_masks,
                self.config.consistency_include_highlights,
            )
        total = (
            self.config.photometric_weight * photometric
            + self.config.geometry_weight * geometry
            + self.config.highlight_weight * highlight
            + self.config.smoothness_weight * smoothness
            + self.config.inpaint_consistency_weight * consistency
        )
        tensors = {
            "loss_total": total,
            "loss_photometric": photometric,
            "loss_geometry": geometry,
            "loss_highlight": highlight,
            "loss_smoothness": smoothness,
            "loss_inpaint_consistency": consistency,
            "highlight_fraction": highlight_masks.float().mean(),
            "valid_depth_fraction": teacher_outputs["valid_mask"].float().mean(),
            **consistency_parts,
        }
        logs = {name: float(value.detach().cpu()) for name, value in tensors.items()}
        logs["temporal_pair_count"] = float(pair_count)
        return total, logs
