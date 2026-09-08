"""Read-only diagnostics for the existing highlight and smoothness objectives."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from losses.direct_teacher_distillation_loss import surface_normals


def loss_share_logs(logs):
    """Call after attention is added; shares refer to the final training total."""
    total = float(logs["loss/total"])
    values = {name: float(logs.get("loss/" + name + "_weighted", 0.0))
              for name in ("depth", "camera", "highlight", "smooth", "attention")}
    valid = math.isfinite(total) and total > 0 and all(math.isfinite(v) for v in values.values())
    result = {"stats/loss_share_valid": float(valid)}
    for name, value in values.items():
        result["stats/loss_share_" + name] = value / total if valid else 0.0
    result["stats/loss_sum_residual"] = total - sum(values.values())
    return result


@torch.no_grad()
def regularizer_diagnostics(points, highlight_mask, clean_images, eps=1e-6):
    """Mask coverage and normal conditioning, with no effect on loss/backward.

    Normal-valid pixels here match the existing loss's finite five-point stencil.
    A short normal indicates its area vector hit surface_normals' epsilon floor;
    it is reported, not silently removed from the objective.
    """
    points = points.detach().float()
    batch, frames, height, width, _ = points.shape
    finite = torch.isfinite(points).all(-1) & (points[..., 2] > eps)
    stencil = torch.zeros_like(finite)
    stencil[..., 1:-1, 1:-1] = (finite[..., 1:-1, 1:-1] & finite[..., 1:-1, :-2]
        & finite[..., 1:-1, 2:] & finite[..., :-2, 1:-1] & finite[..., 2:, 1:-1])
    highlight = highlight_mask[:, :, 0].bool()
    selected = highlight & stencil
    safe = torch.nan_to_num(points).reshape(batch * frames, height, width, 3).permute(0, 3, 1, 2)
    normals = surface_normals(safe)
    norm = torch.linalg.vector_norm(normals, dim=1).reshape(batch, frames, height, width)
    cosine = (normals * F.normalize(-safe, dim=1, eps=eps)).sum(1).reshape(batch, frames, height, width)
    # Only unit-length normals support an interpretable geometric angle.
    unit = selected & (norm > 0.999) & torch.isfinite(cosine)
    depth_valid = torch.isfinite(points[..., 2]) & (points[..., 2] > eps)
    valid_x = depth_valid[..., 1:] & depth_valid[..., :-1] & ~highlight[..., 1:] & ~highlight[..., :-1]
    valid_y = depth_valid[..., 1:, :] & depth_valid[..., :-1, :] & ~highlight[..., 1:, :] & ~highlight[..., :-1, :]
    count = selected.sum()
    def masked_mean(value, mask):
        return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)
    angles = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
    result = {
        "stats/highlight_mask_ratio": highlight.float().mean(),
        "stats/highlight_valid_pixel_count": count,
        "stats/highlight_valid_mask_ratio": selected.float().mean(),
        "stats/highlight_retained_mask_fraction": count / highlight.sum().clamp_min(1),
        "stats/highlight_empty": (count == 0).float(),
        "stats/highlight_frames_with_valid_fraction": selected.flatten(2).any(2).float().mean(),
        "stats/highlight_normal_short_fraction": masked_mean((norm < 0.999).float(), selected),
        "stats/highlight_unit_normal_pixel_count": unit.sum(),
        "stats/highlight_normal_view_angle_degrees": masked_mean(angles, unit),
        "stats/highlight_normal_view_angle_valid": unit.any().float(),
        "stats/smooth_valid_pair_count": valid_x.sum() + valid_y.sum(),
        "stats/smooth_valid_pair_fraction": (valid_x.sum() + valid_y.sum()) / max(1, valid_x.numel() + valid_y.numel()),
        "stats/smooth_empty": ((valid_x.sum() + valid_y.sum()) == 0).float(),
        "stats/clean_image_min": clean_images.detach().min(),
        "stats/clean_image_max": clean_images.detach().max(),
    }
    return {name: float(value.cpu()) for name, value in result.items()}
