from __future__ import annotations

import torch

from losses.teacher_self_supervised import TeacherSelfSupervisedLoss
from losses.teacher_self_supervised.geometry_warp import (
    relative_camera_transform,
    warp_source_to_target,
)
from utils.geometry import unproject_depth_to_points


def _camera(batch: int, frames: int, height: int, width: int):
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(batch, frames, 1, 1)
    intrinsics[..., 0, 0] = 20.0
    intrinsics[..., 1, 1] = 20.0
    intrinsics[..., 0, 2] = (width - 1) / 2
    intrinsics[..., 1, 2] = (height - 1) / 2
    extrinsics = torch.zeros(batch, frames, 3, 4)
    extrinsics[..., :3, :3] = torch.eye(3)
    return intrinsics, extrinsics


def test_identity_camera_warp() -> None:
    batch, height, width = 1, 12, 16
    intrinsics, extrinsics = _camera(batch, 2, height, width)
    transform = relative_camera_transform(extrinsics[:, 0], extrinsics[:, 1])
    image = torch.rand(batch, 3, height, width)
    depth = torch.ones(batch, 1, height, width)
    result = warp_source_to_target(
        image,
        depth,
        depth,
        intrinsics[:, 0],
        intrinsics[:, 1],
        transform,
    )
    torch.testing.assert_close(result["warped_image"], image, atol=1e-5, rtol=1e-5)
    assert result["valid_mask"].all()


def test_complete_teacher_loss_backward_is_finite() -> None:
    torch.manual_seed(0)
    batch, frames, height, width = 1, 3, 12, 16
    intrinsics, extrinsics = _camera(batch, frames, height, width)
    depth = torch.ones(batch, frames, height, width, requires_grad=True)
    local, global_points = unproject_depth_to_points(depth, intrinsics, extrinsics)
    images = torch.rand(batch, frames, 3, height, width)
    valid = torch.ones(batch, frames, height, width, dtype=torch.bool)
    confidence = torch.full_like(depth, 0.5)
    outputs = {
        "depth": depth,
        "xyz_local": local,
        "xyz_global": global_points,
        "valid_mask": valid,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "conf_local": confidence,
        "conf_global": confidence,
        "pose_enc": torch.zeros(batch, frames, 9),
    }
    masks = torch.zeros(batch, frames, 1, height, width)
    masks[:, :, :, 4:6, 5:7] = 1
    loss_function = TeacherSelfSupervisedLoss(
        {
            "auto_mask": False,
            "light_alignment": True,
            "temporal_offsets": [-1, 1],
        }
    )
    loss, logs = loss_function(
        outputs,
        images,
        intrinsics,
        masks,
        images,
        outputs,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()
    assert logs["temporal_pair_count"] == 4.0
    assert set(
        (
            "loss_photometric",
            "loss_geometry",
            "loss_highlight",
            "loss_smoothness",
            "loss_inpaint_consistency",
        )
    ).issubset(logs)
