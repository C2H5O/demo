from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import models.student.da3_small_student as da3_module
from models.student.da3_small_student import DA3SmallConfig, DA3SmallStudent
from utils.da3_geometry import (
    depth_intrinsics_to_local_points,
    global_to_camera_points,
    local_to_global_points,
)


def test_da3_small_config_is_fixed_to_448x560_and_no_ray() -> None:
    config = DA3SmallConfig()
    config.validate()
    assert (config.image_height // config.patch_size, config.image_width // config.patch_size) == (32, 40)
    with pytest.raises(ValueError, match="ray"):
        DA3SmallConfig(use_ray=True).validate()
    with pytest.raises(ValueError, match="448x560"):
        DA3SmallConfig(image_height=406).validate()


def test_depth_camera_geometry_preserves_z_and_round_trips() -> None:
    depth = torch.full((1, 2, 3, 4), 2.0, requires_grad=True)
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    intrinsics[..., 0, 0] = 2.0
    intrinsics[..., 1, 1] = 2.0
    intrinsics.requires_grad_()
    extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)[..., :3, :]
    extrinsics = extrinsics.clone().requires_grad_()
    local = depth_intrinsics_to_local_points(depth, intrinsics)
    global_points = local_to_global_points(local, extrinsics)
    round_trip = global_to_camera_points(global_points, extrinsics)
    torch.testing.assert_close(local[..., 2], depth)
    torch.testing.assert_close(round_trip, local)
    (global_points.square().mean()).backward()
    assert depth.grad is not None
    assert intrinsics.grad is not None
    assert extrinsics.grad is not None


def test_geometry_rejects_nonpositive_depth() -> None:
    depth = torch.zeros(1, 1, 2, 2)
    intrinsics = torch.eye(3).view(1, 1, 3, 3)
    with pytest.raises(ValueError, match="strictly positive"):
        depth_intrinsics_to_local_points(depth, intrinsics)


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, images, **_kwargs):
        batch, frames = images.shape[:2]
        tokens = self.scale * torch.ones(batch, frames, 1, 1, device=images.device)
        camera = self.scale * torch.ones(batch, frames, 1, device=images.device)
        return [(tokens, camera) for _ in range(4)], []


class _FakeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Parameter(torch.ones(()))
        self.scratch = nn.Module()
        for name in (
            "refinenet1_aux", "refinenet2_aux", "refinenet3_aux", "refinenet4_aux",
            "output_conv1_aux", "output_conv2_aux",
        ):
            setattr(self.scratch, name, nn.Sequential(nn.Linear(1, 1)))


class _FakeCameraDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, features):
        batch, frames = features.shape[:2]
        pose = features.new_zeros(batch, frames, 9)
        pose[..., 6] = self.scale
        pose[..., 7:] = 1.0
        return pose


class _FakeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeBackbone()
        self.head = _FakeHead()
        self.cam_enc = nn.Linear(1, 1)
        self.cam_dec = _FakeCameraDecoder()


def test_student_depth_only_contract_never_executes_ray_modules(monkeypatch) -> None:
    def fake_pose_transform(pose, image_size):
        batch, frames = pose.shape[:2]
        c2w = torch.eye(4, device=pose.device).view(1, 1, 4, 4).repeat(batch, frames, 1, 1)[..., :3, :]
        height, width = image_size
        intrinsics = torch.eye(3, device=pose.device).view(1, 1, 3, 3).repeat(batch, frames, 1, 1)
        intrinsics[..., 0, 0] = width
        intrinsics[..., 1, 1] = height
        intrinsics[..., 0, 2] = width / 2
        intrinsics[..., 1, 2] = height / 2
        return c2w, intrinsics

    monkeypatch.setattr(
        da3_module, "_require_official_da3", lambda: (None, None, fake_pose_transform)
    )
    model = DA3SmallStudent(DA3SmallConfig(), network=_FakeNetwork())
    monkeypatch.setattr(
        model,
        "_forward_depth_main",
        lambda feats, height, width: (
            torch.ones(feats[0][0].shape[:2] + (height, width)),
            torch.ones(feats[0][0].shape[:2] + (height, width)),
        ),
    )
    output = model(torch.zeros(1, 16, 3, 448, 560))
    assert set(output) == {
        "depth", "depth_conf", "intrinsics", "extrinsics",
        "xyz_local", "xyz_global", "pts3d_local",
    }
    assert output["depth"].shape == (1, 16, 448, 560)
    assert model._ray_forward_count == 0
