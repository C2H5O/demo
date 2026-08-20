from __future__ import annotations

import math

import torch

from utils.geometry import camera_to_camera, camera_to_world, world_to_camera


def _pose(rotation: torch.Tensor | None = None, translation=(0.0, 0.0, 0.0)) -> torch.Tensor:
    pose = torch.eye(4)
    if rotation is not None:
        pose[:3, :3] = rotation
    pose[:3, 3] = torch.tensor(translation)
    return pose[:3]


def test_camera_from_world_roundtrip() -> None:
    angle = math.pi / 3
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    pose = _pose(rotation, (1.0, -2.0, 0.5))
    local = torch.randn(2, 4, 5, 3)
    batched_pose = pose.unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(world_to_camera(camera_to_world(local, batched_pose), batched_pose), local, atol=1e-5)


def test_teacher_coordinate_transform() -> None:
    pose_a = _pose(translation=(1.0, 0.0, 0.0))
    pose_b = _pose(translation=(0.0, 2.0, 0.0))
    point_b = torch.tensor([[[3.0, 4.0, 5.0]]])
    expected = world_to_camera(camera_to_world(point_b, pose_b), pose_a)
    assert torch.allclose(camera_to_camera(point_b, pose_b, pose_a), expected)
    assert not torch.allclose(point_b, expected)


def test_identity_camera_case() -> None:
    identity = _pose()
    points = torch.randn(3, 7, 3)
    assert torch.equal(camera_to_camera(points, identity, identity), points)


def test_synthetic_translation_rotation_case() -> None:
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    source = _pose(rotation, (0.0, 0.0, 0.0))
    target = _pose(translation=(2.0, 0.0, 0.0))
    source_point = torch.tensor([1.0, 0.0, 3.0])
    world = camera_to_world(source_point, source)
    assert torch.allclose(world, torch.tensor([0.0, -1.0, 3.0]))
    assert torch.allclose(camera_to_camera(source_point, source, target), torch.tensor([2.0, -1.0, 3.0]))
