from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.student.dune_fast3r_head import (
    DuneFast3RHeadStudent,
    _initialize_camera_facing_output,
)
from trainers.crossclip_projection_trainer import build_crossclip_optimizer


class _FakeDune(nn.Module):
    patch_size = 14
    embed_dim = 384
    chunked_blocks = False

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.anchor = nn.Parameter(torch.ones(()))
        self.requested_layers = None

    def get_intermediate_layers(
        self, images, n, reshape, return_class_token, norm
    ):
        self.requested_layers = list(n)
        assert not reshape and not return_class_token and norm
        tokens = (images.shape[-2] // 14) * (images.shape[-1] // 14)
        base = images.new_ones(images.shape[0], tokens, 384) * self.anchor
        return tuple(base * float(index + 1) for index in range(4))


class _FakeDPTHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, features, image_size):
        height, width = image_size
        batch = features[0].shape[0]
        points = features[0].new_zeros(batch, height, width, 3)
        signal = features[0].mean(dim=(1, 2))
        points[..., 2] = (self.scale + signal)[:, None, None]
        return {"pts3d": points}


def _config(**overrides):
    value = {
        "architecture": "dune_fast3r_dpt",
        "image_height": 28,
        "image_width": 42,
        "encoder_variant": "dune_vitsmall14_448",
        "encoder_layers": [2, 5, 8, 11],
        "encoder_dim": 384,
        "encoder_blocks": 12,
        "patch_size": 14,
        "dune_checkpoint": "unused.pth",
        "freeze_encoder": True,
        "use_fast3r_decoder": False,
        "normalize_mode": "minus_one_one",
    }
    value.update(overrides)
    return value


def test_crossclip_student_uses_exact_dune_layers_without_decoder() -> None:
    encoder = _FakeDune()
    head = _FakeDPTHead()
    model = DuneFast3RHeadStudent(
        _config(),
        encoder_factory=lambda: encoder,
        head_factory=lambda dim, patch: head,
    )
    output = model(torch.zeros(2, 16, 3, 28, 42))
    assert encoder.requested_layers == [2, 5, 8, 11]
    assert output["pts3d_local"].shape == (2, 16, 28, 42, 3)
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    output["pts3d_local"].sum().backward()
    assert head.scale.grad is not None


def test_joint_training_backpropagates_and_uses_separate_learning_rates() -> None:
    encoder = _FakeDune()
    head = _FakeDPTHead()
    model = DuneFast3RHeadStudent(
        _config(freeze_encoder=False),
        encoder_factory=lambda: encoder,
        head_factory=lambda dim, patch: head,
    )
    model.train()
    output = model(torch.zeros(1, 16, 3, 28, 42))
    output["pts3d_local"].sum().backward()
    assert encoder.training
    assert encoder.anchor.requires_grad
    assert encoder.anchor.grad is not None
    assert head.scale.grad is not None

    optimizer = build_crossclip_optimizer(
        model,
        {
            "learning_rate": 1.0e-4,
            "encoder_learning_rate": 1.0e-5,
            "weight_decay": 0.05,
        },
    )
    assert [group["name"] for group in optimizer.param_groups] == [
        "head",
        "encoder",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [1.0e-4, 1.0e-5]


def test_frozen_encoder_stays_out_of_optimizer() -> None:
    model = DuneFast3RHeadStudent(
        _config(freeze_encoder=True),
        encoder_factory=_FakeDune,
        head_factory=lambda dim, patch: _FakeDPTHead(),
    )
    optimizer = build_crossclip_optimizer(
        model, {"learning_rate": 1.0e-4, "weight_decay": 0.05}
    )
    assert [group["name"] for group in optimizer.param_groups] == ["head"]
    assert not model.encoder.training


def test_crossclip_student_rejects_non_16_frame_input() -> None:
    model = DuneFast3RHeadStudent(
        _config(),
        encoder_factory=_FakeDune,
        head_factory=lambda dim, patch: _FakeDPTHead(),
    )
    with pytest.raises(ValueError, match="exactly 16"):
        model(torch.zeros(1, 15, 3, 28, 42))


def test_crossclip_student_moves_normalization_buffers_to_requested_device() -> None:
    model = DuneFast3RHeadStudent(
        _config(),
        encoder_factory=_FakeDune,
        head_factory=lambda dim, patch: _FakeDPTHead(),
        device=torch.device("meta"),
    )
    assert model.imagenet_mean.device.type == "meta"
    assert model.imagenet_std.device.type == "meta"
    assert next(model.encoder.parameters()).device.type == "meta"
    assert next(model.head.parameters()).device.type == "meta"


def test_fast3r_output_initialization_starts_with_positive_camera_depth() -> None:
    head = nn.Module()
    head.dpt = nn.Module()
    head.dpt.head = nn.Sequential(nn.Conv2d(4, 3, kernel_size=1))
    _initialize_camera_facing_output(head, z_bias=1.0, weight_std=0.0)
    raw = head.dpt.head(torch.zeros(1, 4, 2, 3))
    assert torch.count_nonzero(raw[:, 0:2]) == 0
    assert torch.all(raw[:, 2] == 1.0)
    # Fast3R exp-mode postprocessing preserves vector direction.
    radius = raw.square().sum(dim=1, keepdim=True).sqrt()
    points = raw / radius.clamp_min(1.0e-8) * torch.expm1(radius)
    assert torch.all(points[:, 2] > 0.0)


def test_crossclip_student_rejects_fast3r_decoder() -> None:
    with pytest.raises(ValueError, match="decoder is forbidden"):
        DuneFast3RHeadStudent(
            _config(use_fast3r_decoder=True),
            encoder_factory=_FakeDune,
            head_factory=lambda dim, patch: _FakeDPTHead(),
        )
