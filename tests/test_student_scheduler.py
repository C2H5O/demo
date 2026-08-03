from __future__ import annotations

import inspect

import pytest
import torch

from models.student.dune_model import DUNEStudentConfig, DUNEVisionTransformer, PointMapHead
from trainers.student_distillation_trainer import (
    _amp_settings,
    _build_scheduler,
    train,
)
from utils.config import load_config


def _optimizer() -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.ones(1))
    return torch.optim.AdamW([parameter], lr=1.0e-5)


def test_student_cosine_scheduler_decays_from_configured_initial_lr() -> None:
    optimizer = _optimizer()
    scheduler = _build_scheduler(
        optimizer,
        total_steps=10,
        warmup_steps=0,
        initial_learning_rate=1.0e-5,
        minimum_learning_rate=1.0e-6,
    )

    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] == pytest.approx(1.0e-5)
    assert learning_rates[-1] == pytest.approx(1.0e-6)
    assert all(
        following <= previous
        for previous, following in zip(learning_rates, learning_rates[1:])
    )


def test_student_scheduler_resume_preserves_decay_position() -> None:
    optimizer = _optimizer()
    scheduler = _build_scheduler(
        optimizer,
        total_steps=20,
        warmup_steps=0,
        initial_learning_rate=1.0e-5,
        minimum_learning_rate=1.0e-6,
    )
    for _ in range(7):
        optimizer.step()
        scheduler.step()

    resumed_optimizer = _optimizer()
    resumed_scheduler = _build_scheduler(
        resumed_optimizer,
        total_steps=20,
        warmup_steps=0,
        initial_learning_rate=1.0e-5,
        minimum_learning_rate=1.0e-6,
    )
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed_scheduler.load_state_dict(scheduler.state_dict())

    assert resumed_scheduler.last_epoch == scheduler.last_epoch
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )


def test_rectangular_training_resolution_and_patch_grid() -> None:
    config = load_config("configs/student_distillation.yaml")

    assert config["dataset"]["image_height"] == 448
    assert config["dataset"]["image_width"] == 560
    assert config["teacher"]["preprocess_mode"] == "max_size"
    assert config["teacher"]["image_resolution"] == 560
    assert config["teacher"]["image_height"] == 448
    assert config["teacher"]["image_width"] == 560
    assert config["dataset"]["drop_incomplete_clip"] is False
    assert config["dataset"]["ground_truth"]["scale"] == 0.001
    assert config["loss"]["supervised_depth_min_depth"] == 1e-4
    assert config["loss"]["supervised_depth_max_depth"] == 100.0
    assert config["dataloader"]["batch_size"] == 1
    assert config["training"]["gradient_accumulation_steps"] == 32
    assert config["training"]["amp_dtype"] == "auto"
    assert config["training"]["amp_initial_scale"] == 128.0

    encoder = DUNEVisionTransformer(
        DUNEStudentConfig(encoder_depth=0)
    )
    positions = encoder._position_embedding(448 // 14, 560 // 14)
    assert positions.shape[1] == 1 + 32 * 40


def test_point_map_head_uses_resize_convolution_at_runtime_resolution() -> None:
    head = PointMapHead(dimension=8)
    assert not any(isinstance(module, torch.nn.ConvTranspose2d) for module in head.modules())

    tokens = torch.randn(2, 32 * 40, 8, requires_grad=True)
    xyz, confidence = head(tokens, grid=(32, 40), output_size=(448, 560))

    assert xyz.shape == (2, 448, 560, 3)
    assert confidence.shape == (2, 448, 560)
    loss = xyz.square().mean() + confidence.mean()
    loss.backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_epoch_checkpoint_is_written_before_validation() -> None:
    source = inspect.getsource(train)

    assert source.index('_atomic_checkpoint(output_dir / "last.pt"') < source.index(
        "validation_logs = validate("
    )


def test_amp_auto_prefers_bfloat16_without_grad_scaler(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    enabled, dtype, scaler_enabled = _amp_settings(
        {"amp": True, "amp_dtype": "auto"}, torch.device("cuda")
    )

    assert enabled is True
    assert dtype == torch.bfloat16
    assert scaler_enabled is False


def test_amp_float16_uses_grad_scaler(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    enabled, dtype, scaler_enabled = _amp_settings(
        {"amp": True, "amp_dtype": "auto"}, torch.device("cuda")
    )

    assert enabled is True
    assert dtype == torch.float16
    assert scaler_enabled is True
