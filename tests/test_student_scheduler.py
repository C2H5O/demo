from __future__ import annotations

import inspect
import sys
import types

import pytest
import torch

from models.student.distill3r_wrapper import Distill3RStudent, _pinned_dune_hub
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
    assert config["dataset"]["normalize_mode"] == "zero_one"
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
    assert config["student"]["image_height"] == 448
    assert config["student"]["image_width"] == 560
    assert config["student"]["patch_size"] == 14
    assert config["student"]["encoder_type"] == "dune"
    assert config["student"]["decoder_depth"] == 6
    assert config["student"]["decoder_attention_implementation"] == "flash_attention"
    assert config["student"]["pretrained_checkpoint"] == (
        "./checkpoints/dune/dune_vitsmall14_448.pth"
    )
    assert config["student"]["conf_mode"] == ["sigmoid", 0.0, 1.0]
    assert config["student"]["use_local_dune_submodule"] is True
    assert 448 // 14 == 32
    assert 560 // 14 == 40


class _FakeOfficialDistill3R(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.seen_views = None

    def forward(self, views):
        self.seen_views = views
        outputs = []
        for view in views:
            batch, _, height, width = view["img"].shape
            xyz = view["img"].new_zeros(batch, height, width, 3)
            confidence = view["img"].new_full((batch, height, width), 0.25)
            outputs.append(
                {
                    "pts3d_in_other_view": xyz,
                    "pts3d_local": xyz + 1.0,
                    "conf": confidence,
                    "conf_local": confidence * 3.0,
                }
            )
        return outputs


def test_official_distill3r_adapter_preserves_448x560_contract() -> None:
    config = load_config("configs/student_distillation.yaml")["student"]
    model = Distill3RStudent(config, model_factory=_FakeOfficialDistill3R)
    images = torch.rand(2, 3, 3, 448, 560)

    output = model(images)

    assert output["xyz_global"].shape == (2, 3, 448, 560, 3)
    assert output["xyz_local"].shape == (2, 3, 448, 560, 3)
    assert output["conf_global"].shape == (2, 3, 448, 560)
    assert output["conf_local"].shape == (2, 3, 448, 560)
    assert output["conf_global"].min().item() == pytest.approx(0.25)
    assert output["conf_local"].max().item() == pytest.approx(0.75)
    assert len(model.student.seen_views) == 3
    assert model.student.seen_views[0]["true_shape"].tolist() == [
        [448, 560],
        [448, 560],
    ]
    assert model.student.kwargs["decoder_depth"] == 6
    assert model.student.kwargs["encoder_type"] == "dune"
    assert model.student.kwargs["conf_mode"] == ["sigmoid", 0.0, 1.0]
    assert "decoder_attention_implementation" not in model.student.kwargs
    assert "pretrained_checkpoint" not in model.student.kwargs


def test_distill3r_adapter_rejects_other_runtime_resolution() -> None:
    config = load_config("configs/student_distillation.yaml")["student"]
    model = Distill3RStudent(config, model_factory=_FakeOfficialDistill3R)

    with pytest.raises(ValueError, match="expects 448x560"):
        model(torch.rand(1, 2, 3, 448, 546))


def test_distill3r_requires_configured_local_checkpoint(tmp_path) -> None:
    config = load_config("configs/student_distillation.yaml")["student"]
    config["pretrained_checkpoint"] = str(tmp_path / "missing.pth")

    with pytest.raises(FileNotFoundError, match="Configured DUNE checkpoint"):
        Distill3RStudent(config)


def test_distill3r_hub_redirect_loads_configured_checkpoint(
    monkeypatch, tmp_path
) -> None:
    checkpoint = tmp_path / "dune_vitsmall14_448.pth"
    checkpoint.write_bytes(b"test checkpoint path")
    encoder = torch.nn.Identity()
    seen = []

    model_package = types.ModuleType("model")
    model_package.__path__ = []
    dune_module = types.ModuleType("model.dune")

    def fake_loader(path):
        seen.append(path)
        return encoder, 0

    dune_module.load_dune_encoder_from_checkpoint = fake_loader
    model_package.dune = dune_module
    monkeypatch.setitem(sys.modules, "model", model_package)
    monkeypatch.setitem(sys.modules, "model.dune", dune_module)

    with _pinned_dune_hub(True, checkpoint):
        loaded = torch.hub.load(
            "naver/dune",
            "dune_vitsmall_14_448_encoder",
            trust_repo=True,
        )

    assert loaded is encoder
    assert seen == [str(checkpoint)]


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
