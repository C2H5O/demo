from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from endodaveval.endodav import (
    ModelForwardTimer,
    _checkpoint_uses_temporal_lora,
    _weights_for_model,
    disp_to_depth,
    infer_official_full_video,
    official_constructor_kwargs,
    validate_official_repository,
)


class FakeOfficialModel(nn.Module):
    def forward(self, value):
        return value

    def infer_video_depth(self, frames):
        value = torch.from_numpy(frames[..., 0].astype(np.float32) / 255.0)
        return self.forward(value).numpy()


def test_official_constructor_contract_is_fixed(tmp_path: Path) -> None:
    kwargs = official_constructor_kwargs(tmp_path)
    assert kwargs["encoder"] == "vits"
    assert kwargs["lora_type"] == "ssb" and kwargs["r"] == 4
    assert kwargs["image_shape"] == (224, 280)
    assert kwargs["residual_block_indexes"] == []
    assert kwargs["include_cls_token"] is True
    assert kwargs["disable_conv_head"] is True
    assert kwargs["inv_sigmoid"] is False
    assert kwargs["temporal_lora"] is False
    assert kwargs["out_sigmoid"] is False


def test_checkpoint_temporal_lora_and_metadata_contract(tmp_path: Path) -> None:
    temporal_key = (
        "head.motion_modules.0.temporal_transformer."
        "transformer_blocks.0.ff.net.2.lora_A"
    )
    state = {
        "height": 256,
        "width": 320,
        "use_stereo": False,
        "model.weight": torch.ones(1),
        temporal_key: torch.ones(1),
    }

    assert _checkpoint_uses_temporal_lora(state) is True
    kwargs = official_constructor_kwargs(tmp_path, temporal_lora=True)
    assert kwargs["temporal_lora"] is True

    weights, unexpected = _weights_for_model(
        state, ["model.weight", temporal_key]
    )
    assert set(weights) == {"model.weight", temporal_key}
    assert unexpected == []


def test_unknown_checkpoint_tensor_still_fails_strict_contract() -> None:
    weights, unexpected = _weights_for_model(
        {"model.weight": torch.ones(1), "unknown.weight": torch.ones(1)},
        ["model.weight"],
    )
    assert set(weights) == {"model.weight"}
    assert unexpected == ["unknown.weight"]


def test_official_disp_to_depth_then_reciprocal_has_expected_semantics() -> None:
    raw = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    scaled, depth = disp_to_depth(raw, 0.1, 150.0)
    np.testing.assert_allclose(1.0 / depth, scaled, rtol=1e-6)
    assert depth[0] == pytest.approx(150.0)
    assert depth[-1] == pytest.approx(0.1)


def test_forward_timer_restores_model_and_counts_only_forward_calls() -> None:
    model = FakeOfficialModel().eval()
    original = model.forward
    with ModelForwardTimer(model, "cpu") as timer:
        model.forward(torch.ones(1))
        model.forward(torch.ones(1))
    assert timer.call_count == 2 and timer.seconds >= 0
    assert model.forward == original


def test_full_video_adapter_calls_official_method_without_gt() -> None:
    model = FakeOfficialModel().eval()
    frames = np.full((3, 4, 6, 3), 128, dtype=np.uint8)
    depth, timing = infer_official_full_video(model, frames, "cpu")
    assert depth.shape == (3, 4, 6)
    assert timing["model_forward_call_count"] == 1
    assert timing["sequence_pipeline_seconds"] >= timing["model_forward_seconds"]


def test_repository_validation_is_read_only(tmp_path: Path) -> None:
    files = [
        tmp_path / "models/endodav/endodav.py",
        tmp_path / "models/endodav/__init__.py",
        tmp_path / "utils/layers.py",
        tmp_path / "evaluate_depth_video_pose.py",
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sentinel", encoding="utf-8")
    before = {path: path.read_bytes() for path in files}
    validate_official_repository(tmp_path)
    after = {path: path.read_bytes() for path in files}
    assert after == before
