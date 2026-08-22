from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import evaluate_vggtomast3r as dispatcher
from evaluate_vggtomast3r import select_protocol
from evaluation.evaluate_vggtomast3r_vda import (
    _pair_reference_disparities,
    _select_evaluation_config,
    evaluate,
)
from utils.checkpoint import require_student_cache_protocol
from utils.config import load_config


class _ReferenceDepthModel(torch.nn.Module):
    def forward(self, images: torch.Tensor):
        depth_a = images[:, 0, 0]
        depth_b = images[:, 1, 0]
        zeros = torch.zeros(*depth_a.shape, 2, dtype=depth_a.dtype)
        return {
            "pts3d_ref": torch.cat((zeros, depth_a.unsqueeze(-1)), -1),
            "pts3d_other_local": torch.cat((zeros, depth_b.unsqueeze(-1)), -1),
        }


def test_vggtomast3r_defaults_to_vda_and_retains_endo3r() -> None:
    config = load_config("configs/vggtomast3r_v1.yaml")

    assert select_protocol(config) == "vda"
    assert config["vda_evaluation"]["output"].endswith(
        "evaluation_test_vda.json"
    )
    assert config["endo3r_evaluation"]["protocol"] == "endo3r"
    assert config["endo3r_evaluation"]["output"].endswith(
        "evaluation_test_endo3r.json"
    )
    assert select_protocol(config, "endo3r") == "endo3r"


def test_vda_config_section_is_selected() -> None:
    config = {
        "evaluation": {"protocol": "vda"},
        "vda_evaluation": {"protocol": "vda", "split": "test"},
    }

    assert _select_evaluation_config(config) == config["vda_evaluation"]


def test_dispatcher_routes_explicit_endo3r(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("evaluation:\n  protocol: vda\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        dispatcher,
        "evaluate_vda",
        lambda *args: calls.append(("vda", args)),
    )
    monkeypatch.setattr(
        dispatcher,
        "evaluate_endo3r",
        lambda *args: calls.append(("endo3r", args)),
    )

    dispatcher.evaluate(config_path, protocol="endo3r")

    assert calls[0][0] == "endo3r"


def test_pair_adapter_uses_two_camera_local_depths() -> None:
    images = torch.zeros(1, 2, 3, 2, 3)
    images[:, 0, 0] = 2.0
    images[:, 1, 0] = 4.0

    disparities = _pair_reference_disparities(_ReferenceDepthModel(), images)

    np.testing.assert_allclose(disparities[0], 0.5)
    np.testing.assert_allclose(disparities[1], 0.25)


def test_v1_vda_reuses_streaming_vda_core() -> None:
    source = inspect.getsource(evaluate)

    assert "vda_core._SequencePredictionSpool" in source
    assert "vda_core._evaluate_sequence" in source
    assert "pts3d_other_in_ref" not in inspect.getsource(
        _pair_reference_disparities
    )


def test_student_checkpoint_requires_frame_local_protocol() -> None:
    require_student_cache_protocol(
        {"config": {"teacher": {"cache_protocol": "frame_local_v1"}}}
    )
    with pytest.raises(ValueError, match="incompatible teacher cache protocol"):
        require_student_cache_protocol({"config": {}})
