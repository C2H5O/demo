from __future__ import annotations

from pathlib import Path

import torch
import pytest

from utils.checkpoint import (
    MERGED_STUDENT_CHECKPOINT_FORMAT,
    require_merged_student_checkpoint,
)
from utils.config import load_config
import utils.merge_student_checkpoint as merge_checkpoint
import evaluation.evaluate_crossclip_projection as crossclip_evaluation


def _checkpoint(key: str = "network.backbone.weight") -> dict:
    return {
        "merge_metadata": {"format": MERGED_STUDENT_CHECKPOINT_FORMAT},
        "config": {"student": {"use_backbone_lora": False}},
        "model": {key: torch.zeros(1)},
    }


def test_baseline_evaluation_selects_training_checkpoint_for_automatic_merge() -> None:
    config = load_config("configs/baselines/J.yaml")
    assert config["vda_evaluation"]["checkpoint"] == "./outputs/baseline_J/last.pt"


def test_merged_checkpoint_contract_accepts_plain_student_state() -> None:
    require_merged_student_checkpoint(_checkpoint())


@pytest.mark.parametrize(
    "key",
    (
        "network.backbone.block.mlp.fc1.lora_A",
        "network.backbone.block.mlp.fc1.lora_B",
        "network.backbone.block.mlp.fc1.base_layer.weight",
    ),
)
def test_merged_checkpoint_contract_rejects_adapter_keys(key: str) -> None:
    with pytest.raises(ValueError, match="LoRA wrapper keys"):
        require_merged_student_checkpoint(_checkpoint(key))


def test_merged_checkpoint_contract_rejects_training_checkpoint() -> None:
    checkpoint = _checkpoint()
    checkpoint.pop("merge_metadata")
    with pytest.raises(ValueError, match="evaluate ours.pt, not last.pt"):
        require_merged_student_checkpoint(checkpoint)


def test_selected_merged_checkpoint_is_used_without_remerging(
    tmp_path, monkeypatch
) -> None:
    selected = tmp_path / "selected.pt"
    selected.write_bytes(b"merged")
    monkeypatch.setattr(
        merge_checkpoint, "_load_torch_checkpoint", lambda path: _checkpoint()
    )
    monkeypatch.setattr(
        merge_checkpoint,
        "merge_student_checkpoint",
        lambda *args, **kwargs: pytest.fail("already merged checkpoint was remerged"),
    )

    assert (
        merge_checkpoint.ensure_merged_student_checkpoint(selected, {})
        == selected.resolve()
    )


def test_evaluator_routes_selected_training_checkpoint_through_auto_merge(
    tmp_path, monkeypatch
) -> None:
    selected = tmp_path / "last.pt"
    merged = tmp_path / "ours.pt"
    output = tmp_path / "evaluation.json"
    config = {
        "device": "cpu",
        "inference": {"acceleration": "none"},
        "student": {"checkpoint": "base.safetensors"},
        "vda_evaluation": {
            "checkpoint": str(selected),
            "output": str(output),
            "split": "test",
            "tae": {"enabled": False},
        },
    }
    monkeypatch.setattr(crossclip_evaluation, "load_config", lambda path: config)
    monkeypatch.setattr(
        crossclip_evaluation,
        "_dataset_and_ground_truth",
        lambda *args: (
            object(),
            {"sequence": {"dataset_id": 1, "frame_paths": []}},
            {"sequence": object()},
            [],
        ),
    )
    calls = []
    monkeypatch.setattr(
        crossclip_evaluation,
        "ensure_merged_student_checkpoint",
        lambda checkpoint, loaded_config: calls.append((checkpoint, loaded_config))
        or merged,
    )

    class ModelLoadReached(RuntimeError):
        pass

    def stop_at_model_load(checkpoint, loaded_config, device, model_source):
        assert checkpoint == merged
        raise ModelLoadReached

    monkeypatch.setattr(crossclip_evaluation, "_evaluation_model", stop_at_model_load)
    with pytest.raises(ModelLoadReached):
        crossclip_evaluation.evaluate_vda(tmp_path / "config.yaml")
    assert calls == [(selected, config)]


@pytest.mark.parametrize("cached_source_sha", ("source-sha", "stale-sha"))
def test_training_checkpoint_reuses_only_fresh_ours_pt(
    tmp_path, monkeypatch, cached_source_sha
) -> None:
    selected = tmp_path / "last.pt"
    merged = tmp_path / "ours.pt"
    base = tmp_path / "model.safetensors"
    base_config = tmp_path / "config.json"
    for path in (selected, merged, base, base_config):
        path.write_bytes(path.name.encode("ascii"))
    training = {
        "config": {"student": {}, "teacher": {"cache_protocol": "test"}},
        "model": {"network.weight": torch.zeros(1)},
    }
    cached = _checkpoint()
    cached["merge_metadata"].update(
        {
            "source_checkpoint_sha256": cached_source_sha,
            "base_checkpoint_sha256": "base-sha",
            "base_config_sha256": "config-sha",
        }
    )
    monkeypatch.setattr(
        merge_checkpoint,
        "_load_torch_checkpoint",
        lambda path: cached if Path(path).name == "ours.pt" else training,
    )
    monkeypatch.setattr(
        merge_checkpoint,
        "_resolved_student_config",
        lambda checkpoint, config: ({}, base, base_config),
    )
    monkeypatch.setattr(
        merge_checkpoint,
        "_sha256",
        lambda path: {
            "last.pt": "source-sha",
            "model.safetensors": "base-sha",
            "config.json": "config-sha",
        }[Path(path).name],
    )
    calls = []
    monkeypatch.setattr(
        merge_checkpoint,
        "merge_student_checkpoint",
        lambda checkpoint, config, output: calls.append((checkpoint, output)) or output,
    )

    assert merge_checkpoint.ensure_merged_student_checkpoint(selected, {}) == merged
    if cached_source_sha == "source-sha":
        assert calls == []
    else:
        assert calls == [(selected.resolve(), merged)]
