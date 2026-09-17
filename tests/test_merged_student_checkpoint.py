from __future__ import annotations

import torch
import pytest

from utils.checkpoint import (
    MERGED_STUDENT_CHECKPOINT_FORMAT,
    require_merged_student_checkpoint,
)
from utils.config import load_config


def _checkpoint(key: str = "network.backbone.weight") -> dict:
    return {
        "merge_metadata": {"format": MERGED_STUDENT_CHECKPOINT_FORMAT},
        "config": {"student": {"use_backbone_lora": False}},
        "model": {key: torch.zeros(1)},
    }


def test_baseline_evaluation_uses_merged_checkpoint() -> None:
    config = load_config("configs/baselines/J.yaml")
    assert config["vda_evaluation"]["checkpoint"] == "./outputs/baseline_J/ours.pt"


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
