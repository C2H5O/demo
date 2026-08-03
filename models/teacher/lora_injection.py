"""Discover and replace VGGT-Omega MLP projections with native LoRA layers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Union

import torch.nn as nn

from models.teacher.lora import LoRALinear


_VGGT_IMAGE_ENCODER_MLP_PATTERN = re.compile(
    r"(?:^|\.)patch_embed\.blocks\.(\d+)\.mlp\."
    r"(fc1|fc2|linear1|linear2)$"
)


@dataclass(frozen=True)
class ParameterSummary:
    total: int
    trainable: int
    percentage: float
    trainable_names: List[str]


def _layer_is_selected(name: str, target_layers: Union[str, Sequence[int]]) -> bool:
    if target_layers == "all":
        return True
    match = _VGGT_IMAGE_ENCODER_MLP_PATTERN.search(name)
    if match is None:
        return False
    selected = {int(index) for index in target_layers}
    return int(match.group(1)) in selected


def discover_mlp_linear_names(
    model: nn.Module,
    target_layers: Union[str, Sequence[int]] = "all",
) -> List[str]:
    """Inspect the model instance; no fixed list of block paths is assumed."""
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if (
            _VGGT_IMAGE_ENCODER_MLP_PATTERN.search(name)
            and _layer_is_selected(name, target_layers)
        ):
            names.append(name)
    return names


def inject_lora_into_mlp(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target: str = "mlp",
    target_layers: Union[str, Sequence[int]] = "all",
) -> List[str]:
    """Freeze the teacher and replace MLP projections with ordinary LoRA.

    EndoDAC applies standard LoRA only to its DINO image encoder's
    ``encoder.blocks.*.mlp.fc1/fc2``.  The architectural counterpart in
    VGGT-Omega is ``aggregator.patch_embed.blocks``.  The temporal
    ``frame_blocks`` and ``inter_frame_blocks`` remain frozen.
    """
    if target != "mlp":
        raise ValueError("Only target='mlp' is supported; attention Q/K/V are intentionally excluded")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    target_names = discover_mlp_linear_names(model, target_layers)
    if not target_names:
        raise RuntimeError(
            "No VGGT-Omega MLP linear layers were found. "
            "Expected image-encoder names under "
            "aggregator.patch_embed.blocks.*.mlp.fc1/fc2."
        )
    for name in target_names:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        base_layer = getattr(parent, child_name)
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("{} changed during LoRA injection".format(name))
        setattr(parent, child_name, LoRALinear(base_layer, rank, alpha, dropout))

    assert_only_lora_trainable(model)
    return target_names


def summarize_trainable_parameters(model: nn.Module) -> ParameterSummary:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_items = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = sum(parameter.numel() for _, parameter in trainable_items)
    percentage = 100.0 * trainable / total if total else 0.0
    return ParameterSummary(
        total=total,
        trainable=trainable,
        percentage=percentage,
        trainable_names=[name for name, _ in trainable_items],
    )


def assert_only_lora_trainable(model: nn.Module) -> ParameterSummary:
    summary = summarize_trainable_parameters(model)
    unexpected = [
        name
        for name in summary.trainable_names
        if not name.endswith(".lora_A") and not name.endswith(".lora_B")
    ]
    if unexpected:
        raise RuntimeError("Unexpected trainable teacher parameters: {}".format(unexpected))
    if not summary.trainable_names:
        raise RuntimeError("Teacher has no trainable LoRA parameters")
    return summary


def print_trainable_parameters(model: nn.Module) -> None:
    summary = assert_only_lora_trainable(model)
    print("Total parameters: {:,}".format(summary.total))
    print("Trainable parameters: {:,}".format(summary.trainable))
    print("Trainable percentage: {:.6f}%".format(summary.percentage))
    print("Trainable parameter names:")
    for name in summary.trainable_names:
        print("  - {}".format(name))


def set_lora_training_mode(model: nn.Module) -> None:
    """Keep the frozen teacher in eval mode while enabling LoRA dropout."""
    model.eval()
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.training = True
            module.dropout.train()


def lora_config_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    lora_type = str(config.get("type", "lora")).lower()
    if lora_type != "lora":
        raise ValueError(
            "Only ordinary LoRA is supported for teacher adaptation; "
            "set teacher.lora.type='lora' (DV-LoRA is intentionally excluded)"
        )
    return {
        "rank": int(config.get("rank", 4)),
        "alpha": float(config.get("alpha", 1.0)),
        "dropout": float(config.get("dropout", 0.0)),
        "target": str(config.get("target", "mlp")),
        "target_layers": config.get("target_layers", "all"),
    }
