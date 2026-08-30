"""Minimal standard LoRA layers used by the DA3 DINOv2 backbone."""

from __future__ import annotations

import math
import re
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer plus the standard low-rank ``B(A(x))`` update."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0,1)")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(
            base_layer.weight.new_empty((rank, base_layer.in_features))
        )
        self.lora_B = nn.Parameter(
            base_layer.weight.new_zeros((base_layer.out_features, rank))
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        update = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return base + update * self.scaling


_DA3_MLP_PATTERN = re.compile(
    r"(?:^|\.)pretrained\.blocks(?:\.\d+)+\.mlp\.(fc1|fc2)$"
)


def inject_da3_mlp_lora(
    backbone: nn.Module, rank: int, alpha: float, dropout: float
) -> Dict[str, LoRALinear]:
    """Inject LoRA into only the official DINOv2 block MLP fc1/fc2 layers."""

    targets = [
        (name, module)
        for name, module in backbone.named_modules()
        if isinstance(module, nn.Linear) and _DA3_MLP_PATTERN.search(name)
    ]
    injected: Dict[str, LoRALinear] = {}
    for name, module in targets:
        parent_name, child_name = name.rsplit(".", 1)
        parent = backbone.get_submodule(parent_name)
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, wrapped)
        injected[name] = wrapped
    return injected


__all__ = ["LoRALinear", "inject_da3_mlp_lora"]
