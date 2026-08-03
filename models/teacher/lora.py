"""EndoDAC-style standard LoRA for frozen ``torch.nn.Linear`` layers."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Keep a frozen projection and add ``alpha / rank * B(A(x))``.

    This follows EndoDAC's ordinary ``Linear`` LoRA branch: ``lora_A`` and
    ``lora_B`` are direct parameters, A uses Kaiming initialization, B starts
    at zero, and the pretrained weight and bias stay frozen.  The existing
    VGGT-Omega linear module is retained so its loaded weights are never
    reconstructed.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(
            base_layer.weight.new_zeros((self.rank, base_layer.in_features))
        )
        self.lora_B = nn.Parameter(
            base_layer.weight.new_zeros((base_layer.out_features, self.rank))
        )

        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        update = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return base + update * self.scaling


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return only trainable low-rank tensors, detached on CPU."""
    state = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            state[name] = tensor.detach().cpu()
    if not state:
        raise RuntimeError("No LoRA parameters were found in the model")
    return state


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """Load a LoRA-only state dict without touching pretrained base weights."""
    expected = set(extract_lora_state_dict(model))
    provided = set(state_dict)
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "LoRA checkpoint mismatch. Missing: {}; unexpected: {}".format(
                missing, unexpected
            )
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    non_lora_unexpected = [
        name for name in incompatible.unexpected_keys if name not in unexpected
    ]
    if non_lora_unexpected:
        raise RuntimeError("Unexpected non-LoRA keys: {}".format(non_lora_unexpected))
