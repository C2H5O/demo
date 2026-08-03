"""Public DUNE student construction and freezing helpers."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from models.student.dune_model import DUNEViTSmallPointMapStudent


def build_dune_student(config: Dict[str, Any]) -> DUNEViTSmallPointMapStudent:
    return DUNEViTSmallPointMapStudent(config)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module
