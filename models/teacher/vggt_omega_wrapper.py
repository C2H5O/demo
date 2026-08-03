"""VGGT-Omega loading, base-weight freezing, and LoRA lifecycle."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.teacher.lora_injection import (
    assert_only_lora_trainable,
    inject_lora_into_mlp,
    lora_config_kwargs,
)
from utils.checkpoint import load_lora_checkpoint


def _import_vggt_omega_class() -> type:
    try:
        module = importlib.import_module("vggt_omega.models")
    except ImportError as error:
        raise RuntimeError(
            "The VGGT-Omega package is not importable. Install the external repository "
            "into this environment (for example, `pip install -e /path/to/vggt-omega`)."
        ) from error
    return module.VGGTOmega


def _checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError("VGGT-Omega checkpoint not found: {}".format(path))
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return checkpoint
    return checkpoint.get("model", checkpoint.get("state_dict", checkpoint))


class VGGTOmegaTeacher(nn.Module):
    """Thin independent-project wrapper around the installed VGGT-Omega package."""

    def __init__(
        self,
        model: nn.Module,
        injected_module_names: Tuple[str, ...],
        uses_lora: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.injected_module_names = injected_module_names
        self.uses_lora = bool(uses_lora)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        load_lora: bool = True,
        inject_lora: bool = True,
    ) -> "VGGTOmegaTeacher":
        if not bool(config.get("freeze_backbone", True)):
            raise ValueError("VGGT-Omega backbone must remain frozen for LoRA adaptation")
        if not bool(config.get("freeze_heads", True)):
            raise ValueError("All VGGT-Omega output heads must remain frozen")
        checkpoint_value = config.get("pretrained_checkpoint")
        if not checkpoint_value:
            raise ValueError("teacher.pretrained_checkpoint must be configured")
        model_class = _import_vggt_omega_class()
        model = model_class()
        model.load_state_dict(_checkpoint_state(Path(checkpoint_value)), strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        if inject_lora:
            lora_config = dict(config.get("lora", {}))
            if not bool(lora_config.get("enabled", True)):
                raise ValueError(
                    "LoRA teacher construction requires teacher.lora.enabled=true"
                )
            names = inject_lora_into_mlp(
                model, **lora_config_kwargs(lora_config)
            )
            wrapper = cls(model, tuple(names), uses_lora=True)
            lora_checkpoint = config.get("lora_checkpoint")
            if load_lora and lora_checkpoint:
                load_lora_checkpoint(Path(lora_checkpoint), wrapper)
            assert_only_lora_trainable(wrapper)
        else:
            if load_lora:
                raise ValueError(
                    "load_lora must be false when inject_lora is false"
                )
            wrapper = cls(model, (), uses_lora=False)
            if any(parameter.requires_grad for parameter in wrapper.parameters()):
                raise RuntimeError(
                    "Base teacher construction left trainable parameters"
                )
        return wrapper.to(device) if device is not None else wrapper

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(images)

    def freeze_for_distillation(self) -> "VGGTOmegaTeacher":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self
