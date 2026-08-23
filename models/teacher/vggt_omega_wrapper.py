"""Frozen base VGGT-Omega loading for offline teacher-cache inference."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _import_vggt_omega_class() -> type:
    try:
        module = importlib.import_module("vggt_omega.models")
    except ImportError as error:
        raise RuntimeError(
            "VGGT-Omega is not importable. Install its source as an editable package."
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
    """Frozen pretrained teacher used only for offline cache generation."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> "VGGTOmegaTeacher":
        if str(config.get("variant", "base")) != "base":
            raise ValueError("This experiment requires teacher.variant=base")
        if not bool(config.get("frozen", True)):
            raise ValueError("The teacher must remain frozen")
        if not bool(config.get("freeze_backbone", True)):
            raise ValueError("VGGT-Omega backbone must remain frozen")
        if not bool(config.get("freeze_heads", True)):
            raise ValueError("VGGT-Omega output heads must remain frozen")
        checkpoint_value = config.get("pretrained_checkpoint")
        if not checkpoint_value:
            raise ValueError("teacher.pretrained_checkpoint must be configured")
        model = _import_vggt_omega_class()()
        model.load_state_dict(_checkpoint_state(Path(checkpoint_value)), strict=True)
        wrapper = cls(model).freeze_for_inference()
        return wrapper.to(device) if device is not None else wrapper

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(images)

    def freeze_for_inference(self) -> "VGGTOmegaTeacher":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


__all__ = ["VGGTOmegaTeacher"]
