"""Frozen base VGGT-Omega loading for offline teacher-cache inference."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.attention_capture import VGGTOmegaAttentionCapture


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

    def __init__(
        self,
        model: nn.Module,
        attention_capture: Optional[VGGTOmegaAttentionCapture] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.attention_capture = attention_capture

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
        save_attention = bool(config.get("save_attention", False))
        attention_capture = None
        if save_attention:
            dtype_name = str(config.get("attention_cache_dtype", "float16")).lower()
            dtypes = {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32}
            if dtype_name not in dtypes:
                raise ValueError("teacher.attention_cache_dtype must be float16 or float32")
            layers = config.get("attention_layers")
            if layers is None:
                raise ValueError("teacher.attention_layers must be configured when save_attention=true")
            attention_capture = VGGTOmegaAttentionCapture(
                model, layers, cache_dtype=dtypes[dtype_name]
            )
        wrapper = cls(model, attention_capture=attention_capture).freeze_for_inference()
        return wrapper.to(device) if device is not None else wrapper

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.attention_capture is None:
            return self.model(images)
        self.attention_capture.begin(images)
        output = dict(self.model(images))
        output["attention"] = self.attention_capture.take()
        return output

    def freeze_for_inference(self) -> "VGGTOmegaTeacher":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


__all__ = ["VGGTOmegaTeacher"]
