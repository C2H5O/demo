"""Checkpoint helpers for LoRA-only teachers and complete student training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


FRAME_LOCAL_CACHE_PROTOCOL = "frame_local_v1"


def require_student_cache_protocol(
    checkpoint: Dict[str, Any], expected: str = FRAME_LOCAL_CACHE_PROTOCOL
) -> None:
    """Reject checkpoints trained against a different teacher-target geometry."""
    actual = checkpoint.get("config", {}).get("teacher", {}).get("cache_protocol")
    if actual != expected:
        raise ValueError(
            "Student checkpoint uses incompatible teacher cache protocol {!r}; "
            "expected {!r}".format(actual, expected)
        )


def atomic_torch_save(path: Path, state: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp{}".format(path.suffix or ".pt"))
    torch.save(state, temporary)
    temporary.replace(path)


def save_lora_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    global_step: int,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    config: Dict[str, Any],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    from models.teacher.lora import extract_lora_state_dict

    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "lora_state_dict": extract_lora_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config,
    }
    atomic_torch_save(Path(path), state)


def load_lora_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    from models.teacher.lora import load_lora_state_dict

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("LoRA checkpoint not found: {}".format(path))
    checkpoint = torch.load(str(path), map_location=map_location, weights_only=False)
    if "lora_state_dict" not in checkpoint:
        raise KeyError("Checkpoint has no lora_state_dict: {}".format(path))
    load_lora_state_dict(model, checkpoint["lora_state_dict"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def save_student_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    atomic_torch_save(Path(path), state)


def load_student_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Student checkpoint not found: {}".format(path))
    return torch.load(str(path), map_location=map_location, weights_only=False)
