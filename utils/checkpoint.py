"""Checkpoint helpers for DA3 distillation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


DIRECT_TEACHER_DISTILLATION_PROTOCOL = "direct_teacher_distillation_v1"
MERGED_STUDENT_CHECKPOINT_FORMAT = "merged_da3_small_student_v1"


def require_student_cache_protocol(
    checkpoint: Dict[str, Any], expected: str
) -> None:
    actual = checkpoint.get("config", {}).get("teacher", {}).get("cache_protocol")
    if actual != expected:
        raise ValueError(
            "Student checkpoint uses incompatible teacher cache protocol {!r}; "
            "expected {!r}".format(actual, expected)
        )


def require_training_objective(checkpoint: Dict[str, Any], expected: str) -> None:
    actual = checkpoint.get("objective_protocol")
    if actual != expected:
        raise ValueError(
            "Checkpoint objective protocol {!r} is incompatible with {!r}. "
            "Start a new training run; optimizer and scheduler state cannot be reused."
            .format(actual, expected)
        )


def require_merged_student_checkpoint(checkpoint: Dict[str, Any]) -> None:
    """Reject training checkpoints and adapter-bearing states at inference."""
    metadata = checkpoint.get("merge_metadata")
    actual_format = metadata.get("format") if isinstance(metadata, dict) else None
    if actual_format != MERGED_STUDENT_CHECKPOINT_FORMAT:
        raise ValueError(
            "Student inference requires a merged checkpoint with format {!r}; "
            "got {!r}. Run the LoRA merge script and evaluate ours.pt, not last.pt."
            .format(MERGED_STUDENT_CHECKPOINT_FORMAT, actual_format)
        )
    student = checkpoint.get("config", {}).get("student", {})
    if student.get("use_backbone_lora") is not False:
        raise ValueError("Merged student config must set use_backbone_lora=false")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("Merged student checkpoint has no non-empty model state")
    adapter_keys = [
        key
        for key in state
        if ".lora_A" in key or ".lora_B" in key or ".base_layer." in key
    ]
    if adapter_keys:
        raise ValueError(
            "Merged student checkpoint still contains LoRA wrapper keys: {}"
            .format(adapter_keys[:20])
        )


def atomic_torch_save(path: Path, state: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp{}".format(path.suffix or ".pt"))
    torch.save(state, temporary)
    temporary.replace(path)


__all__ = [
    "DIRECT_TEACHER_DISTILLATION_PROTOCOL",
    "MERGED_STUDENT_CHECKPOINT_FORMAT",
    "atomic_torch_save",
    "require_merged_student_checkpoint",
    "require_student_cache_protocol",
    "require_training_objective",
]
