"""Checkpoint helpers for the cross-clip student."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def require_student_cache_protocol(
    checkpoint: Dict[str, Any], expected: str
) -> None:
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


__all__ = ["atomic_torch_save", "require_student_cache_protocol"]
