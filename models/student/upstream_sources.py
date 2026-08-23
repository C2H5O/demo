"""Import boundary for the pinned DUNE encoder and Fast3R DPT head."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DUNE_ROOT = PROJECT_ROOT / "external" / "DUNE"
FAST3R_ROOT = PROJECT_ROOT / "external" / "Distill3R" / "external" / "fast3r"


def _prepend_package_path(package_name: str, path: Path) -> None:
    package = sys.modules.get(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    value = str(path)
    if value in package_path:
        package_path.remove(value)
    package_path.insert(0, value)


def ensure_dune_source_importable() -> None:
    required = (DUNE_ROOT / "model", DUNE_ROOT / "utils")
    missing = [path for path in required if not path.is_dir()]
    if missing:
        raise RuntimeError(
            "Pinned DUNE source is incomplete: {}. Run git submodule update "
            "--init --recursive.".format(", ".join(str(path) for path in missing))
        )
    value = str(DUNE_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)
    _prepend_package_path("utils", DUNE_ROOT / "utils")


def ensure_fast3r_source_importable() -> None:
    package = FAST3R_ROOT / "fast3r"
    if not package.is_dir():
        raise RuntimeError(
            "Pinned Fast3R source is incomplete: {}. Run git submodule update "
            "--init --recursive.".format(package)
        )
    value = str(FAST3R_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)


def load_pinned_dune_encoder(
    dune_checkpoint: Path, device: torch.device
) -> nn.Module:
    ensure_dune_source_importable()
    if not dune_checkpoint.is_file():
        raise FileNotFoundError(
            "Local DUNE encoder checkpoint not found: {}".format(dune_checkpoint)
        )
    from model.dune import load_dune_encoder_from_checkpoint

    encoder, _ = load_dune_encoder_from_checkpoint(str(dune_checkpoint))
    return encoder.to(device)


__all__ = [
    "DUNE_ROOT",
    "FAST3R_ROOT",
    "PROJECT_ROOT",
    "ensure_dune_source_importable",
    "ensure_fast3r_source_importable",
    "load_pinned_dune_encoder",
]
