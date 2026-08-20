"""Single import/load boundary for the pinned official MASt3R and DUNE sources."""

from __future__ import annotations

import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAST3R_ROOT = PROJECT_ROOT / "external" / "MASt3R"
DUNE_ROOT = PROJECT_ROOT / "external" / "DUNE"


def _prepend_package_path(package_name: str, path: Path) -> None:
    """Let pinned upstream sibling imports coexist with project packages."""
    package = sys.modules.get(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    value = str(path)
    if value in package_path:
        package_path.remove(value)
    package_path.insert(0, value)


def ensure_official_sources_importable() -> None:
    required = (MAST3R_ROOT / "mast3r", MAST3R_ROOT / "dust3r" / "dust3r", DUNE_ROOT / "model")
    missing = [path for path in required if not path.is_dir()]
    if missing:
        raise RuntimeError(
            "Official sources are incomplete: {}. Run git submodule update --init --recursive.".format(
                ", ".join(str(path) for path in missing)
            )
        )
    for path in (DUNE_ROOT, MAST3R_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    # CroCo uses the generic top-level package name ``models`` and DUNE uses
    # ``utils``. The host project already owns packages with those names, so
    # plain sys.path insertion is insufficient once they are in sys.modules.
    _prepend_package_path("models", MAST3R_ROOT / "dust3r" / "croco" / "models")
    _prepend_package_path("utils", DUNE_ROOT / "utils")


@contextmanager
def local_dune_hub(dune_checkpoint: Path) -> Iterator[None]:
    """Replace only the official ``naver/dune`` hub call with pinned local code/data."""
    ensure_official_sources_importable()
    if not dune_checkpoint.is_file():
        raise FileNotFoundError("Local DUNE encoder checkpoint not found: {}".format(dune_checkpoint))
    original = torch.hub.load

    def load(repo_or_dir: Any, model: str, *args: Any, **kwargs: Any) -> Any:
        if str(repo_or_dir).lower().rstrip("/") == "naver/dune":
            if not str(model).endswith("_encoder"):
                raise ValueError("DUNE-MASt3R requested a non-encoder hub model: {}".format(model))
            from model.dune import load_dune_encoder_from_checkpoint

            encoder, _ = load_dune_encoder_from_checkpoint(str(dune_checkpoint))
            return encoder
        return original(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = original


def load_pinned_dune_mast3r(
    combined_checkpoint: Path,
    dune_checkpoint: Path,
    device: torch.device,
) -> nn.Module:
    """Call the official checkpoint loader while making its DUNE load offline."""
    ensure_official_sources_importable()
    if not combined_checkpoint.is_file():
        raise FileNotFoundError("DUNE-MASt3R checkpoint not found: {}".format(combined_checkpoint))
    from mast3r.model import load_dune_mast3r_model

    imported = Path(inspect.getfile(load_dune_mast3r_model)).resolve()
    try:
        imported.relative_to(MAST3R_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("MASt3R was imported outside pinned submodule: {}".format(imported)) from error
    with local_dune_hub(dune_checkpoint):
        model = load_dune_mast3r_model(str(combined_checkpoint), device=device, verbose=True)
    return model


__all__ = [
    "DUNE_ROOT", "MAST3R_ROOT", "ensure_official_sources_importable",
    "load_pinned_dune_mast3r", "local_dune_hub",
]
