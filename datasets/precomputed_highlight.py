"""Strict on-disk contract for offline highlight masks and inpainted RGB."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from datasets.highlight import HighlightDetectionConfig


HIGHLIGHT_PRECOMPUTE_VERSION = "vggtoda3-highlight-v1"
HIGHLIGHT_MANIFEST_NAME = "_highlight_precompute_complete.json"
DEFAULT_MASK_DIRECTORY = "student_highlight_mask"
DEFAULT_INPAINTED_DIRECTORY = "student_inpainted_rgb"
_STORAGE_KEYS = {
    "storage",
    "mask_directory_name",
    "inpainted_directory_name",
}


def parse_highlight_options(
    options: Mapping[str, Any] | None,
) -> Tuple[bool, HighlightDetectionConfig, str, str]:
    values = dict(options or {})
    enabled = bool(values.get("enabled", False))
    storage = str(values.pop("storage", "precomputed"))
    mask_directory = str(values.pop("mask_directory_name", DEFAULT_MASK_DIRECTORY))
    inpainted_directory = str(
        values.pop("inpainted_directory_name", DEFAULT_INPAINTED_DIRECTORY)
    )
    if enabled and storage != "precomputed":
        raise ValueError("dataset.highlight.storage must be precomputed")
    for label, name in (
        ("mask_directory_name", mask_directory),
        ("inpainted_directory_name", inpainted_directory),
    ):
        path = Path(name)
        if path.is_absolute() or len(path.parts) != 1 or name in ("", ".", ".."):
            raise ValueError("{} must be one safe directory name".format(label))
    detection = {key: value for key, value in values.items() if key not in _STORAGE_KEYS}
    detection["enabled"] = enabled
    return enabled, HighlightDetectionConfig(**detection), mask_directory, inpainted_directory


def precomputed_highlight_paths(
    sequence_root: str | Path,
    frame_name: str,
    mask_directory: str,
    inpainted_directory: str,
) -> Tuple[Path, Path]:
    root = Path(sequence_root)
    name = Path(frame_name).name
    return root / mask_directory / name, root / inpainted_directory / name


def highlight_manifest_payload(
    config: HighlightDetectionConfig,
    frame_count: int,
    mask_directory: str,
    inpainted_directory: str,
) -> Dict[str, Any]:
    return {
        "version": HIGHLIGHT_PRECOMPUTE_VERSION,
        "frame_count": int(frame_count),
        "mask_directory_name": mask_directory,
        "inpainted_directory_name": inpainted_directory,
        "detection_config": asdict(config),
    }


def validate_precomputed_highlight(
    sequence_root: str | Path,
    frame_count: int,
    config: HighlightDetectionConfig,
    mask_directory: str,
    inpainted_directory: str,
) -> None:
    root = Path(sequence_root)
    marker = root / HIGHLIGHT_MANIFEST_NAME
    if not marker.is_file():
        raise FileNotFoundError(
            "Precomputed highlight marker is missing: {}. Run precompute_highlights.py first.".format(
                marker
            )
        )
    try:
        actual = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("Invalid precomputed highlight marker {}".format(marker)) from error
    expected = highlight_manifest_payload(
        config, frame_count, mask_directory, inpainted_directory
    )
    if actual != expected:
        raise RuntimeError(
            "Precomputed highlight marker does not match current config: {}".format(marker)
        )


__all__ = [
    "HIGHLIGHT_MANIFEST_NAME",
    "highlight_manifest_payload",
    "parse_highlight_options",
    "precomputed_highlight_paths",
    "validate_precomputed_highlight",
]
