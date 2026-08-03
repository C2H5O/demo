"""Portable JSON manifest support for discovered SCARED temporal RGB sequences."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Union

from datasets.scared_discovery import discover_scared_sequences, expected_dataset_ids


LOGGER = logging.getLogger(__name__)


def build_scared_manifest(root: Union[str, Path], output_path: Union[str, Path], split: str = "train", frame_source: str = "auto") -> Dict[str, object]:
    """Discover sequences and save a path-portable JSON manifest."""
    root_path = Path(root).expanduser().resolve()
    sequences, malformed = discover_scared_sequences(root_path, split, frame_source, strict=False)
    manifest = {
        "format_version": 1,
        "split": split,
        "expected_dataset_ids": list(expected_dataset_ids(split)),
        "frame_source": frame_source,
        "sequences": [sequence.to_manifest_dict(root_path) for sequence in sequences],
        "malformed_sequences": malformed,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_scared_manifest(path: Union[str, Path]) -> Dict[str, object]:
    """Load and minimally validate a SCARED JSON manifest."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError("SCARED manifest not found: {}".format(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("Malformed SCARED manifest {}: {}".format(manifest_path, error)) from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sequences"), list):
        raise RuntimeError("Malformed SCARED manifest {}: expected an object with a sequences list".format(manifest_path))
    return manifest


def resolve_manifest_sequences(root: Union[str, Path], manifest: Dict[str, object], split: str) -> List[Dict[str, object]]:
    """Resolve portable manifest paths and enforce the requested official split."""
    root_path = Path(root).expanduser().resolve()
    expected = set(expected_dataset_ids(split))
    resolved: List[Dict[str, object]] = []
    seen_ids = set()
    for raw in manifest["sequences"]:
        if not isinstance(raw, dict):
            raise RuntimeError("Malformed SCARED manifest sequence entry: expected object, got {!r}".format(raw))
        try:
            dataset_id = int(raw["dataset_id"])
            frame_paths = raw["frame_paths"]
            frame_directory = raw["frame_directory"]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Malformed SCARED manifest sequence entry: {}".format(raw)) from error
        if dataset_id not in expected:
            continue
        if not isinstance(frame_paths, list) or not frame_paths:
            raise RuntimeError("Manifest sequence has no RGB frame paths: {}".format(raw.get("sequence_id", raw)))

        def resolve_path(value: object) -> Union[str, None]:
            if value is None:
                return None
            candidate = Path(str(value))
            return str(candidate if candidate.is_absolute() else root_path / candidate)

        item = dict(raw)
        item["frame_directory"] = resolve_path(frame_directory)
        item["frame_paths"] = [resolve_path(value) for value in frame_paths]
        for key in ("calibration_path", "depth_directory", "disparity_directory", "frame_data_directory", "reprojection_directory", "scene_points_directory", "point_cloud_path", "video_path"):
            item[key] = resolve_path(item.get(key))
        item["sequence_length"] = int(item.get("sequence_length", len(item["frame_paths"])))
        if item["sequence_length"] != len(item["frame_paths"]):
            raise RuntimeError("Manifest sequence length does not match frame paths: {}".format(item.get("sequence_id", item)))
        if len({Path(path).resolve() for path in item["frame_paths"]}) != len(item["frame_paths"]):
            raise RuntimeError("Manifest contains duplicate resolved frame paths: {}".format(item.get("sequence_id", item)))
        resolved.append(item)
        seen_ids.add(dataset_id)
    missing = sorted(expected - seen_ids)
    if missing:
        # A SCARED dataset may contain no keyframe with temporal RGB frames.
        # Keep all usable datasets instead of failing the complete training run.
        LOGGER.warning(
            "Skipping SCARED dataset IDs without usable RGB sequences for split %s: %s",
            split,
            missing,
        )
    return resolved
