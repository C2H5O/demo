"""SCARED ground-truth depth I/O shared by spatial and temporal evaluation."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np



SCARED_MAX_DEPTH = 100.0
SCARED_GT_SCALE = 1.0 / 1000.0
SCARED_GT_DIRECTORY = "data/depth"
SUPPORTED_DEPTH_SUFFIXES = {".png", ".tif", ".tiff", ".npy"}
FRAME_ID_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def _opencv() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise RuntimeError("SCARED evaluation requires opencv-python") from error


def extract_frame_id(path: str | Path) -> int:
    match = FRAME_ID_PATTERN.search(Path(path).stem)
    if match is None:
        raise ValueError("Cannot extract a numeric frame ID from {}".format(path))
    return int(match.group(1))


def _build_unique_frame_map(paths: Iterable[Path], label: str) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    for path in paths:
        identifier = extract_frame_id(path)
        if identifier in result:
            raise RuntimeError(
                "Duplicate {} frame ID {}: {} and {}".format(
                    label, identifier, result[identifier], path
                )
            )
        result[identifier] = path
    return result


def _keyframe_directory(sequence: Dict[str, Any]) -> Path:
    value = sequence.get("keyframe_directory")
    if value:
        return Path(str(value))
    return Path(str(sequence["frame_directory"])).parent.parent


def _find_gt_depths(
    keyframe_directory: Path,
    relative_directory: str = SCARED_GT_DIRECTORY,
) -> Tuple[Path, Dict[int, Path]]:
    candidate = Path(relative_directory)
    directory = candidate if candidate.is_absolute() else keyframe_directory / candidate
    if not directory.is_dir():
        raise FileNotFoundError(
            "SCARED GT directory does not exist: {}".format(directory)
        )
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_DEPTH_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(
            "SCARED GT directory has no supported depth files: {}".format(directory)
        )
    return directory, _build_unique_frame_map(paths, "GT")


def load_scared_gt_depth(path: Path, channel: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(str(path))
    else:
        cv2 = _opencv()
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError("Failed to read GT depth {}".format(path))
    depth = np.asarray(depth)
    if depth.ndim == 3:
        if not 0 <= channel < depth.shape[-1]:
            raise ValueError(
                "GT depth channel {} is invalid for {} with shape {}".format(
                    channel, path, depth.shape
                )
            )
        depth = depth[..., channel]
    if depth.ndim != 2:
        raise ValueError(
            "Expected a 2D GT depth map at {}, got {}".format(path, depth.shape)
        )
    return depth.astype(np.float32) * SCARED_GT_SCALE
