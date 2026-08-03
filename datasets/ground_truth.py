"""SCARED ground-truth depth loading aligned by numeric frame ID."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_FRAME_ID = re.compile(r"(\d+)(?!.*\d)")
_DEPTH_SUFFIXES = {".png", ".tif", ".tiff", ".npy"}


def frame_id(path: str | Path) -> int:
    match = _FRAME_ID.search(Path(path).stem)
    if match is None:
        raise ValueError("Cannot extract numeric frame ID from {}".format(path))
    return int(match.group(1))


def index_depth_directory(directory: str | Path) -> Dict[int, Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    result: Dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in _DEPTH_SUFFIXES:
            continue
        identifier = frame_id(path)
        if identifier in result:
            raise RuntimeError(
                "Duplicate ground-truth frame ID {} in {}".format(
                    identifier, directory
                )
            )
        result[identifier] = path
    return result


def load_depth(path: Path, scale: float, channel: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(str(path))
    else:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required to load SCARED PNG/TIFF ground truth"
            ) from error
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError("Failed to read ground-truth depth {}".format(path))
    depth = np.asarray(depth)
    if depth.ndim == 3:
        if not 0 <= channel < depth.shape[-1]:
            raise ValueError(
                "Ground-truth channel {} is invalid for {} with shape {}".format(
                    channel, path, depth.shape
                )
            )
        depth = depth[..., channel]
    if depth.ndim != 2:
        raise ValueError("Ground-truth depth must be 2D, got {}".format(depth.shape))
    return depth.astype(np.float32) * float(scale)


def load_clip_ground_truth(
    frame_names: Sequence[str],
    candidate_directories: Iterable[Optional[str]],
    output_size: Tuple[int, int],
    scale: float = 1.0,
    channel: int = 0,
    required: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    index: Dict[int, Path] = {}
    selected_directory = None
    for value in candidate_directories:
        if not value:
            continue
        candidate = index_depth_directory(value)
        if candidate:
            index = candidate
            selected_directory = value
            break
    if not index:
        if required:
            raise FileNotFoundError(
                "No SCARED depth files found in candidate directories {}".format(
                    [value for value in candidate_directories if value]
                )
            )
        shape = (len(frame_names),) + output_size
        return torch.zeros(shape), torch.zeros(shape, dtype=torch.bool)

    depths: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for name in frame_names:
        identifier = frame_id(name)
        path = index.get(identifier)
        if path is None:
            if required:
                raise FileNotFoundError(
                    "No GT frame {} for {} in {}".format(
                        identifier, name, selected_directory
                    )
                )
            depth = torch.zeros(output_size)
            valid = torch.zeros(output_size, dtype=torch.bool)
        else:
            depth = torch.from_numpy(load_depth(path, scale, channel))
            valid = torch.isfinite(depth) & (depth > 0)
            if tuple(depth.shape) != output_size:
                depth = F.interpolate(
                    depth[None, None], size=output_size, mode="nearest"
                )[0, 0]
                valid = F.interpolate(
                    valid.float()[None, None], size=output_size, mode="nearest"
                )[0, 0].bool()
        depths.append(depth)
        masks.append(valid)
    return torch.stack(depths), torch.stack(masks)
