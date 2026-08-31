"""Streaming Video-Depth-Anything protocol helpers for cross-clip evaluation."""

from __future__ import annotations

import gc
import importlib
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from datasets.scared_discovery import KEYFRAME_PATTERN, extract_dataset_id
from evaluation.evaluate_depth import (
    ENDO3R_GT_DIRECTORY,
    ENDO3R_MAX_DEPTH,
    _find_gt_depths,
    _keyframe_directory,
    _load_endo3r_gt_depth,
    extract_frame_id,
)


VDA_METRIC_NAMES = (
    "abs_relative_difference",
    "rmse_linear",
    "delta1_acc",
)


def _keyframe_identity(value: str) -> Tuple[Any, ...]:
    """Normalize keyframe spelling while preserving its numeric identity."""
    match = KEYFRAME_PATTERN.match(value)
    suffix = match.group(1) if match else value
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", suffix)
        if part
    )


def _find_gt_keyframe(sequence: Dict[str, Any], gt_root: Path) -> Path:
    """Map a processed RGB sequence to its keyframe in a separate GT tree."""
    if not gt_root.is_dir():
        raise FileNotFoundError("Configured SCARED GT root does not exist: {}".format(gt_root))
    dataset_id = int(sequence["dataset_id"])
    dataset_directories = [
        path
        for path in gt_root.iterdir()
        if path.is_dir() and extract_dataset_id(path.name) == dataset_id
    ]
    if len(dataset_directories) != 1:
        raise FileNotFoundError(
            "Expected one dataset directory for dataset {} under {}; found {}".format(
                dataset_id, gt_root, [str(path) for path in dataset_directories]
            )
        )
    keyframe_id = str(sequence["keyframe_id"])
    identity = _keyframe_identity(keyframe_id)
    keyframes = [
        path
        for path in dataset_directories[0].iterdir()
        if path.is_dir()
        and KEYFRAME_PATTERN.match(path.name)
        and _keyframe_identity(path.name) == identity
    ]
    if len(keyframes) != 1:
        raise FileNotFoundError(
            "Expected one GT keyframe matching {} under {}; found {}".format(
                keyframe_id,
                dataset_directories[0],
                [str(path) for path in keyframes],
            )
        )
    return keyframes[0]


def _opencv() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise RuntimeError("VDA evaluation requires opencv-python") from error


def _student_depth_to_vda_disparity(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if not np.all(np.isfinite(depth)):
        raise ValueError("Student depth contains NaN or Inf")
    return np.reciprocal(np.clip(depth, 1e-3, None))


def abs_relative_difference(output, target, valid_mask=None):
    values = torch.abs(output - target) / target
    if valid_mask is not None:
        values[~valid_mask] = 0
        count = valid_mask.sum((-1, -2))
    else:
        count = output.shape[-1] * output.shape[-2]
    return (values.sum((-1, -2)) / count).mean()


def rmse_linear(output, target, valid_mask=None):
    difference = output - target
    if valid_mask is not None:
        difference[~valid_mask] = 0
        count = valid_mask.sum((-1, -2))
    else:
        count = output.shape[-1] * output.shape[-2]
    return torch.sqrt(difference.square().sum((-1, -2)) / count).mean()


def delta1_acc(output, target, valid_mask=None):
    threshold = torch.maximum(output / target, target / output)
    hits = (threshold.cpu() < 1.25).to(torch.float32)
    if valid_mask is not None:
        hits[~valid_mask.cpu()] = 0
        count = valid_mask.sum((-1, -2)).cpu()
    else:
        count = output.shape[-1] * output.shape[-2]
    return (hits.sum((-1, -2)) / count).mean()


def depth_to_disparity(depth: np.ndarray) -> np.ndarray:
    disparity = np.zeros_like(depth)
    valid = depth > 0
    disparity[valid] = 1.0 / depth[valid]
    return disparity


class _SequencePredictionSpool:
    """Disk-backed sums for overlapping predictions of one sequence."""

    def __init__(self, directory: Path, frame_count: int, height: int, width: int) -> None:
        handle = tempfile.NamedTemporaryFile(
            prefix=".vda_spool_", suffix=".dat", dir=str(directory), delete=False
        )
        self.path = Path(handle.name)
        handle.close()
        self.frame_count = frame_count
        self.height = height
        self.width = width
        self.sums = np.memmap(
            self.path,
            mode="w+",
            dtype=np.float32,
            shape=(frame_count, height, width),
        )
        self.counts = np.zeros(frame_count, dtype=np.uint16)

    def add(self, frame_indices: Sequence[int], disparities: np.ndarray) -> None:
        cv2 = _opencv()
        for offset, frame_index in enumerate(frame_indices):
            resized = cv2.resize(
                disparities[offset], (self.width, self.height)
            ).astype(np.float32, copy=False)
            self.sums[frame_index] += resized
            self.counts[frame_index] += 1

    def prediction(self, frame_index: int) -> np.ndarray:
        count = int(self.counts[frame_index])
        if count <= 0:
            raise RuntimeError("No prediction for frame {}".format(frame_index))
        return np.asarray(self.sums[frame_index] / float(count), dtype=np.float32)

    def flush(self) -> None:
        self.sums.flush()

    def close(self) -> None:
        self.sums.flush()
        mmap_object = getattr(self.sums, "_mmap", None)
        if mmap_object is not None:
            mmap_object.close()
        self.sums = None
        gc.collect()
        self.path.unlink(missing_ok=True)


def _resized_gt(path: Path, channel: int, height: int, width: int) -> np.ndarray:
    cv2 = _opencv()
    return cv2.resize(
        _load_endo3r_gt_depth(path, channel),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def _frame_pairs(
    sequence: Dict[str, Any],
    spool: _SequencePredictionSpool,
    gt_by_id: Dict[int, Path],
    require_all_gt: bool,
) -> Tuple[List[Tuple[int, int, Path]], List[int]]:
    rgb_index_by_id: Dict[int, int] = {}
    for index, frame_path in enumerate(sequence["frame_paths"]):
        identifier = extract_frame_id(frame_path)
        if identifier in rgb_index_by_id:
            raise RuntimeError("Duplicate RGB frame ID {}".format(identifier))
        rgb_index_by_id[identifier] = index
    predicted_ids = {
        identifier
        for identifier, index in rgb_index_by_id.items()
        if spool.counts[index] > 0
    }
    gt_ids = set(gt_by_id)
    matched_ids = sorted(predicted_ids & gt_ids)
    missing = sorted(gt_ids - predicted_ids)
    if require_all_gt and missing:
        raise RuntimeError("VDA scoring is missing GT frame predictions: {}".format(missing[:20]))
    if not matched_ids:
        raise RuntimeError("No prediction/GT frame IDs match")
    return [
        (identifier, rgb_index_by_id[identifier], gt_by_id[identifier])
        for identifier in matched_ids
    ], missing


def _streaming_scale_shift(
    pairs: Sequence[Tuple[int, int, Path]],
    spool: _SequencePredictionSpool,
    gt_channel: int,
    sequence_id: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    reduced = None
    valid_pixel_count = 0
    for _, frame_index, gt_path in pairs:
        gt = _resized_gt(gt_path, gt_channel, spool.height, spool.width)
        prediction = np.clip(spool.prediction(frame_index), 1e-3, None)
        valid = (gt > 1e-3) & (gt < ENDO3R_MAX_DEPTH)
        if not np.any(valid):
            continue
        target = 1.0 / (gt[valid].reshape(-1, 1).astype(np.float64) + 1e-8)
        predicted = prediction[valid].reshape(-1, 1).astype(np.float64)
        augmented = np.concatenate([predicted, np.ones_like(predicted), target], axis=-1)
        _, frame_reduction = np.linalg.qr(augmented, mode="reduced")
        if reduced is None:
            reduced = frame_reduction
        else:
            _, reduced = np.linalg.qr(
                np.concatenate([reduced, frame_reduction], axis=0), mode="reduced"
            )
        valid_pixel_count += int(valid.sum())
    if reduced is None:
        raise RuntimeError("No valid GT pixels remain for {}".format(sequence_id))
    scale, shift = np.linalg.lstsq(reduced[:, :2], reduced[:, 2:3], rcond=None)[0]
    return scale, shift, valid_pixel_count


def _streaming_metrics(
    pairs: Sequence[Tuple[int, int, Path]],
    spool: _SequencePredictionSpool,
    gt_channel: int,
    scale: np.ndarray,
    shift: np.ndarray,
    sequence_id: str,
) -> Tuple[List[float], int]:
    metric_sums = np.zeros(len(VDA_METRIC_NAMES), dtype=np.float64)
    valid_frame_count = 0
    functions = (abs_relative_difference, rmse_linear, delta1_acc)
    for _, frame_index, gt_path in pairs:
        gt = _resized_gt(gt_path, gt_channel, spool.height, spool.width)
        valid = (gt > 1e-3) & (gt < ENDO3R_MAX_DEPTH)
        if not np.any(valid):
            continue
        disparity = np.clip(spool.prediction(frame_index), 1e-3, None)
        aligned = np.clip(scale * disparity + shift, 1e-3, None)
        prediction = np.clip(depth_to_disparity(aligned), 1e-3, ENDO3R_MAX_DEPTH)
        pred_tensor = torch.from_numpy(prediction[None])
        gt_tensor = torch.from_numpy(gt[None])
        valid_tensor = torch.from_numpy(valid[None])
        for index, function in enumerate(functions):
            metric_sums[index] += function(pred_tensor, gt_tensor, valid_tensor).item()
        valid_frame_count += 1
    if valid_frame_count == 0:
        raise RuntimeError("No valid frames remain for {}".format(sequence_id))
    return (metric_sums / valid_frame_count).tolist(), valid_frame_count


def _evaluate_sequence(
    sequence: Dict[str, Any],
    spool: _SequencePredictionSpool,
    gt_channel: int,
    gt_depths: Tuple[Path, Dict[int, Path]],
    require_all_gt: bool,
) -> Dict[str, Any]:
    gt_directory, gt_by_id = gt_depths
    pairs, missing = _frame_pairs(sequence, spool, gt_by_id, require_all_gt)
    scale, shift, valid_pixels = _streaming_scale_shift(
        pairs, spool, gt_channel, str(sequence["sequence_id"])
    )
    metrics, valid_frames = _streaming_metrics(
        pairs, spool, gt_channel, scale, shift, str(sequence["sequence_id"])
    )
    return {
        "sequence_id": str(sequence["sequence_id"]),
        "gt_directory": str(gt_directory),
        "matched_frame_count": len(pairs),
        "evaluated_frame_count": valid_frames,
        "valid_pixel_count": valid_pixels,
        "evaluation_size": [spool.width, spool.height],
        "evaluation_shape_hxw": [spool.height, spool.width],
        "missing_prediction_count": len(missing),
        "missing_prediction_ids_preview": missing[:20],
        "disparity_scale": float(np.asarray(scale).reshape(-1)[0]),
        "disparity_shift": float(np.asarray(shift).reshape(-1)[0]),
        "metrics": {name: float(value) for name, value in zip(VDA_METRIC_NAMES, metrics)},
    }


def _find_sequence_gt_depths(
    sequence: Dict[str, Any],
    eval_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
) -> Tuple[Path, Dict[int, Path]]:
    keyframe_directories: List[Path] = []
    gt_root = eval_config.get("gt_root", dataset_config.get("gt_root"))
    if gt_root:
        keyframe_directories.append(
            _find_gt_keyframe(sequence, Path(str(gt_root)).expanduser())
        )
    processed_keyframe_directory = _keyframe_directory(sequence)
    if processed_keyframe_directory not in keyframe_directories:
        keyframe_directories.append(processed_keyframe_directory)
    candidates: List[Path] = []
    explicit = eval_config.get("gt_relative_directory")
    if explicit:
        candidates.append(Path(str(explicit)))
    for key in ("depth_directory", "scene_points_directory"):
        value = sequence.get(key)
        if value:
            candidates.append(Path(str(value)))
    if not candidates:
        candidates.append(Path(ENDO3R_GT_DIRECTORY))
    checked: List[str] = []
    for keyframe_directory in keyframe_directories:
        for candidate in candidates:
            resolved = candidate if candidate.is_absolute() else keyframe_directory / candidate
            checked.append(str(resolved))
            try:
                return _find_gt_depths(keyframe_directory, str(resolved))
            except FileNotFoundError:
                continue
    raise FileNotFoundError(
        "No supported depth GT found for {}. Checked: {}".format(
            sequence.get("sequence_id"), checked
        )
    )


__all__ = [
    "VDA_METRIC_NAMES",
    "_SequencePredictionSpool",
    "_evaluate_sequence",
    "_find_sequence_gt_depths",
    "_streaming_metrics",
    "_streaming_scale_shift",
    "_student_depth_to_vda_disparity",
]
