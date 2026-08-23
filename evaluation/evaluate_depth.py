"""Endo3R depth protocol helpers used by the cross-clip evaluator."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from evaluation.depth_metrics import METRIC_NAMES, compute_errors


ENDO3R_WIDTH = 320
ENDO3R_HEIGHT = 256
ENDO3R_MIN_DEPTH = 0.0001
ENDO3R_MAX_DEPTH = 100.0
ENDO3R_GT_SCALE = 1.0 / 1000.0
ENDO3R_GT_DIRECTORY = "data/depth"
SUPPORTED_DEPTH_SUFFIXES = {".png", ".tif", ".tiff", ".npy"}
FRAME_ID_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def _opencv() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise RuntimeError("Endo3R evaluation requires opencv-python") from error


def extract_frame_id(path: str | Path) -> int:
    match = FRAME_ID_PATTERN.search(Path(path).stem)
    if match is None:
        raise ValueError("Cannot extract a numeric frame ID from {}".format(path))
    return int(match.group(1))


def _metric_dict(values: np.ndarray) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(METRIC_NAMES, values)}


def _distribution_stats(values: np.ndarray) -> Dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot summarize an empty/non-finite depth array")
    percentiles = np.percentile(finite, [0, 1, 5, 50, 95, 99, 100])
    return {
        "count": int(finite.size),
        "min": float(percentiles[0]),
        "p01": float(percentiles[1]),
        "p05": float(percentiles[2]),
        "median": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "max": float(percentiles[6]),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


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
    relative_directory: str = ENDO3R_GT_DIRECTORY,
) -> Tuple[Path, Dict[int, Path]]:
    candidate = Path(relative_directory)
    directory = candidate if candidate.is_absolute() else keyframe_directory / candidate
    if not directory.is_dir():
        raise FileNotFoundError(
            "Endo3R GT directory does not exist: {}".format(directory)
        )
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_DEPTH_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(
            "Endo3R GT directory has no supported depth files: {}".format(directory)
        )
    return directory, _build_unique_frame_map(paths, "GT")


def _load_endo3r_gt_depth(path: Path, channel: int) -> np.ndarray:
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
    return depth.astype(np.float32) * ENDO3R_GT_SCALE


def depth_evaluation(
    gt_depths: Sequence[np.ndarray],
    pred_depths: Sequence[np.ndarray],
    pred_masks: Optional[Sequence[np.ndarray]] = None,
    min_depth: float = ENDO3R_MIN_DEPTH,
    max_depth: float = ENDO3R_MAX_DEPTH,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Use one median scale for the complete SCARED sequence."""
    if len(gt_depths) != len(pred_depths):
        raise ValueError("GT/prediction frame count mismatch")
    if pred_masks is not None and len(pred_masks) != len(gt_depths):
        raise ValueError("pred_masks must have the same frame count as GT")
    cv2 = _opencv()
    gt_valid_frames: List[np.ndarray] = []
    pred_valid_frames: List[np.ndarray] = []
    for index, (gt_depth, pred_depth) in enumerate(zip(gt_depths, pred_depths)):
        mask = (gt_depth > min_depth) & (gt_depth < max_depth)
        if pred_masks is not None:
            height, width = gt_depth.shape[:2]
            mask &= cv2.resize(
                pred_masks[index].astype(np.uint8), (width, height)
            ) > 0.5
        if not np.any(mask):
            continue
        prediction = pred_depth[mask].astype(np.float64)
        if not np.all(np.isfinite(prediction)):
            raise ValueError("Prediction contains NaN or Inf")
        pred_valid_frames.append(prediction)
        gt_valid_frames.append(gt_depth[mask].astype(np.float64))
    if not gt_valid_frames:
        raise RuntimeError("No valid depth pixels remained for Endo3R evaluation")
    prediction_median = float(np.median(np.concatenate(pred_valid_frames)))
    if not np.isfinite(prediction_median) or prediction_median <= 0:
        raise RuntimeError("Endo3R scene scaling requires positive prediction depth")
    ratio = float(np.median(np.concatenate(gt_valid_frames)) / prediction_median)
    errors = np.asarray(
        [
            compute_errors(gt, np.clip(pred * ratio, min_depth, max_depth))
            for gt, pred in zip(gt_valid_frames, pred_valid_frames)
        ],
        dtype=np.float64,
    )
    return errors, errors.mean(0), ratio


def _evaluate_sequence(
    sequence: Dict[str, Any],
    depth_sums: Dict[int, np.ndarray],
    depth_counts: Dict[int, int],
    gt_channel: int,
    gt_relative_directory: str,
    require_all_gt: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    cv2 = _opencv()
    gt_directory, gt_by_id = _find_gt_depths(
        _keyframe_directory(sequence), gt_relative_directory
    )
    prediction_ids, gt_ids = set(depth_sums), set(gt_by_id)
    matched_ids = sorted(prediction_ids & gt_ids)
    missing_prediction_ids = sorted(gt_ids - prediction_ids)
    prediction_without_gt_ids = sorted(prediction_ids - gt_ids)
    if require_all_gt and missing_prediction_ids:
        raise RuntimeError(
            "Endo3R evaluation requires every GT frame in {}. Missing IDs: {}".format(
                sequence["sequence_id"], missing_prediction_ids[:20]
            )
        )
    if not matched_ids:
        raise RuntimeError("No prediction/GT frame IDs match")
    gt_depths: List[np.ndarray] = []
    pred_depths: List[np.ndarray] = []
    for identifier in matched_ids:
        gt = _load_endo3r_gt_depth(gt_by_id[identifier], gt_channel)
        pred = depth_sums[identifier] / float(depth_counts[identifier])
        gt_depths.append(
            cv2.resize(gt, (ENDO3R_WIDTH, ENDO3R_HEIGHT), interpolation=cv2.INTER_NEAREST)
        )
        pred_depths.append(
            cv2.resize(
                pred.astype(np.float32),
                (ENDO3R_WIDTH, ENDO3R_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            )
        )
    errors, mean_errors, ratio = depth_evaluation(gt_depths, pred_depths)
    valid_masks = [
        (depth > ENDO3R_MIN_DEPTH) & (depth < ENDO3R_MAX_DEPTH)
        for depth in gt_depths
    ]
    gt_valid = np.concatenate([depth[mask] for depth, mask in zip(gt_depths, valid_masks)])
    pred_valid = np.concatenate([depth[mask] for depth, mask in zip(pred_depths, valid_masks)])
    result = {
        "sequence_id": str(sequence["sequence_id"]),
        "gt_directory": str(gt_directory),
        "predicted_frame_count": len(prediction_ids),
        "gt_frame_count": len(gt_ids),
        "matched_frame_count": len(matched_ids),
        "evaluated_frame_count": len(errors),
        "missing_prediction_count": len(missing_prediction_ids),
        "prediction_without_gt_count": len(prediction_without_gt_ids),
        "missing_prediction_ids_preview": missing_prediction_ids[:20],
        "prediction_without_gt_ids_preview": prediction_without_gt_ids[:20],
        "scene_median_scaling_ratio": ratio,
        "depth_diagnostics": {
            "gt_valid_m": _distribution_stats(gt_valid),
            "prediction_raw": _distribution_stats(pred_valid),
            "prediction_aligned_m": _distribution_stats(
                np.clip(pred_valid * ratio, ENDO3R_MIN_DEPTH, ENDO3R_MAX_DEPTH)
            ),
        },
        "metrics": _metric_dict(mean_errors),
    }
    return errors, result


__all__ = [
    "ENDO3R_MAX_DEPTH",
    "ENDO3R_MIN_DEPTH",
    "_evaluate_sequence",
    "_find_gt_depths",
    "_keyframe_directory",
    "_load_endo3r_gt_depth",
    "depth_evaluation",
    "extract_frame_id",
]
