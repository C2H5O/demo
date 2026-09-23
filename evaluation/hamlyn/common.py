"""Single common spatial evaluator used by every Hamlyn method."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np

from evaluation.hamlyn.cache import atomic_write_json, load_cache
from evaluation.hamlyn.constants import (
    EVALUATION_RESOLUTION_HW,
    HAMLYN_GT_SCALE,
    HAMLYN_MAX_DEPTH,
    HAMLYN_MIN_DEPTH,
    HAMLYN_SEQUENCE_IDS,
    INFERENCE_RESOLUTIONS_HW,
    METHOD_LABELS,
    METRIC_AGGREGATION,
    SCALE_ALIGNMENT,
)
from evaluation.hamlyn.data import SequenceRecord
from evaluation.hamlyn.gt import load_hamlyn_gt_depth


METRIC_NAMES = (
    "abs_relative_difference",
    "rmse_linear",
    "delta1_acc",
)


class EvaluationError(RuntimeError):
    pass


def resize_predicted_depth(
    depth: np.ndarray,
    target_shape: Tuple[int, int] = EVALUATION_RESOLUTION_HW,
) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise EvaluationError("Predicted depth must be finite and HxW")
    if tuple(value.shape) != tuple(target_shape):
        value = cv2.resize(
            value,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)
    # Match the audited DA3/Endo3R depth boundary after the required depth-space
    # resize. Official Endo3R Z-depth can contain non-positive background values.
    return np.clip(value, 1e-3, None).astype(np.float32, copy=False)


def prediction_to_disparity(
    prediction: np.ndarray,
    representation: str,
    target_shape: Tuple[int, int] = EVALUATION_RESOLUTION_HW,
) -> np.ndarray:
    value = np.asarray(prediction, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise EvaluationError("Prediction must be a finite HxW array")
    if representation == "depth":
        # Endo3R is explicitly resized in depth space before the common fit.
        value = resize_predicted_depth(value, target_shape)
        return np.reciprocal(value).astype(np.float32, copy=False)
    if representation != "disparity":
        raise EvaluationError("Unsupported prediction representation: {}".format(representation))
    if np.any(value <= 0):
        raise EvaluationError("Predicted disparity must be positive")
    if tuple(value.shape) != tuple(target_shape):
        value = cv2.resize(
            value,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return value.astype(np.float32, copy=False)


def fit_sequence_disparity_scale_shift(
    predicted_disparities: Sequence[np.ndarray],
    ground_truth_depths: Sequence[np.ndarray],
) -> Tuple[float, float, int]:
    if len(predicted_disparities) != len(ground_truth_depths) or not predicted_disparities:
        raise EvaluationError("Scale/shift fitting requires paired non-empty sequences")
    predicted_values = []
    target_values = []
    valid_pixel_count = 0
    for prediction, ground_truth in zip(predicted_disparities, ground_truth_depths):
        if prediction.shape != ground_truth.shape:
            raise EvaluationError("Prediction/GT shapes differ during sequence alignment")
        valid = (ground_truth > HAMLYN_MIN_DEPTH) & (ground_truth < HAMLYN_MAX_DEPTH)
        if not np.any(valid):
            continue
        predicted_values.append(np.clip(prediction[valid], 1e-3, None))
        target_values.append(ground_truth[valid])
        valid_pixel_count += int(valid.sum())
    if not predicted_values:
        raise EvaluationError("No valid Hamlyn GT pixels remain for alignment")
    predicted = np.concatenate(predicted_values).reshape(-1, 1).astype(np.float64)
    target_depth = np.concatenate(target_values).reshape(-1, 1).astype(np.float64)
    target = np.reciprocal(target_depth)
    matrix = np.concatenate((predicted, np.ones_like(predicted)), axis=1)
    scale, shift = np.linalg.lstsq(matrix, target, rcond=None)[0]
    scale_value = float(np.asarray(scale).reshape(-1)[0])
    shift_value = float(np.asarray(shift).reshape(-1)[0])
    if not np.isfinite([scale_value, shift_value]).all():
        raise EvaluationError("Sequence disparity scale/shift fit is non-finite")
    return scale_value, shift_value, valid_pixel_count


def _frame_metrics(
    prediction_depth: np.ndarray, ground_truth_depth: np.ndarray, valid: np.ndarray
) -> Dict[str, float]:
    prediction = prediction_depth[valid].astype(np.float64)
    target = ground_truth_depth[valid].astype(np.float64)
    ratio = np.maximum(prediction / target, target / prediction)
    return {
        "abs_relative_difference": float(np.mean(np.abs(prediction - target) / target)),
        "rmse_linear": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "delta1_acc": float(np.mean(ratio < 1.25)),
    }


def macro_mean(sequence_results: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not sequence_results:
        raise EvaluationError("Cannot aggregate an empty result set")
    return {
        name: float(np.mean([float(item["metrics"][name]) for item in sequence_results]))
        for name in METRIC_NAMES
    }


def evaluate_sequence(
    method: str,
    record: SequenceRecord,
    metadata: Mapping[str, Any],
    prediction_paths: Mapping[int, Path],
) -> Dict[str, Any]:
    if tuple(metadata["frame_ids"]) != record.frame_ids:
        raise EvaluationError("Cached frame IDs do not match discovered RGB/GT IDs")
    representation = str(metadata["prediction_representation"])
    predicted_disparities = []
    ground_truth_depths = []
    for identifier in record.frame_ids:
        prediction = np.load(str(prediction_paths[identifier]), allow_pickle=False)
        predicted_disparities.append(
            prediction_to_disparity(prediction, representation)
        )
        ground_truth_depths.append(load_hamlyn_gt_depth(record.depth_by_id[identifier]))
    scale, shift, valid_pixel_count = fit_sequence_disparity_scale_shift(
        predicted_disparities, ground_truth_depths
    )
    sums = {name: 0.0 for name in METRIC_NAMES}
    valid_frame_count = 0
    for prediction, ground_truth in zip(predicted_disparities, ground_truth_depths):
        valid = (ground_truth > HAMLYN_MIN_DEPTH) & (ground_truth < HAMLYN_MAX_DEPTH)
        if not np.any(valid):
            continue
        aligned_disparity = np.clip(scale * prediction + shift, 1e-3, None)
        predicted_depth = np.clip(
            np.reciprocal(aligned_disparity),
            HAMLYN_MIN_DEPTH,
            HAMLYN_MAX_DEPTH,
        )
        values = _frame_metrics(predicted_depth, ground_truth, valid)
        for name in METRIC_NAMES:
            sums[name] += values[name]
        valid_frame_count += 1
    if valid_frame_count == 0:
        raise EvaluationError(
            "No valid frames remain for Hamlyn sequence {}".format(record.sequence_id)
        )
    return {
        "sequence_id": record.sequence_id,
        "frame_ids": list(record.frame_ids),
        "frame_count": record.frame_count,
        "valid_frame_count": valid_frame_count,
        "valid_pixel_count": valid_pixel_count,
        "disparity_scale": scale,
        "disparity_shift": shift,
        "metrics": {name: sums[name] / valid_frame_count for name in METRIC_NAMES},
    }


def evaluate_method(
    method: str,
    records: Sequence[SequenceRecord],
    output_root: Path,
) -> Dict[str, Any]:
    if tuple(record.sequence_id for record in records) != HAMLYN_SEQUENCE_IDS:
        raise EvaluationError("Common evaluator requires the exact ordered 22-sequence split")
    sequence_results = []
    for record in records:
        metadata, prediction_paths = load_cache(output_root, method, record)
        sequence_results.append(
            evaluate_sequence(method, record, metadata, prediction_paths)
        )
    metrics = macro_mean(sequence_results)
    result = {
        "schema_version": 1,
        "method": METHOD_LABELS[method],
        "method_id": method,
        "dataset": "Hamlyn",
        "sequence_ids": list(HAMLYN_SEQUENCE_IDS),
        "sequence_count": len(HAMLYN_SEQUENCE_IDS),
        "inference_resolution_hw": list(INFERENCE_RESOLUTIONS_HW[method]),
        "evaluation_resolution_hw": list(EVALUATION_RESOLUTION_HW),
        "gt_scale": HAMLYN_GT_SCALE,
        "min_depth": HAMLYN_MIN_DEPTH,
        "max_depth": HAMLYN_MAX_DEPTH,
        "scale_alignment": SCALE_ALIGNMENT,
        "metric_aggregation": METRIC_AGGREGATION,
        "tae": {"enabled": False},
        "metrics": metrics,
        "overall": metrics,
        "per_sequence": sequence_results,
    }
    output = output_root / method / "evaluation.json"
    atomic_write_json(output, result)
    print("wrote {}".format(output), flush=True)
    return result
