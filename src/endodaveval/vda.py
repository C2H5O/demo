"""Unified VDA spatial evaluation on a fixed 256x320 disparity grid."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

from endodaveval.config import EVALUATION_RESOLUTION_HW, SCARED_MAX_DEPTH


METRIC_NAMES = ("abs_relative_difference", "rmse_linear", "delta1_acc")


class EvaluationError(RuntimeError):
    pass


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output[valid_mask] if valid_mask is not None else output
    actual_target = target[valid_mask] if valid_mask is not None else target
    return torch.mean(torch.abs(actual_output - actual_target) / actual_target)


def rmse_linear(output, target, valid_mask=None):
    actual_output = output[valid_mask] if valid_mask is not None else output
    actual_target = target[valid_mask] if valid_mask is not None else target
    return torch.sqrt(torch.mean((actual_output - actual_target) ** 2))


def delta1_acc(output, target, valid_mask=None):
    actual_output = output[valid_mask] if valid_mask is not None else output
    actual_target = target[valid_mask] if valid_mask is not None else target
    ratio = torch.maximum(actual_output / actual_target, actual_target / actual_output)
    return torch.mean((ratio < 1.25).to(dtype=output.dtype))


def _load_depth(path: Path, channel: int = 0) -> np.ndarray:
    if path.suffix.casefold() == ".npy":
        value = np.load(str(path))
    else:
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise EvaluationError("Could not read depth: {}".format(path))
    value = np.asarray(value)
    if value.ndim == 3:
        value = value[..., channel]
    if value.ndim != 2:
        raise EvaluationError("Depth must be HxW: {}".format(path))
    value = value.astype(np.float32, copy=False)
    if not np.isfinite(value).all():
        raise EvaluationError("Depth contains NaN or Inf: {}".format(path))
    return value


def load_ground_truth(
    path: Path,
    scale: float,
    channel: int,
    target_shape: Tuple[int, int] = EVALUATION_RESOLUTION_HW,
) -> np.ndarray:
    depth = _load_depth(path, channel) * float(scale)
    return cv2.resize(
        depth,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32, copy=False)


def depth_to_disparity(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    disparity = np.zeros_like(depth)
    valid = depth > 0
    disparity[valid] = 1.0 / depth[valid]
    return disparity


def prediction_depth_to_evaluation_disparity(
    predicted_depth: np.ndarray,
    target_shape: Tuple[int, int] = EVALUATION_RESOLUTION_HW,
) -> np.ndarray:
    """Required order: reciprocal depth, then bilinear disparity resize."""
    disparity = depth_to_disparity(predicted_depth)
    return cv2.resize(
        disparity,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32, copy=False)


def evaluate_sequence_arrays(
    predicted_disparity_by_id: Mapping[int, np.ndarray],
    ground_truth_depth_by_id: Mapping[int, np.ndarray],
    frame_ids: Sequence[int],
    min_depth: float = 0.001,
    max_depth: float = 100.0,
):
    if not frame_ids:
        raise EvaluationError("Cannot evaluate an empty sequence")
    gt_values = []
    pred_values = []
    valid_pixel_count = 0
    for identifier in frame_ids:
        prediction = np.asarray(predicted_disparity_by_id[identifier], dtype=np.float32)
        ground_truth = np.asarray(ground_truth_depth_by_id[identifier], dtype=np.float32)
        if prediction.shape != EVALUATION_RESOLUTION_HW or ground_truth.shape != EVALUATION_RESOLUTION_HW:
            raise EvaluationError("Spatial evaluation requires 256x320 prediction and GT")
        valid = (ground_truth > min_depth) & (ground_truth < max_depth)
        if not np.any(valid):
            continue
        gt_values.append(ground_truth[valid])
        pred_values.append(np.clip(prediction, 1e-3, None)[valid])
        valid_pixel_count += int(valid.sum())
    if not gt_values:
        raise EvaluationError("No valid ground-truth pixels in sequence")
    gt_disparity = 1.0 / (
        np.concatenate(gt_values).reshape(-1, 1).astype(np.float64) + 1e-8
    )
    predicted_disparity = (
        np.concatenate(pred_values).reshape(-1, 1).astype(np.float64)
    )
    matrix = np.concatenate(
        [predicted_disparity, np.ones_like(predicted_disparity)], axis=-1
    )
    scale, shift = np.linalg.lstsq(matrix, gt_disparity, rcond=None)[0]

    sums = np.zeros(len(METRIC_NAMES), dtype=np.float64)
    valid_frame_count = 0
    aligned_depth_by_id: Dict[int, np.ndarray] = {}
    functions = (abs_relative_difference, rmse_linear, delta1_acc)
    for identifier in frame_ids:
        prediction = np.clip(predicted_disparity_by_id[identifier], 1e-3, None)
        ground_truth = ground_truth_depth_by_id[identifier]
        valid = (ground_truth > min_depth) & (ground_truth < max_depth)
        aligned_disparity = np.clip(scale * prediction + shift, 1e-3, None)
        predicted_depth = np.clip(
            np.reciprocal(aligned_disparity), 1e-3, max_depth
        )
        aligned_depth_by_id[identifier] = predicted_depth
        if not np.any(valid):
            continue
        prediction_tensor = torch.from_numpy(predicted_depth[None])
        target_tensor = torch.from_numpy(ground_truth[None])
        valid_tensor = torch.from_numpy(valid[None])
        for index, function in enumerate(functions):
            sums[index] += function(prediction_tensor, target_tensor, valid_tensor).item()
        valid_frame_count += 1
    if valid_frame_count == 0:
        raise EvaluationError("No valid frames remain in sequence")
    metrics = sums / valid_frame_count
    result = {
        "metrics": {name: float(value) for name, value in zip(METRIC_NAMES, metrics)},
        "alignment": {
            "domain": "disparity",
            "scope": "one scale and shift for the complete sequence",
            "implementation": "numpy.linalg.lstsq float64",
            "scale": float(np.asarray(scale).reshape(-1)[0]),
            "shift": float(np.asarray(shift).reshape(-1)[0]),
        },
        "frame_ids": list(frame_ids),
        "frame_count": len(frame_ids),
        "valid_frame_count": valid_frame_count,
        "valid_pixel_count": valid_pixel_count,
        "evaluation_shape_hxw": list(EVALUATION_RESOLUTION_HW),
        "prediction_interpolation": "depth_to_reciprocal_disparity_then_bilinear",
        "ground_truth_interpolation": "nearest_neighbor_depth",
        "valid_depth_range_m": [min_depth, max_depth],
    }
    return result, aligned_depth_by_id


def evaluate_prediction_files(
    native_depth_by_id: Mapping[int, Path],
    ground_truth_by_id: Mapping[int, Path],
    frame_ids: Sequence[int],
    ground_truth_scale: float,
    ground_truth_channel: int,
):
    predicted = {
        identifier: prediction_depth_to_evaluation_disparity(
            _load_depth(native_depth_by_id[identifier])
        )
        for identifier in frame_ids
    }
    ground_truth = {
        identifier: load_ground_truth(
            ground_truth_by_id[identifier],
            ground_truth_scale,
            ground_truth_channel,
        )
        for identifier in frame_ids
    }
    return evaluate_sequence_arrays(predicted, ground_truth, frame_ids)
