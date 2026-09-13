"""Record-only per-window scale diagnostics for online Teacher evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from evaluation.evaluate_vda import _resized_gt
from evaluation.scared_gt import SCARED_MAX_DEPTH, extract_frame_id


class TeacherWindowScaleDiagnostics:
    """Measure raw-window depth scale without changing inference or stitching."""

    def __init__(
        self,
        sequence: Dict[str, Any],
        gt_depths: Tuple[Any, Dict[int, Any]],
        gt_channel: int,
    ) -> None:
        self.sequence = sequence
        self.gt_by_id = gt_depths[1]
        self.gt_channel = int(gt_channel)
        self.records: List[Dict[str, Any]] = []

    def __call__(self, window_index: int, positions: List[int], depth: np.ndarray) -> None:
        numerator = 0.0
        denominator = 0.0
        valid_pixel_count = 0
        frame_ids = []
        for offset, position in enumerate(positions):
            frame_id = extract_frame_id(self.sequence["frame_paths"][position])
            frame_ids.append(frame_id)
            gt_path = self.gt_by_id.get(frame_id)
            if gt_path is None:
                continue
            gt = _resized_gt(
                gt_path, self.gt_channel, int(depth.shape[-2]), int(depth.shape[-1])
            )
            prediction = depth[offset].astype(np.float64, copy=False)
            valid = (
                np.isfinite(prediction)
                & (prediction > 1.0e-3)
                & (gt > 1.0e-3)
                & (gt < SCARED_MAX_DEPTH)
            )
            if not np.any(valid):
                continue
            predicted_values = prediction[valid]
            gt_values = gt[valid].astype(np.float64, copy=False)
            numerator += float(np.dot(predicted_values, gt_values))
            denominator += float(np.dot(predicted_values, predicted_values))
            valid_pixel_count += int(valid.sum())
        scale = numerator / denominator if denominator > 0.0 else None
        if scale is not None and (not np.isfinite(scale) or scale <= 0.0):
            scale = None
        self.records.append(
            {
                "window_index": int(window_index),
                "frame_ids": frame_ids,
                "scale_to_gt": float(scale) if scale is not None else None,
                "valid_pixel_count": valid_pixel_count,
                "definition": "argmin_s sum((s * raw_window_depth - gt_depth)^2), s > 0",
                "record_only": True,
                "applied_to_prediction": False,
                "used_for_stitching": False,
            }
        )


__all__ = ["TeacherWindowScaleDiagnostics"]
