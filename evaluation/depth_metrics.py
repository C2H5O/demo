"""Depth metrics used verbatim by Endo3R's depth evaluation protocol."""

import numpy as np


METRIC_NAMES = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")


def compute_errors(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    threshold = np.maximum(gt / pred, pred / gt)
    a1 = np.mean(threshold < 1.25)
    a2 = np.mean(threshold < 1.25**2)
    a3 = np.mean(threshold < 1.25**3)
    difference = gt - pred
    rmse = np.sqrt(np.mean(difference**2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    abs_rel = np.mean(np.abs(difference) / gt)
    sq_rel = np.mean((difference**2) / gt)
    return np.asarray(
        (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3), dtype=np.float64
    )


__all__ = ["METRIC_NAMES", "compute_errors"]
