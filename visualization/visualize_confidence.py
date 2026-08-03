"""Confidence-map conversion shared by future visualization commands."""

from __future__ import annotations

import numpy as np


def confidence_to_uint8(confidence: np.ndarray) -> np.ndarray:
    return np.round(np.clip(confidence, 0.0, 1.0) * 255.0).astype(np.uint8)
