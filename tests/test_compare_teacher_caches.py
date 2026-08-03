from __future__ import annotations

import numpy as np

from compare_teacher_caches import (
    FrameAccumulator,
    _evaluate_frame,
    _summarize,
)


def _accumulator(depth: np.ndarray) -> FrameAccumulator:
    accumulator = FrameAccumulator.empty(tuple(depth.shape))
    accumulator.add(
        depth=depth,
        confidence=np.full_like(depth, 0.75, dtype=np.float32),
        valid=np.ones_like(depth, dtype=bool),
    )
    return accumulator


def test_paired_cache_comparison_detects_constant_depth_collapse() -> None:
    height, width = 12, 16
    horizontal = np.linspace(1.0, 3.0, width, dtype=np.float32)[None]
    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    ground_truth = horizontal + vertical
    base_depth = ground_truth * 0.5
    collapsed_depth = np.full_like(ground_truth, np.median(ground_truth))

    record = _evaluate_frame(
        gt_depth=ground_truth,
        base=_accumulator(base_depth),
        finetuned=_accumulator(collapsed_depth),
        min_depth=0.1,
        max_depth=10.0,
        min_valid_pixels=10,
    )

    assert record is not None
    assert record["base_errors"][0] < 1e-6
    assert record["finetuned_errors"][0] > record["base_errors"][0]
    assert record["base_depth_cv"] > record["finetuned_depth_cv"]
    assert (
        record["base_normalized_gradient"]
        > record["finetuned_normalized_gradient"]
    )

    summary = _summarize([record])
    assert summary["finetuned_improvement"]["abs_rel"] < 0
    assert summary["finetuned_abs_rel_win_rate"] == 0.0
