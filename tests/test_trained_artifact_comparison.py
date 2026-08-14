from __future__ import annotations

import json

import numpy as np

from diagnostics.compare_trained_artifact_traces import STAGES, compare_traces


def _write_trace(root, offset: float) -> None:
    metadata = {
        "split": "test",
        "sequence_id": "dataset_8/keyframe_0",
        "clip_start": 0,
        "frame_index": 0,
        "frame_name": "000000.png",
        "seed": 0,
    }
    (root / "trace_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for _stage, (relative, period) in STAGES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        side = max(period * 2, 8)
        value = np.arange(side * side, dtype=np.float32).reshape(side, side) + offset
        np.save(path, value)


def test_trained_trace_comparison_uses_one_depth_display_range(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    experiment = tmp_path / "experiment"
    baseline.mkdir()
    experiment.mkdir()
    _write_trace(baseline, 0.0)
    _write_trace(experiment, 100.0)

    output = compare_traces(baseline, experiment, tmp_path / "comparison")
    metadata = json.loads((output / "comparison_metadata.json").read_text(encoding="utf-8"))

    assert metadata["same_sample_verified"] is True
    assert metadata["depth_display"]["vmax"] > metadata["depth_display"]["vmin"]
    assert (output / "depth_original_dpt.npy").is_file()
    assert (output / "depth_bilinear_head.npy").is_file()
    assert (output / "depth_original_dpt.png").is_file()
    assert (output / "depth_bilinear_head.png").is_file()
