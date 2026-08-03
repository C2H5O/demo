from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from visualization.visualize_teacher_cache import visualize_teacher_cache


def test_visualize_teacher_cache_without_rgb_falls_back_to_confidence(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cv2")
    cache_path = (
        tmp_path
        / "teacher_cache"
        / "train"
        / "dataset_01"
        / "keyframe_1"
        / "start_000000_len_002_stride_01.npz"
    )
    cache_path.parent.mkdir(parents=True)
    frames, height, width = 2, 4, 4
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    local = np.zeros((frames, height, width, 3), dtype=np.float32)
    local[..., 0] = xs
    local[..., 1] = ys
    local[..., 2] = 2.0
    global_points = local.copy()
    global_points[1, ..., 0] += 1.0
    confidence = np.linspace(
        0.0, 1.0, frames * height * width, dtype=np.float32
    ).reshape(frames, height, width)
    np.savez_compressed(
        cache_path,
        xyz_local=local,
        xyz_global=global_points,
        conf_local=confidence,
        conf_global=confidence,
        valid_mask=np.ones((frames, height, width), dtype=bool),
        frame_names=np.asarray(["frame_0000.png", "frame_0001.png"]),
        teacher_depth_range=np.asarray([0.1, 10.0], dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps({"frame_paths": ["missing_0.png", "missing_1.png"]})
        ),
    )

    output = visualize_teacher_cache(
        cache_path,
        tmp_path / "output",
        point_stride=2,
        point_color="rgb",
    )

    assert (output / "depth_color" / "frame_0000.png").is_file()
    assert (output / "confidence_color" / "frame_0001.png").is_file()
    assert (output / "panels" / "frame_0000.png").is_file()
    assert (output / "teacher_global_point_cloud.ply").is_file()
    report = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert report["point_color"] == "confidence"
    assert report["rgb_available"] is False
    assert report["point_count"] == 8
