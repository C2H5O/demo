from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from evaluation.evaluate_crossclip_projection import evaluate_vda
from evaluation.evaluate_vda import (
    _SequencePredictionSpool,
    _resized_gt,
    _student_depth_to_vda_disparity,
)
from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class _NativePlane(nn.Module):
    def forward(self, images, include_global_points=False):
        batch, frames, _, height, width = images.shape
        return {
            "depth": torch.ones(batch, frames, height, width),
            "intrinsics": torch.eye(3).repeat(batch, frames, 1, 1),
            "extrinsics": torch.eye(4)[:3].repeat(batch, frames, 1, 1),
        }


def _scared_fixture(root: Path) -> Path:
    for dataset_id in (8, 9):
        data = root / f"dataset_{dataset_id}" / "keyframe_1" / "data"
        for name in ("left_finalpass", "depth", "frame_data"):
            (data / name).mkdir(parents=True)
        for frame_id in range(2):
            Image.new("RGB", (6, 4)).save(
                data / "left_finalpass" / f"frame_{frame_id:06d}.png"
            )
            np.save(
                data / "depth" / f"depth_{frame_id:06d}.npy",
                np.ones((4, 6), dtype=np.float32) * 1000.0,
            )
            (data / "frame_data" / f"frame_data{frame_id:06d}.json").write_text(
                json.dumps(
                    {
                        "camera-calibration": {
                            "KL": [[100, 0, 3], [0, 100, 2], [0, 0, 1]]
                        },
                        "camera-pose": np.eye(4).tolist(),
                    }
                ),
                encoding="utf-8",
            )
    config = {
        "device": "cpu",
        "student": {"checkpoint": "unused"},
        "dataset": {
            "root": str(root),
            "clip_length": 16,
            "sample_stride": 1,
            "window_stride": 8,
            "image_height": 4,
            "image_width": 6,
            "resize_mode": "resize",
            "normalize_mode": "zero_one",
        },
        "vda_evaluation": {
            "rgb_root": str(root),
            "gt_root": str(root),
            "checkpoint": "unused",
            "output": str(root / "result.json"),
            "split": "test",
            "evaluation_height": 2,
            "evaluation_width": 3,
        },
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_baseline_configs_keep_native_input_and_lock_paper_grid() -> None:
    if (ROOT / "configs/baselines/J.yaml").is_file():
        path = ROOT / "configs/baselines/J.yaml"
        section = "vda_evaluation"
    else:
        path = ROOT / "configs/baselines/A.yaml"
        section = "da3_small_baseline_vda_evaluation"
    config = load_config(path)
    assert [config["dataset"]["image_height"], config["dataset"]["image_width"]] == [448, 560]
    assert [config["student"]["image_height"], config["student"]["image_width"]] == [448, 560]
    assert [config[section]["evaluation_height"], config[section]["evaluation_width"]] == [256, 320]


def test_spool_resizes_disparity_bilinearly_before_evaluation(tmp_path: Path) -> None:
    depth = np.array(
        [[[1.0, 2.0], [4.0, 8.0]]], dtype=np.float32
    )
    disparity = _student_depth_to_vda_disparity(depth)
    spool = _SequencePredictionSpool(tmp_path, 1, height=3, width=5)
    try:
        spool.add([0], disparity)
        actual = spool.prediction(0)
        expected = cv2.resize(
            disparity[0], (5, 3), interpolation=cv2.INTER_LINEAR
        )
        wrong_order = np.reciprocal(
            cv2.resize(depth[0], (5, 3), interpolation=cv2.INTER_LINEAR)
        )
        np.testing.assert_allclose(actual, expected)
        assert not np.allclose(actual, wrong_order)
        assert spool.native_prediction_resolutions_hw == {(2, 2)}
    finally:
        spool.close()


def test_ground_truth_uses_nearest_neighbor_at_evaluation_shape(tmp_path: Path) -> None:
    path = tmp_path / "depth_000000.npy"
    np.save(path, np.array([[1000.0, 2000.0], [3000.0, 4000.0]], np.float32))
    resized = _resized_gt(path, 0, height=2, width=4)
    np.testing.assert_allclose(
        resized,
        np.array([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]], np.float32),
    )


def test_result_metadata_separates_native_model_and_evaluation_grids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _scared_fixture(tmp_path)
    monkeypatch.setattr(
        "evaluation.evaluate_crossclip_projection._evaluation_model",
        lambda *args: _NativePlane().eval(),
    )
    result = evaluate_vda(config)
    written = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    for value in (result, written):
        assert value["model_input_resolution_hw"] == [4, 6]
        assert value["native_prediction_resolution_hw"] == [4, 6]
        assert value["evaluation_resolution_hw"] == [2, 3]
        assert all(
            item["evaluation_shape_hxw"] == [2, 3]
            and item["temporal"]["evaluation_resolution_hw"] == [2, 3]
            for item in value["sequences"]
        )
