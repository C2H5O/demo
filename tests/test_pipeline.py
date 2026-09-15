import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from endodaveval.config import ConfigError, load_config
from endodaveval.pipeline import run_pipeline


class FakeEndoDAV(nn.Module):
    def forward(self, value):
        return value

    def infer_video_depth(self, frames):
        raw = np.full(frames.shape[:3], 0.5, dtype=np.float32)
        return self.forward(torch.from_numpy(raw)).numpy()


def _config(tmp_path: Path) -> Path:
    for dataset_id in (8, 9):
        data = tmp_path / "scared" / f"dataset{dataset_id}" / "keyframe_0" / "data"
        for directory in (data / "left", data / "depth", data / "frame_data"):
            directory.mkdir(parents=True)
        for identifier in (0, 1):
            assert cv2.imwrite(
                str(data / "left" / f"frame_{identifier:06d}.png"),
                np.zeros((4, 6, 3), np.uint8),
            )
            np.save(
                data / "depth" / f"depth_{identifier:06d}.npy",
                np.ones((4, 6), np.float32) * 1000.0,
            )
            (data / "frame_data" / f"frame_data{identifier:06d}.json").write_text(
                json.dumps({
                    "camera-calibration": {"KL": [[100, 0, 3], [0, 100, 2], [0, 0, 1]]},
                    "camera-pose": np.eye(4).tolist(),
                }), encoding="utf-8"
            )
    config = {
        "dataset": {
            "root": str(tmp_path / "scared"),
            "dataset_ids": [8, 9],
            "frame_sources": ["left"],
            "ground_truth_directory": "data/depth",
            "ground_truth_scale": 0.001,
            "ground_truth_channel": 0,
        },
        "endodav": {
            "repository": str(tmp_path / "EndoDAV"),
            "checkpoint": str(tmp_path / "depth_model.pth"),
            "pretrained_path": str(tmp_path / "pretrained_model"),
            "device": "cpu",
        },
        "evaluation": {
            "device": "cpu",
            "height": 256,
            "width": 320,
            "min_depth": 0.001,
            "max_depth": 100.0,
            "require_all_frames": True,
            "tae": {"enabled": True, "require_all_pairs": True},
            "result_file": "evaluation_vda.json",
        },
        "output_root": str(tmp_path / "outputs"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_full_pipeline_uses_official_full_video_boundary_and_unified_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _config(tmp_path)
    monkeypatch.setattr(
        "endodaveval.pipeline.preflight",
        lambda *args: {
            "status": "ok",
            "official_repository": "fake",
            "official_source_modified": False,
        },
    )
    monkeypatch.setattr(
        "endodaveval.pipeline.load_official_model",
        lambda *args: FakeEndoDAV().eval(),
    )
    result = run_pipeline(path, stage="all")
    assert result["external_windowing"] is False
    assert result["ground_truth_used_for_inference"] is False
    assert result["internal_model_resolution_hw"] == [224, 280]
    assert result["model_input_resolution_hw"] == [4, 6]
    assert result["native_prediction_resolution_hw"] == [4, 6]
    assert result["evaluation_resolution_hw"] == [256, 320]
    assert result["metric_aggregation"] == "per-frame within sequence, then macro mean over evaluated sequences"
    assert result["metrics"]["abs_relative_difference"] == pytest.approx(0.0, abs=1e-6)
    assert result["metrics"]["rmse_linear"] == pytest.approx(0.0, abs=1e-6)
    assert result["metrics"]["delta1_acc"] == pytest.approx(1.0)
    assert result["metrics"]["tae"] == pytest.approx(0.0)
    assert result["timing"]["evaluation_resize_in_model_forward_timing"] is False
    assert all(item["frame_ids"] == [0, 1] for item in result["sequences"])
    written = json.loads((tmp_path / "outputs/evaluation_vda.json").read_text(encoding="utf-8"))
    assert written["evaluation_resolution_hw"] == [256, 320]


def test_config_rejects_non_reference_evaluation_grid(tmp_path: Path) -> None:
    path = _config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["evaluation"]["height"] = 320
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ConfigError, match="256x320"):
        load_config(path)
