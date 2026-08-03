from pathlib import Path

import numpy as np
import pytest

import evaluation.evaluate_vda as vda
import evaluation.evaluate_endo3r_vda as baseline
from evaluation.evaluate_endo3r_vda import (
    _evaluate_sequence_predictions,
    _index_npy_depths,
)


def test_index_npy_depths_uses_numeric_ids(tmp_path: Path) -> None:
    np.save(tmp_path / "frame_10.npy", np.ones((2, 3), dtype=np.float32))
    np.save(tmp_path / "frame_2.npy", np.ones((2, 3), dtype=np.float32))
    assert set(_index_npy_depths(tmp_path)) == {2, 10}


def test_endo3r_depth_is_routed_to_project_vda_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeCV2:
        INTER_NEAREST = 0

        @staticmethod
        def resize(array, size, interpolation=None):
            del interpolation
            width, height = size
            y = np.linspace(0, array.shape[0] - 1, height).astype(int)
            x = np.linspace(0, array.shape[1] - 1, width).astype(int)
            return array[y[:, None], x[None, :]]

    monkeypatch.setattr(vda, "_opencv", lambda: FakeCV2)
    captured = {}

    def fake_evaluate_sequence(
        sequence, spool, gt_channel, gt_depths, require_all_gt
    ):
        captured["sequence"] = sequence["sequence_id"]
        captured["gt_channel"] = gt_channel
        captured["gt_directory"] = gt_depths[0]
        captured["require_all_gt"] = require_all_gt
        captured["disparities"] = [
            spool.prediction(index) for index in range(spool.frame_count)
        ]
        return {
            "sequence_id": sequence["sequence_id"],
            "matched_frame_count": spool.frame_count,
            "metrics": {
                "abs_relative_difference": 0.0,
                "rmse_linear": 0.0,
                "delta1_acc": 1.0,
            },
        }

    monkeypatch.setattr(baseline, "_evaluate_sequence", fake_evaluate_sequence)
    keyframe = tmp_path / "dataset_8" / "keyframe_1"
    frame_directory = keyframe / "data" / "left_rectified"
    gt_directory = keyframe / "data" / "depthmap_rectified"
    prediction_directory = tmp_path / "predictions"
    frame_directory.mkdir(parents=True)
    gt_directory.mkdir(parents=True)
    prediction_directory.mkdir()

    frame_paths = []
    for frame_id, depth_metres in enumerate((0.5, 0.75, 1.0)):
        frame_path = frame_directory / "frame_{:04d}.png".format(frame_id)
        frame_path.touch()
        frame_paths.append(str(frame_path))
        depth = np.full((3, 4), depth_metres, dtype=np.float32)
        np.save(
            gt_directory / "depth_{:04d}.npy".format(frame_id),
            depth * 1000.0,
        )
        np.save(
            prediction_directory / "frame_{:04d}.npy".format(frame_id),
            depth,
        )

    sequence = {
        "sequence_id": "dataset_8/keyframe_1",
        "keyframe_directory": str(keyframe),
        "frame_directory": str(frame_directory),
        "frame_paths": frame_paths,
        "sequence_length": len(frame_paths),
        "depth_directory": None,
        "scene_points_directory": None,
    }
    result = _evaluate_sequence_predictions(
        sequence,
        prediction_directory,
        tmp_path,
        {"ground_truth": {"relative_directories": ["data/depthmap_rectified"]}},
        {
            "image_height": 6,
            "image_width": 8,
            "gt_relative_directory": "data/depthmap_rectified",
            "gt_depth_channel": 0,
            "require_all_gt": True,
        },
    )

    assert captured["sequence"] == "dataset_8/keyframe_1"
    assert captured["gt_channel"] == 0
    assert captured["require_all_gt"] is True
    for disparity, depth in zip(captured["disparities"], (0.5, 0.75, 1.0)):
        assert disparity == pytest.approx(np.full((6, 8), 1.0 / depth))
    assert result["matched_frame_count"] == 3
