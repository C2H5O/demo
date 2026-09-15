import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from endodaveval.data import SequenceRecord
from endodaveval.temporal_alignment import evaluate_tae, read_scared_camera, tae_torch


def _record(tmp_path: Path) -> SequenceRecord:
    keyframe = tmp_path / "dataset8/keyframe_0"
    rgb_dir = keyframe / "data/left"
    gt_dir = keyframe / "data/depth"
    camera_dir = keyframe / "data/frame_data"
    for directory in (rgb_dir, gt_dir, camera_dir):
        directory.mkdir(parents=True)
    rgb_by_id = {}
    gt_by_id = {}
    for identifier in (0, 1):
        rgb = rgb_dir / f"frame_{identifier:06d}.png"
        gt = gt_dir / f"depth_{identifier:06d}.npy"
        assert cv2.imwrite(str(rgb), np.zeros((512, 640, 3), np.uint8))
        np.save(gt, np.ones((2, 2), np.float32))
        (camera_dir / f"frame_data{identifier:06d}.json").write_text(
            json.dumps({
                "camera-calibration": {"KL": [[400, 0, 320], [0, 500, 256], [0, 0, 1]]},
                "camera-pose": np.eye(4).tolist(),
            }), encoding="utf-8"
        )
        rgb_by_id[identifier] = rgb
        gt_by_id[identifier] = gt
    return SequenceRecord(8, "keyframe_0", "dataset_8/keyframe_0", keyframe, rgb_dir, rgb_by_id, gt_dir, gt_by_id)


def test_intrinsics_scale_from_raw_rgb_to_256x320(tmp_path: Path) -> None:
    record = _record(tmp_path)
    camera = record.keyframe_directory / "data/frame_data/frame_data000000.json"
    K, pose = read_scared_camera(camera, record.rgb_by_id[0])
    np.testing.assert_allclose(K, [[200, 0, 160], [0, 250, 128], [0, 0, 1]])
    np.testing.assert_allclose(pose, np.eye(4))


def test_identity_aligned_sequence_has_zero_bidirectional_percent_tae(tmp_path: Path) -> None:
    record = _record(tmp_path)
    depths = {identifier: np.ones((256, 320), np.float32) for identifier in (0, 1)}
    result = evaluate_tae(record, depths, [0, 1], {"enabled": True, "require_all_pairs": True}, "cpu")
    assert result["tae"] == pytest.approx(0.0)
    assert result["tae_denominator"] == "2 * (num_frames - 1)"
    assert result["evaluation_resolution_hw"] == [256, 320]


def test_empty_projection_is_zero_like_vda() -> None:
    depth = torch.ones((2, 3), dtype=torch.float64)
    error = tae_torch(
        depth,
        depth,
        torch.eye(3, dtype=torch.float64),
        torch.tensor([1000.0, 0.0, 0.0], dtype=torch.float64),
        np.eye(3),
        torch.ones_like(depth, dtype=torch.bool),
    )
    assert error == 0

\n