import json

import numpy as np
import pytest
from PIL import Image

from evaluation.temporal_alignment import project_depth, read_scared_camera, evaluate_tae


def test_identity_and_constant_depth_error():
    depth = np.ones((4, 5))
    error, count = project_depth(depth, depth, np.eye(3), np.eye(3), np.eye(4))
    assert error == 0 and count == 20
    error, count = project_depth(depth, depth * 2, np.eye(3), np.eye(3), np.eye(4))
    assert error == 0.5 and count == 20


def test_camera_z_motion_is_not_mistaken_for_flicker():
    # Same plane becomes twice as far from the target camera.
    transform = np.eye(4)
    transform[2, 3] = 1
    error, count = project_depth(np.ones((4, 5)), np.ones((4, 5)) * 2,
                                  np.eye(3), np.eye(3), transform)
    assert error == 0 and count > 0


def test_zbuffer_keeps_nearest_surface_and_empty_is_not_zero_error():
    k = np.eye(3)
    target_k = np.diag([0.1, 1., 1.])
    error, count = project_depth(np.array([[1., 2.]]), np.ones((1, 1)), k, target_k, np.eye(4))
    assert error == 0 and count == 1
    transform = np.eye(4)
    transform[2, 3] = -10
    assert project_depth(np.ones((2, 2)), np.ones((2, 2)), k, k, transform) == (None, 0)


def test_scared_camera_units_direction_and_intrinsics_resize(tmp_path):
    path = tmp_path / "frame_data000000.json"
    pose = np.eye(4)
    pose[0, 3] = 1000
    path.write_text(json.dumps({"camera-calibration": {"KL": [[100,0,5],[0,100,4],[0,0,1]]},
                                "camera-pose": pose.tolist()}))
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (10, 8)).save(rgb)
    k, extrinsics = read_scared_camera(path, rgb, (4, 5))
    np.testing.assert_allclose(k, [[50,0,2.5],[0,50,2],[0,0,1]])
    assert extrinsics[0, 3] == 1.0


def _fixture(tmp_path, ids=(0, 1, 2)):
    frame_data = tmp_path / "data" / "frame_data"
    frame_data.mkdir(parents=True)
    paths = []
    for i in ids:
        path = tmp_path / f"rgb_{i:06d}.png"
        Image.new("RGB", (3, 2)).save(path)
        paths.append(str(path))
        (frame_data / f"frame_data{i:06d}.json").write_text(json.dumps({
            "camera-calibration": {"KL": np.eye(3).tolist()}, "camera-pose": np.eye(4).tolist()}))
    sequence = {"sequence_id": "fixture", "keyframe_directory": str(tmp_path), "frame_paths": paths}
    class Spool:
        height, width = 2, 3
        counts = np.ones(len(ids), dtype=int)
        def prediction(self, i):
            return np.ones((2, 3)) / (i + 1)
    return sequence, Spool()


def test_tae_uses_both_directions_and_percent_units(tmp_path):
    sequence, spool = _fixture(tmp_path)
    result = evaluate_tae(sequence, spool, {"disparity_scale": 1, "disparity_shift": 0}, {})
    # Depths 1, 2, 3: directed errors 1/2, 1, 1/3, 1/2.
    assert result["tae"] == pytest.approx(100 * (0.5 + 1 + 1/3 + 0.5) / 4)
    assert result["evaluated_pair_count"] == 2
    assert result["status"] == "complete"


def test_missing_camera_and_nonconsecutive_frames_are_reported(tmp_path):
    sequence, spool = _fixture(tmp_path, ids=(0, 1, 3))
    (tmp_path / "data/frame_data/frame_data000001.json").unlink()
    spatial = {"disparity_scale": 1, "disparity_shift": 0}
    with pytest.raises(RuntimeError, match="missing_prediction"):
        evaluate_tae(sequence, spool, spatial, {})
    result = evaluate_tae(sequence, spool, spatial, {"tae": {"require_all_pairs": False}})
    assert result["tae"] is None
    assert result["nonconsecutive_pair_count"] == 1
    assert len(result["skipped_pairs"]) == 1
