import json
import math

import numpy as np
import pytest
from PIL import Image
import torch

from evaluation.temporal_alignment import (
    evaluate_tae,
    read_scared_camera,
    tae_torch,
    vda_relative_pose,
)
from utils.config import load_config


def _compute_errors_torch_reference(gt, pred):
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    return abs_rel


def tae_torch_reference(depth1, depth2, R_2_1, T_2_1, K, mask):
    """Near-verbatim reference from Video-Depth-Anything eval_tae.py."""
    H, W = depth1.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H))
    xx, yy = xx.t(), yy.t()
    xx = xx.to(dtype=depth1.dtype, device=depth1.device)
    yy = yy.to(dtype=depth1.dtype, device=depth1.device)
    X = (xx - cx) * depth1 / fx
    Y = (yy - cy) * depth1 / fy
    Z = depth1
    points3d = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)
    T = torch.tensor(T_2_1, dtype=depth1.dtype, device=depth1.device)
    points3d_transformed = torch.matmul(points3d, R_2_1.T) + T
    X_world = points3d_transformed[:, 0]
    Y_world = points3d_transformed[:, 1]
    Z_world = points3d_transformed[:, 2]
    X_plane = (X_world * fx) / Z_world + cx
    Y_plane = (Y_world * fy) / Z_world + cy
    X_plane = torch.round(X_plane).to(dtype=torch.long)
    Y_plane = torch.round(Y_plane).to(dtype=torch.long)
    valid_mask = (
        (X_plane >= 0) & (X_plane < W) & (Y_plane >= 0) & (Y_plane < H)
    )
    if valid_mask.sum() == 0:
        return 0
    depth_proj = torch.zeros(
        (H, W), dtype=depth1.dtype, device=depth1.device
    )
    valid_X = X_plane[valid_mask]
    valid_Y = Y_plane[valid_mask]
    valid_Z = Z_world[valid_mask]
    depth_proj[valid_Y, valid_X] = valid_Z
    valid_mask = (depth_proj > 0) & (depth2 > 0) & mask
    if valid_mask.sum() == 0:
        return 0
    abs_errors = _compute_errors_torch_reference(
        depth2[valid_mask], depth_proj[valid_mask]
    )
    return abs_errors


def _assert_same_scalar(actual, expected):
    actual_tensor = torch.as_tensor(actual)
    expected_tensor = torch.as_tensor(expected)
    assert torch.allclose(actual_tensor, expected_tensor, rtol=1e-7, atol=1e-9)


def test_reference_parity_fixed_synthetic():
    depth1 = torch.tensor(
        [[1.0, 1.2, 1.4], [1.1, 1.3, 1.5]], dtype=torch.float64
    )
    depth2 = torch.tensor(
        [[1.05, 1.1, 1.45], [1.15, 1.25, 1.55]], dtype=torch.float64
    )
    K = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    R = torch.eye(3, dtype=torch.float64)
    translation = torch.tensor([0.05, 0.0, 0.1], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    expected = tae_torch_reference(depth1, depth2, R, translation, K, mask)
    actual = tae_torch(depth1, depth2, R, translation, K, mask)
    _assert_same_scalar(actual, expected)


def test_identity_is_zero():
    depth = torch.ones((4, 5), dtype=torch.float64)
    mask = torch.ones_like(depth, dtype=torch.bool)
    error = tae_torch(
        depth, depth, torch.eye(3, dtype=depth.dtype), torch.zeros(3), np.eye(3), mask
    )
    assert float(error) == pytest.approx(0.0)


def test_depth_perturbation_is_positive():
    depth1 = torch.ones((4, 5), dtype=torch.float64)
    depth2 = depth1 * 2
    mask = torch.ones_like(depth1, dtype=torch.bool)
    error = tae_torch(
        depth1,
        depth2,
        torch.eye(3, dtype=depth1.dtype),
        torch.zeros(3),
        np.eye(3),
        mask,
    )
    assert float(error) > 0


def test_projection_collision_uses_direct_assignment_not_minimum_z():
    depth1 = torch.full((1, 2), 0.5, dtype=torch.float64)
    angle = math.radians(-80)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=torch.float64,
    )
    translation = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    first_z = cosine * 0.5
    last_z = -sine * 0.5 + cosine * 0.5
    assert last_z > first_z
    depth2 = torch.tensor([[last_z, 1.0]], dtype=torch.float64)
    mask = torch.ones_like(depth1, dtype=torch.bool)
    expected = tae_torch_reference(
        depth1, depth2, rotation, translation, np.eye(3), mask
    )
    actual = tae_torch(depth1, depth2, rotation, translation, np.eye(3), mask)
    # Both source pixels land at (0, 0); VDA direct assignment keeps valid_Z[-1].
    assert float(expected) == pytest.approx(0.0, abs=1e-12)
    assert float(actual) == pytest.approx(0.0, abs=1e-12)
    minimum_z_error = abs(last_z - first_z) / last_z
    assert float(actual) != pytest.approx(minimum_z_error)


def test_empty_projection_returns_zero():
    depth = torch.ones((2, 2), dtype=torch.float64)
    error = tae_torch(
        depth,
        depth,
        torch.eye(3, dtype=depth.dtype),
        torch.tensor([100.0, 0.0, 0.0], dtype=depth.dtype),
        np.eye(3),
        torch.ones_like(depth, dtype=torch.bool),
    )
    assert error == 0


def test_scared_pose_is_converted_to_vda_c2w_and_k_is_resized(tmp_path):
    path = tmp_path / "frame_data000000.json"
    raw_w2c = np.eye(4)
    raw_w2c[0, 3] = 1000
    path.write_text(
        json.dumps(
            {
                "camera-calibration": {
                    "KL": [[100, 0, 5], [0, 100, 4], [0, 0, 1]]
                },
                "camera-pose": raw_w2c.tolist(),
            }
        )
    )
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (10, 8)).save(rgb)
    k, pose_c2w = read_scared_camera(path, rgb, (4, 5))
    np.testing.assert_allclose(k, [[50, 0, 2.5], [0, 50, 2], [0, 0, 1]])
    assert pose_c2w[0, 3] == pytest.approx(-1.0)
    np.testing.assert_allclose(
        vda_relative_pose(np.eye(4), np.eye(4)), np.eye(4)
    )


def _fixture(tmp_path, ids=(0, 1, 2)):
    frame_data = tmp_path / "data" / "frame_data"
    frame_data.mkdir(parents=True)
    paths = []
    for i in ids:
        path = tmp_path / f"rgb_{i:06d}.png"
        Image.new("RGB", (3, 2)).save(path)
        paths.append(str(path))
        (frame_data / f"frame_data{i:06d}.json").write_text(
            json.dumps(
                {
                    "camera-calibration": {"KL": np.eye(3).tolist()},
                    "camera-pose": np.eye(4).tolist(),
                }
            )
        )
    sequence = {
        "sequence_id": "fixture",
        "keyframe_directory": str(tmp_path),
        "frame_paths": paths,
    }

    class Spool:
        height, width = 2, 3
        counts = np.ones(len(ids), dtype=int)

        def prediction(self, i):
            return np.ones((2, 3), dtype=np.float32) / (i + 1)

    return sequence, Spool()


def test_sequence_tae_is_bidirectional_percent_with_fixed_denominator(tmp_path):
    sequence, spool = _fixture(tmp_path)
    result = evaluate_tae(
        sequence,
        spool,
        {"disparity_scale": 1, "disparity_shift": 0},
        {},
    )
    # Depths 1, 2, 3: directed errors 1/2, 1, 1/3, 1/2.
    assert result["tae"] == pytest.approx(
        100 * (0.5 + 1 + 1 / 3 + 0.5) / (2 * (3 - 1))
    )
    assert result["evaluated_pair_count"] == 2
    assert result["status"] == "complete"
    assert result["tae_denominator"] == "2 * (num_frames - 1)"
    assert result["tae_k_usage"] == "first_frame_K_for_both_directions_matching_vda"
    assert result["adjacent_intrinsics_all_equal"]


def test_missing_camera_is_data_coverage_not_empty_projection(tmp_path):
    sequence, spool = _fixture(tmp_path)
    (tmp_path / "data/frame_data/frame_data000001.json").unlink()
    spatial = {"disparity_scale": 1, "disparity_shift": 0}
    with pytest.raises(RuntimeError, match="missing_dataset_camera"):
        evaluate_tae(sequence, spool, spatial, {})
    result = evaluate_tae(
        sequence, spool, spatial, {"tae": {"require_all_pairs": False}}
    )
    assert result["status"] == "partial"
    assert result["evaluated_frame_count"] == 2
    assert result["evaluated_pair_count"] == 1
    assert len(result["skipped_frames"]) == 1


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/vggtoda3.yaml",
        "configs/vggtoda3_attention_distill.yaml",
        *[f"configs/baselines/{name}.yaml" for name in "ABCDEFG"],
    ],
)
def test_all_baselines_and_attention_inherit_the_shared_tae(config_path):
    config = load_config(config_path)
    for section in ("vda_evaluation", "da3_small_baseline_vda_evaluation"):
        assert config[section]["tae"]["enabled"] is True
        assert "frame_id_step" not in config[section]["tae"]
        assert config[section]["tae"]["require_all_pairs"] is True
