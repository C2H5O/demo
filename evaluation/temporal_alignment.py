"""Video-Depth-Anything TAE with only SCARED data-loading adaptations.

The operational reference is:
DepthAnything/Video-Depth-Anything/benchmark/eval/eval_tae.py

The pairwise projection, collision behavior, masking, bidirectional pairing,
empty-projection result, and normalization below intentionally match that file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from evaluation.scared_gt import SCARED_MAX_DEPTH, extract_frame_id


TAE_REFERENCE = "DepthAnything/Video-Depth-Anything benchmark/eval/eval_tae.py"
VDA_TAE_METADATA = {
    "tae_reference": TAE_REFERENCE,
    "tae_unit": "percent",
    "tae_direction": "lower_is_better",
    "tae_bidirectional": True,
    "tae_pairing": "adjacent_frames",
    "tae_projection_collision": "direct_assignment_matching_vda",
    "tae_empty_projection": "zero_error_matching_vda",
    "tae_alignment": "single_sequence_disparity_scale_shift_fit",
    "tae_denominator": "2 * (num_frames - 1)",
}


def compute_errors_torch(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """AbsRel helper matching the Video-Depth-Anything reference."""
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    return abs_rel


def tae_torch(
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    R_2_1: torch.Tensor,
    T_2_1: torch.Tensor,
    K: np.ndarray,
    mask: torch.Tensor,
):
    """Pairwise TAE copied from Video-Depth-Anything's operational reference.

    In particular, duplicate projected indices use VDA's direct assignment.
    """
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
    abs_errors = compute_errors_torch(
        depth2[valid_mask], depth_proj[valid_mask]
    )
    return abs_errors


def camera_index(sequence, eval_config):
    from evaluation.evaluate_vda import _find_gt_keyframe

    cfg = eval_config.get("tae", {})
    keyframe = (
        _find_gt_keyframe(sequence, Path(eval_config["gt_root"]))
        if eval_config.get("gt_root")
        else Path(sequence["keyframe_directory"])
    )
    directory = keyframe / cfg.get(
        "frame_data_relative_directory", "data/frame_data"
    )
    result = {}
    for path in sorted(directory.glob("*.json")):
        frame_id = extract_frame_id(path)
        if frame_id in result:
            raise ValueError(
                "Duplicate TAE camera frame ID {} in {}".format(
                    frame_id, directory
                )
            )
        result[frame_id] = path
    return directory, result


def read_scared_camera(
    path, rgb_path, output_shape, translation_scale=0.001
):
    """Return resized K and a VDA-compatible camera-to-world pose.

    SCARED ``camera-pose`` is world-to-camera: X_camera = R X_world + t.
    After converting its translation from millimetres to metres, it is inverted
    to the camera-to-world convention used by VDA's ScanNet benchmark poses.
    Consequently ``inv(T_2) @ T_1`` maps camera-1 points into camera 2.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    k = np.asarray(record["camera-calibration"]["KL"], dtype=np.float64)
    raw_w2c = np.asarray(record["camera-pose"], dtype=np.float64)
    if k.shape != (3, 3) or raw_w2c.shape != (4, 4):
        raise ValueError(
            "TAE requires KL[3,3] and camera-pose[4,4]: {}".format(path)
        )
    if not (np.isfinite(k).all() and np.isfinite(raw_w2c).all()):
        raise ValueError("Non-finite TAE camera: {}".format(path))
    if (
        k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
    ):
        raise ValueError("Invalid pinhole intrinsics: {}".format(path))
    r = raw_w2c[:3, :3]
    if (
        not np.allclose(raw_w2c[3], [0, 0, 0, 1])
        or not np.allclose(r @ r.T, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(r), 1, atol=1e-3)
    ):
        raise ValueError(
            "TAE camera-pose is not a rigid SCARED world-to-camera transform"
        )
    if not np.isfinite(translation_scale) or translation_scale <= 0:
        raise ValueError("pose_translation_scale must be finite and positive")
    w2c_metres = raw_w2c.copy()
    w2c_metres[:3, 3] *= translation_scale
    pose_c2w = np.linalg.inv(w2c_metres)
    with Image.open(rgb_path) as image:
        width, height = image.size
    k = np.diag(
        [output_shape[1] / width, output_shape[0] / height, 1.0]
    ) @ k
    return k, pose_c2w


def vda_relative_pose(T_1_c2w: np.ndarray, T_2_c2w: np.ndarray) -> np.ndarray:
    """VDA formula for T_2_1, mapping camera-1 coordinates to camera 2.

    Identity inputs therefore produce an identity relative transform.
    """
    return np.linalg.inv(T_2_c2w) @ T_1_c2w


def _metadata(**values):
    return {**VDA_TAE_METADATA, **values}


def evaluate_tae(sequence, spool, spatial_result, eval_config):
    """Evaluate one sequence with VDA's fixed bidirectional denominator."""
    cfg = eval_config.get("tae", {})
    if not cfg.get("enabled", True):
        return _metadata(status="disabled", tae=None)

    directory, cameras = camera_index(sequence, eval_config)
    paths = sequence["frame_paths"]
    ids = [extract_frame_id(path) for path in paths]
    strict = bool(cfg.get("require_all_pairs", True))
    # VDA's lstsq returns (1,) float64 arrays. Reconstruct that dtype/shape from
    # the JSON-safe spatial result so aligned depth follows the same promotion.
    scale = np.asarray([spatial_result["disparity_scale"]], dtype=np.float64)
    shift = np.asarray([spatial_result["disparity_shift"]], dtype=np.float64)
    frames = []
    skipped = []
    for pos, identifier in enumerate(ids):
        reason = None
        if spool.counts[pos] <= 0:
            reason = "missing_prediction"
        elif identifier not in cameras:
            reason = "missing_dataset_camera"
        if reason is not None:
            skipped.append({"frame_id": identifier, "reason": reason})
            continue
        k, pose_c2w = read_scared_camera(
            cameras[identifier],
            paths[pos],
            (spool.height, spool.width),
            float(cfg.get("pose_translation_scale", 0.001)),
        )
        disparity = np.clip(spool.prediction(pos), a_min=1e-3, a_max=None)
        aligned_pred_disp = np.clip(
            scale * disparity + shift, a_min=1e-3, a_max=None
        )
        depth = np.clip(
            np.reciprocal(aligned_pred_disp),
            a_min=1e-3,
            a_max=SCARED_MAX_DEPTH,
        )
        frames.append((identifier, depth, k, pose_c2w))

    if skipped and strict:
        first = skipped[0]
        raise RuntimeError(
            "TAE frame {}: {}; camera directory: {}".format(
                first["frame_id"], first["reason"], directory
            )
        )
    if len(frames) < 2:
        if strict:
            raise RuntimeError(
                "TAE requires at least two frames in {}".format(
                    sequence["sequence_id"]
                )
            )
        return _metadata(
            status="unavailable",
            tae=None,
            evaluated_frame_count=len(frames),
            evaluated_pair_count=0,
            skipped_frames=skipped,
            camera_directory=str(directory),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    error_sum = 0.0
    intrinsic_differences = []
    for index in range(len(frames) - 1):
        _, depth1_numpy, K, T_1 = frames[index]
        _, depth2_numpy, K_2, T_2 = frames[index + 1]
        intrinsic_differences.append(float(np.max(np.abs(K_2 - K))))
        T_2_1 = vda_relative_pose(T_1, T_2)
        R_2_1 = torch.from_numpy(T_2_1[:3, :3]).to(device=device)
        t_2_1 = torch.from_numpy(T_2_1[:3, 3]).to(device=device)
        depth1 = torch.from_numpy(depth1_numpy).to(device=device)
        depth2 = torch.from_numpy(depth2_numpy).to(device=device)
        mask1 = torch.ones_like(depth1, dtype=torch.bool)
        mask2 = torch.ones_like(depth2, dtype=torch.bool)
        # Official VDA uses Ks_cur[i] for both directions of this frame pair.
        error1 = tae_torch(depth1, depth2, R_2_1, t_2_1, K, mask2)
        T_1_2 = np.linalg.inv(T_2_1)
        R_1_2 = torch.from_numpy(T_1_2[:3, :3]).to(device=device)
        t_1_2 = torch.from_numpy(T_1_2[:3, 3]).to(device=device)
        error2 = tae_torch(depth2, depth1, R_1_2, t_1_2, K, mask1)
        error_sum += error1
        error_sum += error2

    pair_count = len(frames) - 1
    tae = error_sum / (2 * pair_count) * 100
    return _metadata(
        status="complete" if not skipped else "partial",
        tae=float(tae),
        evaluated_frame_count=len(frames),
        evaluated_pair_count=pair_count,
        skipped_frames=skipped,
        camera_directory=str(directory),
        camera_source="SCARED frame_data KL and camera-pose",
        scared_raw_pose_convention="world_to_camera",
        vda_pose_convention="camera_to_world",
        relative_transform="inv(T_2_c2w) @ T_1_c2w maps camera_1 to camera_2",
        pose_translation_scale=float(cfg.get("pose_translation_scale", 0.001)),
        tae_mask="all_true_no_additional_scared_evaluation_mask",
        tae_k_usage="first_frame_K_for_both_directions_matching_vda",
        adjacent_intrinsics_all_equal=all(
            difference == 0.0 for difference in intrinsic_differences
        ),
        max_adjacent_intrinsics_abs_difference=max(intrinsic_differences),
    )


__all__ = [
    "TAE_REFERENCE",
    "VDA_TAE_METADATA",
    "camera_index",
    "compute_errors_torch",
    "evaluate_tae",
    "read_scared_camera",
    "tae_torch",
    "vda_relative_pose",
]
