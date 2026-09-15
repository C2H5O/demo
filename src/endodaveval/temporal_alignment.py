"""Literal Video-Depth-Anything TAE with SCARED I/O adaptation only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

from endodaveval.config import EVALUATION_RESOLUTION_HW
from endodaveval.data import SequenceRecord, index_by_frame_id


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
    return torch.mean(torch.abs(gt - pred) / gt)


def tae_torch(depth1, depth2, R_2_1, T_2_1, K, mask):
    height, width = depth1.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xx, yy = torch.meshgrid(torch.arange(width), torch.arange(height))
    xx, yy = xx.t(), yy.t()
    xx = xx.to(dtype=depth1.dtype, device=depth1.device)
    yy = yy.to(dtype=depth1.dtype, device=depth1.device)
    x = (xx - cx) * depth1 / fx
    y = (yy - cy) * depth1 / fy
    points3d = torch.stack((x.flatten(), y.flatten(), depth1.flatten()), dim=1)
    translation = torch.tensor(T_2_1, dtype=depth1.dtype, device=depth1.device)
    transformed = torch.matmul(points3d, R_2_1.T) + translation
    x_world, y_world, z_world = transformed[:, 0], transformed[:, 1], transformed[:, 2]
    x_plane = torch.round((x_world * fx) / z_world + cx).to(dtype=torch.long)
    y_plane = torch.round((y_world * fy) / z_world + cy).to(dtype=torch.long)
    valid = (x_plane >= 0) & (x_plane < width) & (y_plane >= 0) & (y_plane < height)
    if valid.sum() == 0:
        return 0
    depth_projected = torch.zeros((height, width), dtype=depth1.dtype, device=depth1.device)
    depth_projected[y_plane[valid], x_plane[valid]] = z_world[valid]
    valid = (depth_projected > 0) & (depth2 > 0) & mask
    if valid.sum() == 0:
        return 0
    return compute_errors_torch(depth2[valid], depth_projected[valid])


def camera_index(record: SequenceRecord, relative_directory: str):
    directory = record.keyframe_directory / relative_directory
    return directory, index_by_frame_id(p for p in directory.glob("*.json") if p.is_file())


def preflight_tae(record: SequenceRecord, config: Mapping[str, Any]) -> None:
    if not bool(config.get("enabled", True)):
        return
    directory, cameras = camera_index(
        record, str(config.get("frame_data_relative_directory", "data/frame_data"))
    )
    missing = sorted(set(record.frame_ids) - set(cameras))
    if missing and bool(config.get("require_all_pairs", True)):
        raise FileNotFoundError("TAE dataset cameras missing in {}: {}".format(directory, missing[:20]))


def read_scared_camera(
    path: Path,
    rgb_path: Path,
    output_shape: Tuple[int, int] = EVALUATION_RESOLUTION_HW,
    translation_scale: float = 0.001,
):
    record = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = np.asarray(record["camera-calibration"]["KL"], dtype=np.float64)
    raw_w2c = np.asarray(record["camera-pose"], dtype=np.float64)
    if intrinsics.shape != (3, 3) or raw_w2c.shape != (4, 4):
        raise ValueError("TAE requires KL[3,3] and camera-pose[4,4]: {}".format(path))
    if not np.isfinite(intrinsics).all() or not np.isfinite(raw_w2c).all():
        raise ValueError("Non-finite TAE camera: {}".format(path))
    rotation = raw_w2c[:3, :3]
    if (
        intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], [0, 0, 1])
    ):
        raise ValueError("Invalid pinhole intrinsics: {}".format(path))
    if (
        not np.allclose(raw_w2c[3], [0, 0, 0, 1])
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
    ):
        raise ValueError("SCARED camera-pose must be a rigid world-to-camera transform")
    w2c_metres = raw_w2c.copy()
    w2c_metres[:3, 3] *= float(translation_scale)
    pose_c2w = np.linalg.inv(w2c_metres)
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read RGB for TAE: {}".format(rgb_path))
    original_height, original_width = image.shape[:2]
    sx = output_shape[1] / original_width
    sy = output_shape[0] / original_height
    resized_intrinsics = np.diag([sx, sy, 1.0]) @ intrinsics
    return resized_intrinsics, pose_c2w


def vda_relative_pose(T_1_c2w, T_2_c2w):
    return np.linalg.inv(T_2_c2w) @ T_1_c2w


def evaluate_tae(
    record: SequenceRecord,
    aligned_depth_by_id: Mapping[int, np.ndarray],
    frame_ids: Sequence[int],
    config: Mapping[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    if not bool(config.get("enabled", True)):
        return {**VDA_TAE_METADATA, "status": "disabled", "tae": None}
    directory, cameras = camera_index(
        record, str(config.get("frame_data_relative_directory", "data/frame_data"))
    )
    strict = bool(config.get("require_all_pairs", True))
    frames = []
    skipped = []
    for identifier in frame_ids:
        if identifier not in cameras:
            skipped.append({"frame_id": identifier, "reason": "missing_dataset_camera"})
            continue
        # The sequence-global float64 lstsq in the spatial path promotes the
        # aligned depth to float64; retain that production dtype for VDA TAE.
        depth = np.asarray(aligned_depth_by_id[identifier], dtype=np.float64)
        if depth.shape != EVALUATION_RESOLUTION_HW:
            raise ValueError("TAE aligned depth must be 256x320")
        K, pose = read_scared_camera(
            cameras[identifier],
            record.rgb_by_id[identifier],
            EVALUATION_RESOLUTION_HW,
            float(config.get("pose_translation_scale", 0.001)),
        )
        frames.append((identifier, depth, K, pose))
    if skipped and strict:
        raise RuntimeError("TAE frame {} missing dataset camera in {}".format(skipped[0]["frame_id"], directory))
    if len(frames) < 2:
        if strict:
            raise RuntimeError("TAE requires at least two frames")
        return {**VDA_TAE_METADATA, "status": "unavailable", "tae": None}
    torch_device = torch.device(device)
    error_sum = 0.0
    for index in range(len(frames) - 1):
        _, depth1_np, K, pose1 = frames[index]
        _, depth2_np, _, pose2 = frames[index + 1]
        transform = vda_relative_pose(pose1, pose2)
        depth1 = torch.from_numpy(depth1_np).to(torch_device)
        depth2 = torch.from_numpy(depth2_np).to(torch_device)
        R = torch.from_numpy(transform[:3, :3]).to(torch_device)
        t = torch.from_numpy(transform[:3, 3]).to(torch_device)
        error_sum += tae_torch(depth1, depth2, R, t, K, torch.ones_like(depth2, dtype=torch.bool))
        inverse = np.linalg.inv(transform)
        inverse_R = torch.from_numpy(inverse[:3, :3]).to(torch_device)
        inverse_t = torch.from_numpy(inverse[:3, 3]).to(torch_device)
        error_sum += tae_torch(depth2, depth1, inverse_R, inverse_t, K, torch.ones_like(depth1, dtype=torch.bool))
    pair_count = len(frames) - 1
    tae = error_sum / (2 * pair_count) * 100
    return {
        **VDA_TAE_METADATA,
        "status": "complete" if not skipped else "partial",
        "tae": float(tae),
        "evaluated_frame_count": len(frames),
        "evaluated_pair_count": pair_count,
        "skipped_frames": skipped,
        "camera_directory": str(directory),
        "evaluation_resolution_hw": list(EVALUATION_RESOLUTION_HW),
        "camera_source": "SCARED data/frame_data KL and camera-pose",
        "scared_raw_pose_convention": "world_to_camera",
        "vda_pose_convention": "camera_to_world",
        "relative_transform": "inv(T_2_c2w) @ T_1_c2w",
        "pose_translation_scale": float(config.get("pose_translation_scale", 0.001)),
        "tae_k_usage": "first_frame_K_for_both_directions_matching_vda",
    }
