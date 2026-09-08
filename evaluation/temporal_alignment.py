"""Depth Any Video TAE: bidirectional camera reprojection AbsRel, in percent.

Definition: https://arxiv.org/html/2410.10815v2 (Eq. 7).
Operational reference: Video-Depth-Anything/benchmark/eval/eval_tae.py.
We normalize by the actual number of valid directed adjacent-frame comparisons
(2*(T-1) for a complete sequence), use a deterministic nearest-surface z-buffer,
and reject empty projections rather than awarding zero error. This is a SCARED
adaptation, not a bit-identical reproduction of the ScanNet benchmark script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.scared_gt import extract_frame_id


def project_depth(source, target, source_k, target_k, source_to_target):
    """Nearest-pixel forward projection with deterministic minimum-Z collisions."""
    source, target = np.asarray(source), np.asarray(target)
    y, x = np.indices(source.shape)
    pixels = np.stack((x.ravel(), y.ravel(), np.ones(source.size)))
    points = (np.linalg.inv(source_k) @ pixels) * source.ravel()[None]
    points = source_to_target[:3, :3] @ points + source_to_target[:3, 3, None]
    projected = target_k @ points
    good = np.isfinite(points).all(axis=0) & (points[2] > 1e-6)
    good &= np.isfinite(source).ravel() & (source.ravel() > 0)
    u = np.zeros(source.size, dtype=np.int64)
    v = u.copy()
    u[good] = np.rint(projected[0, good] / projected[2, good]).astype(np.int64)
    v[good] = np.rint(projected[1, good] / projected[2, good]).astype(np.int64)
    height, width = target.shape
    good &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    warped = np.full(target.size, np.inf, dtype=np.float64)
    np.minimum.at(warped, v[good] * width + u[good], points[2, good])
    warped = warped.reshape(target.shape)
    valid = np.isfinite(warped) & np.isfinite(target) & (target > 0)
    if not valid.any():
        return None, 0
    return float(np.mean(np.abs(warped[valid] - target[valid]) / target[valid])), int(valid.sum())


def camera_index(sequence, eval_config):
    from evaluation.evaluate_vda import _find_gt_keyframe
    cfg = eval_config.get("tae", {})
    keyframe = ( _find_gt_keyframe(sequence, Path(eval_config["gt_root"]))
                 if eval_config.get("gt_root") else Path(sequence["keyframe_directory"]) )
    directory = keyframe / cfg.get("frame_data_relative_directory", "data/frame_data")
    result = {}
    for path in sorted(directory.glob("*.json")):
        frame_id = extract_frame_id(path)
        if frame_id in result:
            raise ValueError("Duplicate TAE camera frame ID {} in {}".format(frame_id, directory))
        result[frame_id] = path
    return directory, result


def read_scared_camera(path, rgb_path, output_shape, translation_scale=0.001):
    """SCARED KL and world-to-camera camera-pose; translation mm -> metres.

    See Ruyi-Zha/endosurf/data/scared2019/preprocess.py for this convention.
    RGB must have the calibrated full field of view (only direct resizing).
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    k = np.asarray(record["camera-calibration"]["KL"], dtype=np.float64)
    pose = np.asarray(record["camera-pose"], dtype=np.float64)
    if k.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("TAE requires KL[3,3] and camera-pose[4,4]: {}".format(path))
    if not (np.isfinite(k).all() and np.isfinite(pose).all()):
        raise ValueError("Non-finite TAE camera: {}".format(path))
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("Invalid pinhole intrinsics: {}".format(path))
    r = pose[:3, :3]
    if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(r @ r.T, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(r), 1, atol=1e-3):
        raise ValueError("TAE camera-pose is not a rigid world-to-camera transform")
    if not np.isfinite(translation_scale) or translation_scale <= 0:
        raise ValueError("pose_translation_scale must be finite and positive")
    pose = pose.copy()
    pose[:3, 3] *= translation_scale
    with Image.open(rgb_path) as image:
        width, height = image.size
    k = np.diag([output_shape[1] / width, output_shape[0] / height, 1.0]) @ k
    return k, pose


def evaluate_tae(sequence, spool, spatial_result, eval_config):
    cfg = eval_config.get("tae", {})
    if not cfg.get("enabled", True):
        return {"status": "disabled", "tae": None}
    directory, cameras = camera_index(sequence, eval_config)
    paths = sequence["frame_paths"]
    ids = [extract_frame_id(path) for path in paths]
    step = int(cfg.get("frame_id_step", 1))
    if step <= 0:
        raise ValueError("TAE frame_id_step must be positive")
    strict = bool(cfg.get("require_all_pairs", True))
    errors, valid_pixels, skipped = [], 0, []
    eligible = 0
    gap_count = 0
    previous = None
    scale, shift = spatial_result["disparity_scale"], spatial_result["disparity_shift"]
    for pos, identifier in enumerate(ids):
        current = None
        if spool.counts[pos] > 0 and identifier in cameras:
            k, pose = read_scared_camera(cameras[identifier], paths[pos],
                                        (spool.height, spool.width),
                                        float(cfg.get("pose_translation_scale", 0.001)))
            disparity = np.maximum(spool.prediction(pos), 1e-3)
            depth = np.clip(1.0 / np.maximum(scale * disparity + shift, 1e-3), 1e-3, 100.0)
            current = (depth, k, pose)
        if pos > 0:
            if identifier - ids[pos - 1] != step:
                gap_count += 1
                previous = current
                continue
            eligible += 1
            reason = None
            if previous is None or current is None:
                reason = "missing_prediction_or_dataset_camera"
            else:
                first, k1, e1 = previous
                second, k2, e2 = current
                transform = e2 @ np.linalg.inv(e1)
                forward, n1 = project_depth(first, second, k1, k2, transform)
                backward, n2 = project_depth(second, first, k2, k1, np.linalg.inv(transform))
                if forward is None or backward is None:
                    reason = "empty_projection"
                else:
                    errors.extend((forward, backward))
                    valid_pixels += n1 + n2
            if reason:
                if strict:
                    raise RuntimeError("TAE {} -> {}: {}; camera directory: {}".format(ids[pos-1], identifier, reason, directory))
                skipped.append({"frame_ids": [ids[pos-1], identifier], "reason": reason})
        previous = current
    if strict and not errors:
        raise RuntimeError("No valid adjacent-frame TAE pairs in {}".format(sequence["sequence_id"]))
    return {"status": "complete" if errors and not skipped and not gap_count else "partial" if errors else "unavailable",
            "tae": float(np.mean(errors) * 100) if errors else None,
            "unit": "percent", "evaluated_pair_count": len(errors) // 2,
            "eligible_pair_count": eligible, "frame_id_step": step,
            "nonconsecutive_pair_count": gap_count,
            "valid_projected_pixel_count": valid_pixels,
            "skipped_pairs": skipped, "camera_directory": str(directory),
            "camera_source": "SCARED frame_data KL and camera-pose (world_to_camera)",
            "pose_translation_scale": float(cfg.get("pose_translation_scale", 0.001)),
            "alignment": "same single GT disparity scale/shift as spatial metrics, shared by entire sequence",
            "projection": "bidirectional nearest-pixel forward splat with minimum-Z buffer; no optical flow or predicted poses",
            "normalization": "100 * mean of valid directed pair AbsRel; denominator is target prediction"}
