"""Export a complete student sequence using the evaluation inference pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.evaluate_crossclip_projection import (
    OFFICIAL_DA3_SMALL_SOURCE, _evaluation_model,
)
from evaluation.evaluate_vda import _SequencePredictionSpool
from evaluation.scared_gt import extract_frame_id
from inference.student_video import infer_student_video, sequence_frames
from inference.kv_sampling import resolve_kv_sampling
from utils.config import ensure_dir


def export_student_video(config, dataset, sequence_index, output_root, source,
                         checkpoint_path, split, min_depth, max_depth, point_stride):
    from visualization.crossclip_projection import _depth_to_magma, _write_binary_ply
    if not 0 <= sequence_index < len(dataset.sequences):
        raise IndexError("sequence_index={} is outside [0,{})".format(sequence_index, len(dataset.sequences)))
    sequence = dataset.sequences[sequence_index]
    baseline = source == OFFICIAL_DA3_SMALL_SOURCE
    eval_config = config.get("da3_small_baseline_vda_evaluation" if baseline else "vda_evaluation", {})
    visual = config.get("da3_small_baseline_visualization" if baseline else "visualization", {})
    frames = sequence_frames(sequence, config["dataset"], raw_rgb=bool(eval_config.get("rgb_root")))
    output = ensure_dir(Path(output_root) / source / sequence["sequence_id"].replace("/", "_") / "full_sequence")
    directories = {name: ensure_dir(output / name) for name in
                   ("rgb", "depth", "depth_fixed", "depth_adaptive", "panels", "pointcloud_local", "camera_windows")}
    device = torch.device(str(config.get("device", "cuda")))
    model = _evaluation_model(checkpoint_path, config, device,
                              OFFICIAL_DA3_SMALL_SOURCE if baseline else "trained_student_checkpoint")
    spool = _SequencePredictionSpool(output, len(frames), frames.height, frames.width)
    intrinsics = np.empty((len(frames), 3, 3), dtype=np.float32)
    samples = []
    window_number = 0
    ids = [extract_frame_id(p) for p in sequence["frame_paths"]]
    try:
        def emit(start, disparities, ks):
            spool.add(range(start, start + len(disparities)), disparities)
            intrinsics[start:start + len(ks)] = ks
            # Deterministic spatial/temporal sampling bounds percentile RAM on long videos.
            for offset, disparity in enumerate(disparities):
                if (start + offset) % max(1, len(frames) // 128) == 0:
                    samples.append((1.0 / np.maximum(disparity[::16, ::16], 1e-3)).ravel())
        def emit_window(positions, cameras):
            nonlocal window_number
            np.savez(directories["camera_windows"] / "window_{:06d}.npz".format(window_number),
                     frame_positions=np.asarray(positions), absolute_frame_ids=np.asarray([ids[p] for p in positions]),
                     intrinsics=cameras["intrinsics"], extrinsics_w2c=cameras["extrinsics"],
                     coordinate_system="native independent DA3 window; not aligned to fused disparity")
            window_number += 1
        timing = infer_student_video(model, frames, emit, device=device,
                                     amp=bool(eval_config.get("amp", True)), emit_window=emit_window,
                                     kv_sampling=resolve_kv_sampling(config))
        spool.flush()
        del model
        percentiles = tuple(visual.get("adaptive_percentiles", [5.0, 95.0]))
        if len(percentiles) != 2 or not 0 <= percentiles[0] < percentiles[1] <= 100:
            raise ValueError("adaptive_percentiles requires 0 <= low < high <= 100")
        low, high = np.percentile(np.concatenate(samples), percentiles)
        high = max(high, low + 1e-6)
        y, x = np.indices((frames.height, frames.width))
        pixel_grid = np.stack((x, y, np.ones_like(x))).reshape(3, -1)
        for position, path in enumerate(frames.paths):
            rgb = np.round(frames[position].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            depth = 1.0 / np.maximum(spool.prediction(position), 1e-3)
            valid = np.isfinite(depth) & (depth > 0)
            fixed = _depth_to_magma(depth, valid & (depth >= min_depth) & (depth <= max_depth), min_depth, max_depth)
            adaptive = _depth_to_magma(depth, valid, float(low), float(high))
            stem = "{:06d}_{}".format(position, Path(path).stem)
            for key, value in (("rgb", rgb), ("depth_fixed", fixed), ("depth_adaptive", adaptive),
                               ("panels", np.concatenate((rgb, fixed, adaptive), axis=1))):
                Image.fromarray(value).save(directories[key] / (stem + ".png"))
            np.save(directories["depth"] / (stem + ".npy"), depth.astype(np.float32))
            points = ((np.linalg.inv(intrinsics[position]) @ pixel_grid) * depth.reshape(1, -1)).T.reshape(*depth.shape, 3)
            sampled = np.zeros_like(valid)
            sampled[::point_stride, ::point_stride] = True
            mask = valid & sampled
            _write_binary_ply(directories["pointcloud_local"] / (stem + ".ply"), points[mask].astype(np.float32), rgb[mask])
        np.save(output / "fused_intrinsics.npy", intrinsics)
        metadata = {
            "source": source, "sequence_id": sequence["sequence_id"], "sequence_index": sequence_index,
            "split": split, "absolute_frame_ids": ids, "frame_paths": [str(p) for p in frames.paths],
            "checkpoint": str(config["student"]["checkpoint"]) if baseline else str(checkpoint_path),
            "inference": timing, "inference_mode": "full_sequence_vda_windows",
            "coordinate_system": "camera-local points reconstructed from fused depth and blended predicted K; native poses saved separately per window",
            "global_pointcloud": "not exported: disparity affine fusion is not a 3D similarity transform",
            "fixed_depth_range": [min_depth, max_depth], "adaptive_depth_range": [float(low), float(high)],
            "adaptive_percentiles": percentiles, "adaptive_sampling": "at most about 256 frames, spatial stride 16",
            "point_stride": point_stride,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    finally:
        spool.close()
    print("Exported full student sequence to {}".format(output))
    return output
