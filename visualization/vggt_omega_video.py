"""Export an online VGGT-Omega reconstruction with baseline-T VDA semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from datasets.scared_clip_dataset import make_scared_rgb_dataset
from evaluation.evaluate_crossclip_projection import (
    VGGT_OMEGA_SOURCE,
    _evaluation_model,
    _require_scared_8_9,
)
from evaluation.evaluate_vda import _SequencePredictionSpool
from evaluation.scared_gt import extract_frame_id
from inference.vggt_omega_video import VGGTOmegaSequenceFrames, infer_vggt_omega_video
from utils.config import ensure_dir, load_config

from visualization.crossclip_projection import (
    _adaptive_range,
    _depth_to_magma,
    _write_binary_ply,
)


def _visualization_dataset(config: Dict[str, Any], eval_config: Dict[str, Any], split: str) -> Any:
    dataset_config = dict(config["dataset"])
    rgb_root = eval_config.get("rgb_root")
    if not rgb_root:
        raise ValueError("vggt_omega_baseline_vda_evaluation.rgb_root is required")
    dataset_config["root"] = str(rgb_root)
    dataset_config["legacy_scared_root"] = str(rgb_root)
    dataset_config["canonical_root"] = None
    dataset_config["frame_source"] = str(
        eval_config.get("frame_source", dataset_config.get("frame_source", "auto"))
    )
    dataset_config.update(clip_length=1, sample_stride=1, window_stride=1)
    dataset_config["drop_incomplete_clip"] = False
    dataset_config["highlight"] = {"enabled": False}
    dataset = make_scared_rgb_dataset(dataset_config, split)
    _require_scared_8_9({str(item["sequence_id"]): item for item in dataset.sequences})
    return dataset


def _rgb_uint8(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected a single RGB tensor, got {}".format(tuple(image.shape)))
    return np.round(
        image.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    ).astype(np.uint8)


def export_vggt_omega_video(
    config_path: Path,
    *,
    split: str = "test",
    sequence_index: int = 0,
    output_root: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    point_stride: Optional[int] = None,
    max_windows: Optional[int] = None,
) -> Path:
    """Export one raw SCARED sequence using online VGGT-Omega inference."""
    if split != "test":
        raise ValueError("Baseline-T VGGT-Omega visualization is fixed to the SCARED test split")
    config = load_config(config_path)
    eval_config = dict(config.get("vggt_omega_baseline_vda_evaluation", {}))
    visual = dict(config.get("vggt_omega_baseline_visualization", {}))
    if not visual:
        visual = dict(config.get("visualization", {}))
    minimum_depth = float(min_depth if min_depth is not None else visual.get("min_depth", 0.1))
    maximum_depth = float(max_depth if max_depth is not None else visual.get("max_depth", 10.0))
    stride = int(point_stride if point_stride is not None else visual.get("point_stride", 4))
    if (
        not np.isfinite(minimum_depth)
        or not np.isfinite(maximum_depth)
        or minimum_depth < 0.0
        or maximum_depth <= minimum_depth
    ):
        raise ValueError("Visualization depth range must satisfy 0 <= min_depth < max_depth")
    if stride <= 0:
        raise ValueError("point_stride must be positive")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive")
    adaptive_percentiles = tuple(
        float(value) for value in visual.get("adaptive_percentiles", [5.0, 95.0])
    )
    if (
        len(adaptive_percentiles) != 2
        or not 0.0 <= adaptive_percentiles[0] < adaptive_percentiles[1] <= 100.0
    ):
        raise ValueError("adaptive_percentiles requires 0 <= low < high <= 100")

    dataset = _visualization_dataset(config, eval_config, split)
    if not 0 <= sequence_index < len(dataset.sequences):
        raise IndexError(
            "sequence_index={} is outside [0,{})".format(sequence_index, len(dataset.sequences))
        )
    sequence = dataset.sequences[sequence_index]
    frames = VGGTOmegaSequenceFrames(
        sequence["frame_paths"],
        image_resolution=int(eval_config.get("preprocessing", {}).get("image_resolution", 512)),
        mode=str(eval_config.get("preprocessing", {}).get("mode", "balanced")),
        patch_size=int(eval_config.get("preprocessing", {}).get("patch_size", 16)),
    )
    # Probe one frame before constructing the disk spool so its dimensions match
    # the native VGGT-Omega model output rather than the student's score grid.
    probe = frames.load_indices([0])
    model_height, model_width = (int(value) for value in probe.shape[-2:])
    del probe

    root = Path(output_root or visual.get("output_dir", "outputs/baseline_T/visualization"))
    output = ensure_dir(
        root / VGGT_OMEGA_SOURCE / str(sequence["sequence_id"]).replace("/", "_") / "full_sequence"
    )
    directories = {
        name: ensure_dir(output / name)
        for name in (
            "rgb",
            "depth",
            "depth_fixed",
            "depth_adaptive",
            "panels",
            "pointcloud_local",
            "camera_windows",
        )
    }

    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA visualization requested but unavailable")
    checkpoint = checkpoint_path or Path(str(config["teacher"]["pretrained_checkpoint"]))
    model = _evaluation_model(checkpoint, config, device, VGGT_OMEGA_SOURCE)
    spool = _SequencePredictionSpool(output, len(frames), model_height, model_width)
    intrinsics = np.zeros((len(frames), 3, 3), dtype=np.float32)
    camera_seen = np.zeros(len(frames), dtype=bool)
    samples = []
    sample_period = max(1, (len(frames) + 127) // 128)
    frame_ids = [extract_frame_id(path) for path in sequence["frame_paths"]]
    window_number = 0

    try:
        def emit(start: int, disparities: np.ndarray, ks: np.ndarray) -> None:
            end = start + len(disparities)
            if end > len(frames):
                raise RuntimeError("VDA emitter returned frames outside the sequence")
            spool.add(range(start, end), disparities)
            intrinsics[start:end] = np.asarray(ks, dtype=np.float32)
            camera_seen[start:end] = True
            selected = [
                offset
                for offset in range(len(disparities))
                if (start + offset) % sample_period == 0
            ]
            if selected:
                samples.append(
                    (1.0 / np.maximum(disparities[selected, ::16, ::16], 1.0e-3)).reshape(-1)
                )

        def emit_window(positions, cameras) -> None:
            nonlocal window_number
            np.savez(
                directories["camera_windows"] / "window_{:06d}.npz".format(window_number),
                frame_positions=np.asarray(positions, dtype=np.int64),
                absolute_frame_ids=np.asarray([frame_ids[position] for position in positions], dtype=np.int64),
                intrinsics=np.asarray(cameras["intrinsics"], dtype=np.float32),
                extrinsics_w2c=np.asarray(cameras["extrinsics"], dtype=np.float32),
                coordinate_system="native independent VGGT-Omega window; poses are not fused",
            )
            window_number += 1

        timing = infer_vggt_omega_video(
            model,
            frames,
            emit,
            device=device,
            amp=bool(eval_config.get("amp", True)),
            max_windows=max_windows,
            inspect_window=None,
            emit_window=emit_window,
        )
        spool.flush()
        del model
        model = None
        output_count = int(timing["output_frame_count"])
        if output_count <= 0 or output_count > len(frames):
            raise RuntimeError(
                "VGGT-Omega produced an invalid finalized frame count: {} (sequence length {})".format(
                    output_count, len(frames)
                )
            )
        if not np.all(camera_seen[:output_count]):
            missing = np.flatnonzero(~camera_seen[:output_count]).tolist()
            raise RuntimeError("Missing fused intrinsics for output frames {}".format(missing[:20]))

        sampled_depth = np.concatenate(samples).astype(np.float32, copy=False)
        sample_valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
        adaptive_low, adaptive_high = _adaptive_range(
            sampled_depth, sample_valid, adaptive_percentiles
        )
        y, x = np.indices((model_height, model_width))
        pixel_grid = np.stack((x, y, np.ones_like(x))).reshape(3, -1)
        for position in range(output_count):
            image = frames.load_indices([position])[0]
            rgb = _rgb_uint8(image)
            disparity = spool.prediction(position)
            depth = 1.0 / np.maximum(disparity, 1.0e-3)
            valid = np.isfinite(depth) & (depth > 0.0)
            fixed = _depth_to_magma(
                depth,
                valid & (depth >= minimum_depth) & (depth <= maximum_depth),
                minimum_depth,
                maximum_depth,
            )
            adaptive = _depth_to_magma(depth, valid, adaptive_low, adaptive_high)
            stem = "{:06d}_{}".format(position, Path(sequence["frame_paths"][position]).stem)
            Image.fromarray(rgb).save(directories["rgb"] / (stem + ".png"))
            np.save(directories["depth"] / (stem + ".npy"), depth.astype(np.float32))
            Image.fromarray(fixed).save(directories["depth_fixed"] / (stem + ".png"))
            Image.fromarray(adaptive).save(directories["depth_adaptive"] / (stem + ".png"))
            Image.fromarray(np.concatenate((rgb, fixed, adaptive), axis=1)).save(
                directories["panels"] / (stem + ".png")
            )
            K = intrinsics[position].astype(np.float64)
            if not np.isfinite(K).all() or abs(float(np.linalg.det(K))) <= 1.0e-9:
                raise RuntimeError("VGGT-Omega fused intrinsics are invalid at frame {}".format(position))
            points = ((np.linalg.inv(K) @ pixel_grid) * depth.reshape(1, -1)).T.reshape(
                model_height, model_width, 3
            )
            sampled = np.zeros_like(valid)
            sampled[::stride, ::stride] = True
            mask = valid & sampled
            _write_binary_ply(
                directories["pointcloud_local"] / (stem + ".ply"),
                points[mask].astype(np.float32),
                rgb[mask],
            )
        np.save(output / "fused_intrinsics.npy", intrinsics[:output_count])
        metadata: Dict[str, Any] = {
            "source": VGGT_OMEGA_SOURCE,
            "model": "VGGT-Omega-1B-512",
            "checkpoint": str(checkpoint),
            "split": split,
            "sequence_id": sequence["sequence_id"],
            "sequence_index": sequence_index,
            "absolute_frame_ids": frame_ids[:output_count],
            "frame_paths": [str(path) for path in sequence["frame_paths"][:output_count]],
            "inference": timing,
            "inference_mode": "full_sequence_vda_windows_online_teacher",
            "teacher_preprocessing": frames.metadata(),
            "model_input_shape_hw": [model_height, model_width],
            "complete_sequence": output_count == len(frames),
            "camera_window_count": window_number,
            "coordinate_system": "camera-local points reconstructed from fused VGGT-Omega depth and blended intrinsics",
            "global_pointcloud": "not exported: disparity affine fusion is not a 3D similarity transform",
            "fixed_depth_range": [minimum_depth, maximum_depth],
            "adaptive_percentiles": list(adaptive_percentiles),
            "adaptive_depth_range": [adaptive_low, adaptive_high],
            "adaptive_sampling": "up to 128 frames, spatial stride 16",
            "panel_order": ["rgb", "fixed depth", "adaptive depth"],
            "point_stride": stride,
            "teacher_cache_used": False,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    finally:
        if model is not None:
            del model
        spool.close()
    print("Exported online VGGT-Omega sequence visualization to {}".format(output))
    return output


__all__ = ["export_vggt_omega_video"]
