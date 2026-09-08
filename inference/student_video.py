"""VDA-style full-video depth inference for DA3.

Reference: DepthAnything/Video-Depth-Anything, video_depth_anything/video_depth.py.
32 views, 10 reference views (2 anchors + 8 recent views), stride 22.
Only depth is affine-aligned: native window camera poses do not share its gauge.
The callback receives finalized frames exactly once, in original sequence order.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from datasets.transforms import load_precomputed_student_rgb_tensor, load_rgb_tensor

WINDOW = 32
OVERLAP = 10
BLEND = 8
STEP = WINDOW - OVERLAP
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]


class SequenceFrames:
    """Lazy RGB decoding: accept all paths without retaining an entire video in RAM."""

    def __init__(self, paths: Sequence[str | Path], *, resize_mode: str = "resize",
                 height: int = 448, width: int = 560):
        self.paths = list(paths)
        self.resize_mode, self.height, self.width = resize_mode, height, width

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.resize_mode == "precomputed":
            return load_precomputed_student_rgb_tensor(self.paths[index], "zero_one")
        return load_rgb_tensor(self.paths[index], self.height, self.width,
                               self.resize_mode, "zero_one")


def sequence_frames(sequence, dataset_config, *, raw_rgb=False):
    precomputed = sequence.get("preprocessing_identity", "legacy_scared") != "legacy_scared"
    mode = "precomputed" if precomputed and not raw_rgb else dataset_config.get("resize_mode", "resize")
    return SequenceFrames(sequence["frame_paths"], resize_mode=mode,
                          height=int(dataset_config.get("image_height", 448)),
                          width=int(dataset_config.get("image_width", 560)))


def align_disparity(current: np.ndarray, reference: np.ndarray):
    """Least-squares scale/shift on the two corresponding anchors, without GT."""
    x, y = current.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise FloatingPointError("Non-finite window anchor disparity")
    xm, ym = x.mean(), y.mean()
    variance = np.mean((x - xm) ** 2)
    scale = np.mean((x - xm) * (y - ym)) / variance if variance > 1e-12 else 1.0
    # A negative fit reverses depth ordering. Degenerate anchors get a positive
    # scale-only fallback, explicitly recorded in inference metadata.
    fallback = variance <= 1e-12 or not np.isfinite(scale) or scale <= 0
    if fallback:
        scale = float(np.median(y / np.maximum(x, 1e-6)))
        shift = 0.0
    else:
        shift = ym - scale * xm
    return float(scale), float(shift), bool(fallback)


@dataclass
class InferenceStats:
    output_frame_count: int = 0
    window_count: int = 0
    model_input_frame_count: int = 0
    model_forward_seconds: float = 0.0
    sequence_pipeline_seconds: float = 0.0
    alignment_fallback_count: int = 0

    def as_dict(self):
        n = self.output_frame_count
        seconds = self.model_forward_seconds
        return {**vars(self), "window_length": WINDOW, "window_stride": STEP,
                "overlap": OVERLAP, "blend_frames": BLEND,
                "mean_frame_inference_seconds": seconds / n if n else None,
                "mean_frame_inference_ms": seconds * 1000 / n if n else None,
                "inference_fps": n / seconds if seconds else None,
                "mean_frame_pipeline_seconds": self.sequence_pipeline_seconds / n if n else None,
                "timing_scope": "synchronized model forwards / unique output frames; includes repeated anchors and padding; excludes RGB decode, transfers, stitching, GT scoring and export",
                "pipeline_timing_scope": "RGB decode, transfer, model, stitching and output callback; excludes model loading and GT scoring",
                "warmup_excluded": False}


@torch.inference_mode()
def infer_student_video(model, frames, emit: Callable, *, device, amp=True,
                        max_windows=None, emit_window=None) -> dict:
    """Emit (start, disparities[N,H,W], intrinsics[N,3,3]) once per finalized span.

    `emit_window(indices, raw_predictions)` optionally saves camera predictions
    in their original window coordinate system. It must not treat these poses
    as a global trajectory after disparity scale/shift correction.
    """
    if model.training:
        raise ValueError("Full-video inference requires model.eval()")
    if len(frames) == 0:
        raise ValueError("Cannot infer an empty sequence")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive")
    device = torch.device(device)
    starts = list(range(0, len(frames), STEP))
    if max_windows is not None:
        starts = starts[:max_windows]
    stats = InferenceStats()
    started = time.perf_counter()
    previous_ids = None
    anchors = None
    pending_disp = pending_k = None
    next_output = 0
    for window_number, start in enumerate(starts):
        ids = [min(start + j, len(frames) - 1) for j in range(WINDOW)]
        if previous_ids is not None:
            ids[:OVERLAP] = [previous_ids[j] for j in KEYFRAMES]
        # Duplicate padding/anchor frames are decoded only once per window.
        decoded = {j: frames[j] for j in dict.fromkeys(ids)}
        images = torch.stack([decoded[j] for j in ids]).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
            prediction = model(images, include_global_points=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stats.model_forward_seconds += time.perf_counter() - tick
        stats.window_count += 1
        stats.model_input_frame_count += WINDOW
        depth = prediction["depth"][0].float().cpu().numpy()
        if depth.shape != (WINDOW, *images.shape[-2:]) or not np.isfinite(depth).all():
            raise FloatingPointError("Invalid DA3 sequence depth output")
        disparity = 1.0 / np.maximum(depth, 1e-3)
        intrinsics = prediction["intrinsics"][0].float().cpu().numpy()
        if emit_window is not None:
            emit_window(ids, {key: prediction[key][0].float().cpu().numpy()
                              for key in ("intrinsics", "extrinsics")})
        del prediction, images, decoded
        if anchors is None:
            combined_disp, combined_k = disparity, intrinsics
        else:
            scale, shift, fallback = align_disparity(disparity[:2], anchors)
            stats.alignment_fallback_count += int(fallback)
            disparity = np.maximum(disparity * scale + shift, 1e-3)
            weights = np.linspace(0, 1, BLEND, dtype=np.float32)[:, None, None]
            blended = pending_disp * (1 - weights) + disparity[2:OVERLAP] * weights
            blended_k = pending_k * (1 - weights) + intrinsics[2:OVERLAP] * weights
            combined_disp = np.concatenate((blended, disparity[OVERLAP:]))
            combined_k = np.concatenate((blended_k, intrinsics[OVERLAP:]))
        anchors = disparity[KEYFRAMES[:2]].copy() if anchors is None else np.stack((anchors[0], disparity[12]))
        previous_ids = ids
        final_window = window_number == len(starts) - 1
        count = len(combined_disp) if final_window else len(combined_disp) - BLEND
        count = min(count, len(frames) - next_output)
        if count > 0:
            emit(next_output, combined_disp[:count], combined_k[:count])
            next_output += count
        pending_disp = combined_disp[-BLEND:].copy()
        pending_k = combined_k[-BLEND:].copy()
        if next_output == len(frames):
            break
    stats.output_frame_count = next_output
    stats.sequence_pipeline_seconds = time.perf_counter() - started
    return stats.as_dict()
