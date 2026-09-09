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
from inference.kv_sampling import KVSamplingConfig, WindowFrameMetadata

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
    from evaluation.scared_gt import extract_frame_id

    precomputed = sequence.get("preprocessing_identity", "legacy_scared") != "legacy_scared"
    mode = "precomputed" if precomputed and not raw_rgb else dataset_config.get("resize_mode", "resize")
    frames = SequenceFrames(sequence["frame_paths"], resize_mode=mode,
                            height=int(dataset_config.get("image_height", 448)),
                            width=int(dataset_config.get("image_width", 560)))
    frames.absolute_frame_ids = [extract_frame_id(path) for path in sequence["frame_paths"]]
    frames.absolute_id_source = "dataset_filename"
    return frames


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
    peak_cuda_memory_allocated_bytes: int | None = None
    peak_cuda_memory_reserved_bytes: int | None = None

    def as_dict(self):
        n = self.output_frame_count
        seconds = self.model_forward_seconds
        return {**vars(self), "window_length": WINDOW, "window_stride": STEP,
                "overlap": OVERLAP, "blend_frames": BLEND,
                "mean_frame_inference_seconds": seconds / n if n else None,
                "mean_frame_inference_ms": seconds * 1000 / n if n else None,
                "mean_window_model_forward_seconds": seconds / self.window_count if self.window_count else None,
                "mean_window_model_forward_ms": seconds * 1000 / self.window_count if self.window_count else None,
                "inference_fps": n / seconds if seconds else None,
                "mean_frame_pipeline_seconds": self.sequence_pipeline_seconds / n if n else None,
                "pipeline_fps": n / self.sequence_pipeline_seconds if self.sequence_pipeline_seconds else None,
                "timing_scope": "synchronized model forwards including KV selection and gather / unique output frames; includes repeated anchors and padding; excludes RGB decode, transfers, stitching, GT scoring and export",
                "pipeline_timing_scope": "RGB decode, transfer, KV policy setup/selection/gather/restore, model, stitching, audit and output callback; excludes model loading, GT scoring and final result serialization",
                "warmup_excluded": False}


@torch.inference_mode()
def infer_student_video(model, frames, emit: Callable, *, device, amp=True,
                        max_windows=None, emit_window=None, kv_sampling=None) -> dict:
    """Run unchanged VDA stitching with a sequence-scoped inference KV policy."""
    from inference.da3_kv_attention import DA3KVAttention

    started = time.perf_counter()
    config = KVSamplingConfig.from_mapping(kv_sampling)
    with DA3KVAttention(model, config, WINDOW) as sampling:
        result = _infer_student_video(model, frames, emit, device=device, amp=amp,
                                      max_windows=max_windows, emit_window=emit_window,
                                      sampling=sampling, started=started)
    # Include policy setup, audit bookkeeping and adapter restoration in the
    # sequence end-to-end measurement, in addition to the existing RGB pipeline.
    seconds = time.perf_counter() - started
    count = result["output_frame_count"]
    result["sequence_pipeline_seconds"] = seconds
    result["mean_frame_pipeline_seconds"] = seconds / count if count else None
    result["pipeline_fps"] = count / seconds if seconds else None
    return result


def _infer_student_video(model, frames, emit: Callable, *, device, amp,
                         max_windows, emit_window, sampling, started) -> dict:
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
    previous_ids = None
    previous_padding = None
    absolute_ids = getattr(frames, "absolute_frame_ids", range(len(frames)))
    if len(absolute_ids) != len(frames):
        raise ValueError("Absolute frame IDs must match the input sequence")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    anchors = None
    pending_disp = pending_k = None
    next_output = 0
    for window_number, start in enumerate(starts):
        ids = [min(start + j, len(frames) - 1) for j in range(WINDOW)]
        padding = [start + j >= len(frames) for j in range(WINDOW)]
        roles = ["new"] * WINDOW
        if previous_ids is not None:
            ids[:OVERLAP] = [previous_ids[j] for j in KEYFRAMES]
            padding[:OVERLAP] = [previous_padding[j] for j in KEYFRAMES]
            # The same source list supplies the alignment anchors and BLEND
            # recent frames consumed by the unchanged stitching code below.
            roles[:OVERLAP] = ["key"] * (OVERLAP - BLEND) + ["overlap"] * BLEND
        metadata = WindowFrameMetadata(
            window_id=window_number, frame_positions=tuple(ids),
            absolute_frame_ids=tuple(absolute_ids[position] for position in ids),
            frame_roles=tuple(roles), is_padding=tuple(padding),
            first_window=previous_ids is None,
            absolute_id_source=getattr(frames, "absolute_id_source", "sequence_position"),
        )
        # Duplicate padding/anchor frames are decoded only once per window.
        decoded = {j: frames[j] for j in dict.fromkeys(ids)}
        images = torch.stack([decoded[j] for j in ids]).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        sampling.begin_window(metadata, images)
        with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
            prediction = model(images, include_global_points=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stats.model_forward_seconds += time.perf_counter() - tick
        sampling.finish_window()
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
        previous_padding = padding
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
    if device.type == "cuda":
        stats.peak_cuda_memory_allocated_bytes = torch.cuda.max_memory_allocated(device)
        stats.peak_cuda_memory_reserved_bytes = torch.cuda.max_memory_reserved(device)
    return {**stats.as_dict(), **sampling.summary()}
