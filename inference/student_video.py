"""VDA-style full-video depth inference for DA3.

Reference: DepthAnything/Video-Depth-Anything, video_depth_anything/video_depth.py.
32 views, 10 reference views (2 anchors + 8 recent views), stride 22.
Only depth is affine-aligned: native window camera poses do not share its gauge.
The callback receives finalized frames exactly once, in original sequence order.
"""
from __future__ import annotations

import multiprocessing
import time
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from datasets.transforms import load_precomputed_student_rgb_tensor, load_rgb_tensor

WINDOW = 32
OVERLAP = 10
BLEND = 8
STEP = WINDOW - OVERLAP
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
_resolution_audit_printed = False


def _load_frame_tensor(
    path: str | Path, resize_mode: str, height: int, width: int
) -> torch.Tensor:
    if resize_mode == "precomputed":
        image = load_precomputed_student_rgb_tensor(path, "zero_one")
        if image.shape[-2:] != (height, width):
            image = (
                F.interpolate(
                    image.unsqueeze(0),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(0)
                .clamp_(0, 1)
            )
        return image
    return load_rgb_tensor(path, height, width, resize_mode, "zero_one")


def _load_frame_worker(task):
    """Pickleable CPU-only worker. Never receives a model or CUDA tensor."""
    index, path, resize_mode, height, width = task
    tensor = _load_frame_tensor(path, resize_mode, height, width)
    return int(index), tensor.contiguous().numpy()


class _PrefetchedFrames:
    def __init__(
        self,
        loader: "SequenceFrames",
        indices: Sequence[int],
        futures: Dict[int, Future],
    ) -> None:
        self.loader = loader
        self.indices = tuple(int(index) for index in indices)
        self.futures = futures

    def result(self) -> list[torch.Tensor]:
        return self.loader._resolve_prefetch(self.indices, self.futures)


class SequenceFrames:
    """Streaming RGB loader with optional spawn workers, lookahead, and LRU cache."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        resize_mode: str = "resize",
        height: int = 448,
        width: int = 560,
        num_workers: int = 1,
        frame_cache_size: int = 0,
    ) -> None:
        self.paths = [str(path) for path in paths]
        self.resize_mode = str(resize_mode)
        self.height, self.width = int(height), int(width)
        self.num_workers = int(num_workers)
        self.frame_cache_size = int(frame_cache_size)
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if self.frame_cache_size < 0:
            raise ValueError("frame_cache_size must be >= 0")
        self.multiprocessing_context = "spawn" if self.num_workers > 1 else None
        self.loader_backend = (
            "ProcessPoolExecutor(spawn)" if self.num_workers > 1 else "serial"
        )
        self._executor = (
            ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            if self.num_workers > 1
            else None
        )
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._inflight: Dict[int, Future] = {}
        self.closed = False

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        index = int(index)
        if index < 0 or index >= len(self.paths):
            raise IndexError(index)
        return _load_frame_tensor(
            self.paths[index], self.resize_mode, self.height, self.width
        )

    def _cache_get(self, index: int) -> Optional[torch.Tensor]:
        value = self._cache.pop(index, None)
        if value is not None:
            self._cache[index] = value
        return value

    def _cache_put(self, index: int, value: torch.Tensor) -> None:
        if self.frame_cache_size == 0:
            return
        self._cache.pop(index, None)
        self._cache[index] = value
        while len(self._cache) > self.frame_cache_size:
            self._cache.popitem(last=False)

    def prefetch_indices(self, indices: Sequence[int]) -> _PrefetchedFrames:
        if self.closed:
            raise RuntimeError("SequenceFrames is closed")
        requested = tuple(int(index) for index in indices)
        for index in requested:
            if index < 0 or index >= len(self.paths):
                raise IndexError(index)
        futures: Dict[int, Future] = {}
        if self._executor is not None:
            for index in dict.fromkeys(requested):
                if index in self._cache:
                    continue
                future = self._inflight.get(index)
                if future is None:
                    future = self._executor.submit(
                        _load_frame_worker,
                        (
                            index,
                            self.paths[index],
                            self.resize_mode,
                            self.height,
                            self.width,
                        ),
                    )
                    self._inflight[index] = future
                futures[index] = future
        return _PrefetchedFrames(self, requested, futures)

    def _resolve_prefetch(
        self, indices: Sequence[int], futures: Dict[int, Future]
    ) -> list[torch.Tensor]:
        loaded = []
        for index in indices:
            value = self._cache_get(index)
            if value is None:
                future = futures.get(index)
                if future is None:
                    value = self[index]
                else:
                    try:
                        returned_index, array = future.result()
                    finally:
                        if self._inflight.get(index) is future:
                            self._inflight.pop(index, None)
                    if returned_index != index:
                        raise RuntimeError(
                            "RGB worker returned frame {} for requested {}".format(
                                returned_index, index
                            )
                        )
                    value = torch.from_numpy(array)
                self._cache_put(index, value)
            loaded.append(value)
        return loaded

    def load_indices(self, indices: Sequence[int]) -> list[torch.Tensor]:
        return self.prefetch_indices(indices).result()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for future in self._inflight.values():
            future.cancel()
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                self._executor.shutdown(wait=True)
            self._executor = None
        self._inflight.clear()
        self._cache.clear()

    def __enter__(self) -> "SequenceFrames":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def sequence_frames(sequence, dataset_config, *, raw_rgb=False, inference_config=None):
    inference_config = inference_config or {}
    precomputed = sequence.get("preprocessing_identity", "legacy_scared") != "legacy_scared"
    mode = "precomputed" if precomputed and not raw_rgb else dataset_config.get("resize_mode", "resize")
    return SequenceFrames(sequence["frame_paths"], resize_mode=mode,
                          height=int(inference_config.get("image_height", dataset_config.get("image_height", 448))),
                          width=int(inference_config.get("image_width", dataset_config.get("image_width", 560))))


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


def _window_frame_ids(frame_count: int, max_windows=None) -> list[list[int]]:
    starts = list(range(0, frame_count, STEP))
    if max_windows is not None:
        starts = starts[:max_windows]
    windows = []
    previous_ids = None
    for start in starts:
        ids = [min(start + offset, frame_count - 1) for offset in range(WINDOW)]
        if previous_ids is not None:
            ids[:OVERLAP] = [previous_ids[index] for index in KEYFRAMES]
        windows.append(ids)
        previous_ids = ids
    return windows


@dataclass
class InferenceStats:
    output_frame_count: int = 0
    window_count: int = 0
    model_input_frame_count: int = 0
    model_forward_seconds: float = 0.0
    sequence_pipeline_seconds: float = 0.0
    rgb_wait_seconds: float = 0.0
    alignment_fallback_count: int = 0

    def as_dict(self):
        n = self.output_frame_count
        windows = self.window_count
        seconds = self.model_forward_seconds
        return {**vars(self), "window_length": WINDOW, "window_stride": STEP,
                "overlap": OVERLAP, "blend_frames": BLEND,
                "mean_frame_inference_seconds": seconds / n if n else None,
                "mean_frame_inference_ms": seconds * 1000 / n if n else None,
                "inference_fps": n / seconds if seconds else None,
                "mean_frame_pipeline_seconds": self.sequence_pipeline_seconds / n if n else None,
                "mean_rgb_wait_ms_per_window": self.rgb_wait_seconds * 1000 / windows if windows else None,
                "timing_scope": "synchronized model forwards / unique output frames; includes repeated anchors and padding; excludes RGB decode, transfers, stitching, GT scoring and export",
                "pipeline_timing_scope": "RGB decode, transfer, model, stitching and output callback; excludes model loading and GT scoring",
                "warmup_excluded": False}


@torch.inference_mode()
def infer_vda_video(model, frames, emit: Callable, *, device, amp=True,
                    max_windows=None, emit_window=None,
                    forward_model: Optional[Callable] = None,
                    inspect_window: Optional[Callable] = None,
                    prediction_label: str = "model") -> dict:
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
    windows = _window_frame_ids(len(frames), max_windows=max_windows)
    stats = InferenceStats()
    started = time.perf_counter()
    anchors = None
    pending_disp = pending_k = None
    next_output = 0
    prefetched = None
    if hasattr(frames, "prefetch_indices"):
        first_unique_ids = list(dict.fromkeys(windows[0]))
        prefetched = frames.prefetch_indices(first_unique_ids)
    for window_number, ids in enumerate(windows):
        unique_ids = list(dict.fromkeys(ids))
        wait_started = time.perf_counter()
        if prefetched is not None:
            loaded = prefetched.result()
            if len(loaded) != len(unique_ids):
                raise RuntimeError("Frame loader returned the wrong number of images")
            decoded = dict(zip(unique_ids, loaded))
        elif hasattr(frames, "load_indices"):
            loaded = frames.load_indices(unique_ids)
            if len(loaded) != len(unique_ids):
                raise RuntimeError("Frame loader returned the wrong number of images")
            decoded = dict(zip(unique_ids, loaded))
        else:
            decoded = {index: frames[index] for index in unique_ids}
        stats.rgb_wait_seconds += time.perf_counter() - wait_started

        next_prefetched = None
        if window_number + 1 < len(windows) and hasattr(frames, "prefetch_indices"):
            next_unique_ids = list(dict.fromkeys(windows[window_number + 1]))
            next_prefetched = frames.prefetch_indices(next_unique_ids)

        images = torch.stack([decoded[index] for index in ids]).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
            prediction = (
                model(images, include_global_points=False)
                if forward_model is None
                else forward_model(model, images)
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stats.model_forward_seconds += time.perf_counter() - tick
        stats.window_count += 1
        stats.model_input_frame_count += WINDOW
        depth = prediction["depth"][0].float().cpu().numpy()
        if depth.shape != (WINDOW, *images.shape[-2:]) or not np.isfinite(depth).all():
            raise FloatingPointError("Invalid {} sequence depth output".format(prediction_label))
        if inspect_window is not None:
            inspect_window(window_number, ids, depth)
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
        final_window = window_number == len(windows) - 1
        count = len(combined_disp) if final_window else len(combined_disp) - BLEND
        count = min(count, len(frames) - next_output)
        if count > 0:
            emit(next_output, combined_disp[:count], combined_k[:count])
            next_output += count
        pending_disp = combined_disp[-BLEND:].copy()
        pending_k = combined_k[-BLEND:].copy()
        prefetched = next_prefetched
        if next_output == len(frames):
            break
    stats.output_frame_count = next_output
    stats.sequence_pipeline_seconds = time.perf_counter() - started
    result = stats.as_dict()
    result.update(
        {
            "rgb_loader_workers": int(getattr(frames, "num_workers", 1)),
            "rgb_loader_backend": str(
                getattr(frames, "loader_backend", "synchronous caller")
            ),
            "prefetch_windows": int(
                hasattr(frames, "prefetch_indices")
                and int(getattr(frames, "num_workers", 1)) > 1
            ),
            "frame_cache_size": int(getattr(frames, "frame_cache_size", 0)),
        }
    )
    return result


def infer_student_video(model, frames, emit: Callable, *, device, amp=True,
                        max_windows=None, emit_window=None) -> dict:
    """Run DA3 through the shared formal VDA temporal/stitching pipeline."""
    global _resolution_audit_printed
    frame_shape = (getattr(frames, "height", None), getattr(frames, "width", None))
    if not _resolution_audit_printed and frame_shape in {(448, 560), (224, 280)}:
        patch_grid = (frame_shape[0] // 14, frame_shape[1] // 14)
        print(
            "DA3 inference audit: model_input = {}x{}; patch_size = 14; "
            "patch_grid = {}x{}; patches_per_frame = {}; window_length = 32".format(
                *frame_shape, *patch_grid, patch_grid[0] * patch_grid[1]
            )
        )
        _resolution_audit_printed = True
    return infer_vda_video(
        model,
        frames,
        emit,
        device=device,
        amp=amp,
        max_windows=max_windows,
        emit_window=emit_window,
        prediction_label="DA3",
    )
