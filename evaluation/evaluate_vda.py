"""Run the unchanged Video-Depth-Anything depth score on project outputs.

The scoring core is copied from
DepthAnything/Video-Depth-Anything benchmark/eval/eval.py and metric.py.
Only the surrounding SCARED discovery, student/teacher-cache output conversion,
and reporting are project adapters.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

from datasets.scared_clip_dataset import (
    clip_metadata,
    make_scared_rgb_dataset,
    teacher_cache_path,
)
from datasets.scared_dataset import build_scared_dataloader
from evaluation.evaluate_depth import (
    ENDO3R_GT_DIRECTORY,
    ENDO3R_MAX_DEPTH,
    _find_gt_depths,
    _keyframe_directory,
    _load_endo3r_gt_depth,
    extract_frame_id,
)
from models.student.distill3r_wrapper import Distill3RStudent
from models.student.output_adapter import adapt_student_outputs
from utils.config import ensure_dir, load_config


VDA_METRIC_NAMES = (
    "abs_relative_difference",
    "rmse_linear",
    "delta1_acc",
)


def _select_vda_evaluation_config(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Prefer an explicit VDA section while preserving legacy configs."""

    has_vda_config = "vda_evaluation" in config
    selected = config.get("vda_evaluation", config.get("evaluation", {}))
    return dict(selected), has_vda_config


# ---------------------------------------------------------------------------
# BEGIN UNMODIFIED VIDEO-DEPTH-ANYTHING EVALUATION CORE
# Source:
# https://github.com/DepthAnything/Video-Depth-Anything/tree/main/benchmark/eval
# Do not place project dataset/model adaptations inside this section.
# ---------------------------------------------------------------------------


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def evaluate_depth_core(
    infs: np.ndarray,
    gts: np.ndarray,
    max_eval_len: int,
    max_depth_eval: float,
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    seq_length = max_eval_len
    dataset_max_depth = max_depth_eval
    infs = infs[:seq_length]
    gts = gts[:seq_length]
    valid_mask = np.logical_and((gts > 1e-3), (gts < dataset_max_depth))

    gt_disp_masked = 1.0 / (
        gts[valid_mask].reshape((-1, 1)).astype(np.float64) + 1e-8
    )
    infs = np.clip(infs, a_min=1e-3, a_max=None)
    pred_disp_masked = infs[valid_mask].reshape((-1, 1)).astype(np.float64)

    _ones = np.ones_like(pred_disp_masked)
    A = np.concatenate([pred_disp_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_disp_masked, rcond=None)[0]
    scale, shift = X
    aligned_pred = scale * infs + shift
    aligned_pred = np.clip(aligned_pred, a_min=1e-3, a_max=None)

    pred_depth = depth2disparity(aligned_pred)
    gt_depth = gts
    pred_depth = np.clip(
        pred_depth, a_min=1e-3, a_max=dataset_max_depth
    )
    sample_metric = []
    metric_funcs = [
        abs_relative_difference,
        rmse_linear,
        delta1_acc,
    ]

    pred_depth_ts = torch.from_numpy(pred_depth).to("cpu")
    gt_depth_ts = torch.from_numpy(gt_depth).to("cpu")
    valid_mask_ts = torch.from_numpy(valid_mask).to("cpu")

    n = valid_mask.sum((-1, -2))
    valid_frame = n > 0
    pred_depth_ts = pred_depth_ts[valid_frame]
    gt_depth_ts = gt_depth_ts[valid_frame]
    valid_mask_ts = valid_mask_ts[valid_frame]

    for met_func in metric_funcs:
        _metric = met_func(
            pred_depth_ts, gt_depth_ts, valid_mask_ts
        ).item()
        sample_metric.append(_metric)
    return sample_metric, scale, shift


# ---------------------------------------------------------------------------
# END UNMODIFIED VIDEO-DEPTH-ANYTHING EVALUATION CORE
# ---------------------------------------------------------------------------


def _opencv() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise RuntimeError(
            "Video-Depth-Anything evaluation requires opencv-python"
        ) from error


def _student_depth_to_vda_disparity(depth: np.ndarray) -> np.ndarray:
    """Project adapter: DUNE local point Z is depth; VDA expects disparity."""
    depth = np.asarray(depth, dtype=np.float32)
    if not np.all(np.isfinite(depth)):
        raise ValueError("Student depth contains NaN or Inf")
    return np.reciprocal(np.clip(depth, 1e-3, None))


def _teacher_cache_split_root(path: Path, split: str) -> Path:
    path = path.expanduser().resolve()
    if path.name.lower() == split.lower():
        return path
    split_path = path / split
    return split_path if split_path.is_dir() else path


def _load_teacher_cache_clip(
    path: Path,
    metadata: Dict[str, Any],
) -> Tuple[List[int], np.ndarray, str]:
    """Load only the cached teacher fields required by VDA scoring."""
    required = ("xyz_local", "frame_names", "frame_indices")
    try:
        with np.load(str(path), allow_pickle=False) as cache:
            missing = [name for name in required if name not in cache]
            if missing:
                raise RuntimeError(
                    "Teacher cache {} is missing {}".format(path, missing)
                )
            points = np.asarray(cache["xyz_local"], dtype=np.float32)
            frame_names = [
                str(name) for name in np.asarray(cache["frame_names"]).tolist()
            ]
            frame_indices = [
                int(index)
                for index in np.asarray(cache["frame_indices"]).tolist()
            ]
            variant = (
                str(np.asarray(cache["teacher_variant"]).item())
                if "teacher_variant" in cache
                else "unknown"
            )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "Failed to read teacher cache {}: {}".format(path, error)
        ) from error
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(
            "xyz_local in {} must have shape [T,H,W,3], got {}".format(
                path, points.shape
            )
        )
    if frame_names != list(metadata["frame_names"]):
        raise RuntimeError("Teacher cache/RGB frame mismatch at {}".format(path))
    if frame_indices != list(metadata["frame_indices"]):
        raise RuntimeError(
            "Teacher cache/source index mismatch at {}".format(path)
        )
    if points.shape[0] != len(frame_indices):
        raise RuntimeError(
            "Teacher cache frame count mismatch at {}".format(path)
        )
    return (
        frame_indices,
        _student_depth_to_vda_disparity(points[..., 2]),
        variant,
    )


def _load_student_memory_efficient(
    checkpoint_path: Path,
    fallback_config: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Load model weights without retaining optimizer/training tensors."""
    size_gib = checkpoint_path.stat().st_size / (1024 ** 3)
    print(
        "[VDA] loading checkpoint={} size={:.2f} GiB".format(
            checkpoint_path, size_gib
        ),
        flush=True,
    )
    try:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        print("[VDA] checkpoint opened with mmap", flush=True)
    except (TypeError, RuntimeError) as error:
        if "mmap" not in str(error).lower() and not isinstance(error, TypeError):
            raise
        print(
            "[VDA] mmap unavailable; falling back to regular torch.load",
            flush=True,
        )
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )

    checkpoint_config = (
        checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    )
    student_config = dict(
        checkpoint_config.get("student", fallback_config["student"])
    )
    state = (
        checkpoint.get("model", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if state is not checkpoint:
        # Optimizer, scheduler, scaler, and other training state are not used
        # for evaluation. Release them before allocating a second model copy.
        checkpoint.clear()
    checkpoint = None
    checkpoint_config = None
    gc.collect()
    print(
        "[VDA] training state released; constructing student model",
        flush=True,
    )

    model = Distill3RStudent(student_config)
    try:
        model.load_state_dict(state, strict=True, assign=True)
        assigned = True
    except TypeError:
        model.load_state_dict(state, strict=True)
        assigned = False
    state = None
    gc.collect()
    print(
        "[VDA] weights loaded (assign={}); moving model to {}".format(
            assigned, device
        ),
        flush=True,
    )
    model = model.eval().to(device)
    gc.collect()
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        print(
            "[VDA] model ready; CUDA allocated={:.2f} GiB reserved={:.2f} GiB".format(
                allocated, reserved
            ),
            flush=True,
        )
    else:
        print("[VDA] model ready on CPU", flush=True)
    return model


class _SequencePredictionSpool:
    """Disk-backed overlapping prediction sums for one complete sequence."""

    def __init__(
        self,
        directory: Path,
        frame_count: int,
        height: int,
        width: int,
    ) -> None:
        handle = tempfile.NamedTemporaryFile(
            prefix=".vda_spool_",
            suffix=".dat",
            dir=str(directory),
            delete=False,
        )
        self.path = Path(handle.name)
        handle.close()
        self.frame_count = frame_count
        self.height = height
        self.width = width
        self.sums = np.memmap(
            self.path,
            mode="w+",
            dtype=np.float32,
            shape=(frame_count, height, width),
        )
        self.counts = np.zeros(frame_count, dtype=np.uint16)

    @property
    def size_gib(self) -> float:
        return self.path.stat().st_size / (1024 ** 3)

    def add(
        self,
        frame_indices: Sequence[int],
        disparities: np.ndarray,
    ) -> None:
        cv2 = _opencv()
        for offset, frame_index in enumerate(frame_indices):
            resized = cv2.resize(
                disparities[offset],
                (self.width, self.height),
            ).astype(np.float32, copy=False)
            self.sums[frame_index] += resized
            self.counts[frame_index] += 1

    def prediction(self, frame_index: int) -> np.ndarray:
        count = int(self.counts[frame_index])
        if count <= 0:
            raise RuntimeError(
                "No prediction was accumulated for source frame {}".format(
                    frame_index
                )
            )
        return np.asarray(
            self.sums[frame_index] / float(count), dtype=np.float32
        )

    def flush(self) -> None:
        self.sums.flush()

    def close(self) -> None:
        self.sums.flush()
        mmap_object = getattr(self.sums, "_mmap", None)
        if mmap_object is not None:
            mmap_object.close()
        self.sums = None
        gc.collect()
        self.path.unlink(missing_ok=True)


def _resized_gt(
    path: Path,
    gt_channel: int,
    height: int,
    width: int,
) -> np.ndarray:
    cv2 = _opencv()
    return cv2.resize(
        _load_endo3r_gt_depth(path, gt_channel),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def _frame_pairs(
    sequence: Dict[str, Any],
    spool: _SequencePredictionSpool,
    gt_by_id: Dict[int, Path],
    require_all_gt: bool,
) -> Tuple[List[Tuple[int, int, Path]], List[int]]:
    rgb_index_by_id: Dict[int, int] = {}
    for index, frame_path in enumerate(sequence["frame_paths"]):
        frame_id = extract_frame_id(frame_path)
        if frame_id in rgb_index_by_id:
            raise RuntimeError(
                "Duplicate RGB frame ID {} in {}".format(
                    frame_id, sequence["sequence_id"]
                )
            )
        rgb_index_by_id[frame_id] = index
    predicted_ids = {
        frame_id
        for frame_id, index in rgb_index_by_id.items()
        if spool.counts[index] > 0
    }
    gt_ids = set(gt_by_id)
    matched_ids = sorted(predicted_ids & gt_ids)
    missing_prediction_ids = sorted(gt_ids - predicted_ids)
    if require_all_gt and missing_prediction_ids:
        raise RuntimeError(
            "VDA scoring requires every GT frame in {}. Missing IDs: {}".format(
                sequence["sequence_id"], missing_prediction_ids[:20]
            )
        )
    if not matched_ids:
        raise RuntimeError(
            "No prediction/GT frame IDs match for {}".format(
                sequence["sequence_id"]
            )
        )
    return [
        (frame_id, rgb_index_by_id[frame_id], gt_by_id[frame_id])
        for frame_id in matched_ids
    ], missing_prediction_ids


def _streaming_scale_shift(
    pairs: Sequence[Tuple[int, int, Path]],
    spool: _SequencePredictionSpool,
    gt_channel: int,
    sequence_id: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Reduce the same global least-squares system with bounded memory."""
    reduced_augmented = None
    valid_pixel_count = 0
    for position, (_, frame_index, gt_path) in enumerate(pairs, start=1):
        gt_depth = _resized_gt(
            gt_path, gt_channel, spool.height, spool.width
        )
        prediction = np.clip(
            spool.prediction(frame_index), a_min=1e-3, a_max=None
        )
        valid_mask = np.logical_and(
            gt_depth > 1e-3, gt_depth < ENDO3R_MAX_DEPTH
        )
        if not np.any(valid_mask):
            continue
        target = 1.0 / (
            gt_depth[valid_mask].reshape((-1, 1)).astype(np.float64)
            + 1e-8
        )
        predicted = prediction[valid_mask].reshape((-1, 1)).astype(
            np.float64
        )
        augmented = np.concatenate(
            [predicted, np.ones_like(predicted), target], axis=-1
        )
        _, frame_reduction = np.linalg.qr(augmented, mode="reduced")
        if reduced_augmented is None:
            reduced_augmented = frame_reduction
        else:
            _, reduced_augmented = np.linalg.qr(
                np.concatenate(
                    [reduced_augmented, frame_reduction], axis=0
                ),
                mode="reduced",
            )
        valid_pixel_count += int(valid_mask.sum())
        if position == 1 or position % 100 == 0:
            print(
                "[VDA] sequence={} alignment_pass={}/{}".format(
                    sequence_id, position, len(pairs)
                ),
                flush=True,
            )
    if reduced_augmented is None:
        raise RuntimeError(
            "No valid GT pixels remain for {}".format(sequence_id)
        )
    X = np.linalg.lstsq(
        reduced_augmented[:, :2],
        reduced_augmented[:, 2:3],
        rcond=None,
    )[0]
    scale, shift = X
    return scale, shift, valid_pixel_count


def _streaming_metrics(
    pairs: Sequence[Tuple[int, int, Path]],
    spool: _SequencePredictionSpool,
    gt_channel: int,
    scale: np.ndarray,
    shift: np.ndarray,
    sequence_id: str,
) -> Tuple[List[float], int]:
    metric_sums = np.zeros(len(VDA_METRIC_NAMES), dtype=np.float64)
    valid_frame_count = 0
    metric_funcs = [abs_relative_difference, rmse_linear, delta1_acc]
    for position, (_, frame_index, gt_path) in enumerate(pairs, start=1):
        gt_depth = _resized_gt(
            gt_path, gt_channel, spool.height, spool.width
        )
        valid_mask = np.logical_and(
            gt_depth > 1e-3, gt_depth < ENDO3R_MAX_DEPTH
        )
        if not np.any(valid_mask):
            continue
        prediction = np.clip(
            spool.prediction(frame_index), a_min=1e-3, a_max=None
        )
        aligned_prediction = np.clip(
            scale * prediction + shift, a_min=1e-3, a_max=None
        )
        pred_depth = depth2disparity(aligned_prediction)
        pred_depth = np.clip(
            pred_depth, a_min=1e-3, a_max=ENDO3R_MAX_DEPTH
        )
        pred_depth_ts = torch.from_numpy(pred_depth[None]).to("cpu")
        gt_depth_ts = torch.from_numpy(gt_depth[None]).to("cpu")
        valid_mask_ts = torch.from_numpy(valid_mask[None]).to("cpu")
        for metric_index, metric_func in enumerate(metric_funcs):
            metric_sums[metric_index] += metric_func(
                pred_depth_ts, gt_depth_ts, valid_mask_ts
            ).item()
        valid_frame_count += 1
        if position == 1 or position % 100 == 0:
            print(
                "[VDA] sequence={} metric_pass={}/{}".format(
                    sequence_id, position, len(pairs)
                ),
                flush=True,
            )
    if valid_frame_count == 0:
        raise RuntimeError(
            "No valid frames remain for {}".format(sequence_id)
        )
    return (metric_sums / valid_frame_count).tolist(), valid_frame_count


def _evaluate_sequence(
    sequence: Dict[str, Any],
    spool: _SequencePredictionSpool,
    gt_channel: int,
    gt_depths: Tuple[Path, Dict[int, Path]],
    require_all_gt: bool,
) -> Dict[str, Any]:
    gt_directory, gt_by_id = gt_depths
    pairs, missing_prediction_ids = _frame_pairs(
        sequence, spool, gt_by_id, require_all_gt
    )
    scale, shift, valid_pixel_count = _streaming_scale_shift(
        pairs, spool, gt_channel, str(sequence["sequence_id"])
    )
    metrics, valid_frame_count = _streaming_metrics(
        pairs,
        spool,
        gt_channel,
        scale,
        shift,
        str(sequence["sequence_id"]),
    )
    return {
        "sequence_id": str(sequence["sequence_id"]),
        "gt_directory": str(gt_directory),
        "matched_frame_count": len(pairs),
        "evaluated_frame_count": valid_frame_count,
        "valid_pixel_count": valid_pixel_count,
        "evaluation_size": [spool.width, spool.height],
        "evaluation_shape_hxw": [spool.height, spool.width],
        "missing_prediction_count": len(missing_prediction_ids),
        "missing_prediction_ids_preview": missing_prediction_ids[:20],
        "disparity_scale": float(np.asarray(scale).reshape(-1)[0]),
        "disparity_shift": float(np.asarray(shift).reshape(-1)[0]),
        "metrics": {
            name: float(value)
            for name, value in zip(VDA_METRIC_NAMES, metrics)
        },
    }


def _find_sequence_gt_depths(
    sequence: Dict[str, Any],
    eval_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
) -> Tuple[Path, Dict[int, Path]]:
    """Resolve GT from the explicit evaluation path, then dataset fallbacks."""
    keyframe_directory = _keyframe_directory(sequence)
    ground_truth = dict(dataset_config.get("ground_truth", {}))
    candidates: List[Path] = []
    explicit = eval_config.get("gt_relative_directory")
    if explicit:
        candidates.append(keyframe_directory / str(explicit))
    for key in ground_truth.get(
        "directory_keys", ("depth_directory", "scene_points_directory")
    ):
        value = sequence.get(str(key))
        if value:
            candidates.append(Path(str(value)))
    for relative in ground_truth.get("relative_directories", ()):
        candidates.append(keyframe_directory / str(relative))
    if not candidates:
        candidates.append(keyframe_directory / ENDO3R_GT_DIRECTORY)

    checked: List[str] = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        checked.append(str(candidate))
        try:
            return _find_gt_depths(
                keyframe_directory,
                relative_directory=str(candidate),
            )
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "No supported depth GT found for {}. Checked: {}".format(
            sequence.get("sequence_id"), checked
        )
    )


def evaluate(
    config_path: Path,
    checkpoint_override: Optional[Path],
    split_override: Optional[str],
    output_override: Optional[Path],
    limit_clips: Optional[int],
    teacher_cache_override: Optional[Path] = None,
) -> Dict[str, Any]:
    """Project adapter around the unchanged VDA evaluation core."""
    config = load_config(config_path)
    # A dedicated section lets an experiment select VDA as its primary test
    # protocol while retaining the existing Endo3R ``evaluation`` section.
    eval_config, has_vda_config = _select_vda_evaluation_config(config)
    split = split_override or str(eval_config.get("split", "test"))
    teacher_cache_root = (
        _teacher_cache_split_root(teacher_cache_override, split)
        if teacher_cache_override is not None
        else None
    )
    checkpoint_path: Optional[Path] = None
    if teacher_cache_root is None:
        checkpoint_path = checkpoint_override or Path(
            eval_config.get(
                "checkpoint",
                Path(config["training"]["output_dir"]) / "last.pt",
            )
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Student checkpoint not found: {}".format(checkpoint_path)
            )
        default_output = (
            checkpoint_path.parent
            / "evaluation_{}_vda.json".format(split)
        )
    else:
        if checkpoint_override is not None:
            raise ValueError(
                "--checkpoint and --teacher-cache are mutually exclusive"
            )
        if not teacher_cache_root.is_dir():
            raise FileNotFoundError(
                "Teacher cache root not found: {}".format(
                    teacher_cache_root
                )
            )
        default_output = teacher_cache_root / "evaluation_vda.json"
    configured_output = (
        Path(eval_config["output"])
        if has_vda_config
        and teacher_cache_root is None
        and eval_config.get("output")
        else None
    )
    output_path = output_override or configured_output or default_output
    ensure_dir(output_path.parent)
    device = torch.device(
        "cpu"
        if teacher_cache_root is not None
        else str(config.get("device", "cuda"))
    )
    if (
        teacher_cache_root is None
        and device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    print(
        "[VDA] config={} split={} source={} device={}".format(
            config_path,
            split,
            "teacher_cache" if teacher_cache_root is not None else "student",
            device,
        ),
        flush=True,
    )

    dataset_config = dict(config["dataset"])
    dataset_config["frame_source"] = str(
        eval_config.get(
            "frame_source",
            dataset_config.get("frame_source", "auto"),
        )
    )
    if teacher_cache_root is None:
        dataset_config["drop_incomplete_clip"] = False
    evaluation_height = int(dataset_config["image_height"])
    evaluation_width = int(dataset_config["image_width"])
    dataset = make_scared_rgb_dataset(dataset_config, split)
    gt_channel = int(eval_config.get("gt_depth_channel", 0))
    require_all_gt = bool(eval_config.get("require_all_gt", True))
    sequence_by_id = {
        str(sequence["sequence_id"]): sequence
        for sequence in dataset.sequences
    }

    evaluable_sequence_ids = set()
    gt_depths_by_sequence: Dict[
        str, Tuple[Path, Dict[int, Path]]
    ] = {}
    skipped_sequences = []
    for sequence_id, sequence in sequence_by_id.items():
        try:
            gt_depths = _find_sequence_gt_depths(
                sequence, eval_config, dataset_config
            )
        except FileNotFoundError as error:
            skipped_sequences.append(
                {"sequence_id": sequence_id, "reason": str(error)}
            )
            continue
        gt_depths_by_sequence[sequence_id] = gt_depths
        evaluable_sequence_ids.add(sequence_id)
        print(
            "[VDA] GT sequence={} directory={} frames={}".format(
                sequence_id, gt_depths[0], len(gt_depths[1])
            ),
            flush=True,
        )
    clip_indices = [
        index
        for index, clip in enumerate(dataset.clips)
        if str(clip.sequence["sequence_id"]) in evaluable_sequence_ids
    ]
    if not clip_indices:
        raise RuntimeError(
            "No evaluable clips have configured depth GT. Details: {}".format(
                skipped_sequences
            )
        )
    cache_paths: Dict[int, Path] = {}
    if teacher_cache_root is not None:
        for index in clip_indices:
            metadata = clip_metadata(dataset, index)
            cache_paths[index] = teacher_cache_path(
                teacher_cache_root, metadata
            )
        missing_cache_paths = [
            path for path in cache_paths.values() if not path.is_file()
        ]
        if missing_cache_paths:
            raise FileNotFoundError(
                "Teacher cache is incomplete for the configured {} split: "
                "missing {}/{} clips. First missing paths: {}".format(
                    split,
                    len(missing_cache_paths),
                    len(cache_paths),
                    [str(path) for path in missing_cache_paths[:10]],
                )
            )
    print(
        "[VDA] preflight sequences={} evaluable_sequences={} clips={} "
        "skipped_without_gt={} evaluation_shape_hxw={}x{}".format(
            len(sequence_by_id),
            len(evaluable_sequence_ids),
            len(clip_indices),
            len(skipped_sequences),
            evaluation_height,
            evaluation_width,
        ),
        flush=True,
    )

    loader = None
    model = None
    amp_enabled = False
    if teacher_cache_root is None:
        # Forking workers after a large checkpoint is loaded can multiply the
        # container's apparent host memory and trigger the Linux OOM killer.
        loader = build_scared_dataloader(
            Subset(dataset, clip_indices),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=bool(eval_config.get("pin_memory", True)),
            persistent_workers=False,
            prefetch_factor=int(
                config["dataloader"].get("prefetch_factor", 2)
            ),
            drop_last=False,
            seed=int(config.get("seed", 42)),
        )
        print("[VDA] DataLoader ready with num_workers=0", flush=True)
        model = _load_student_memory_efficient(
            checkpoint_path, config, device
        )
        amp_enabled = (
            bool(eval_config.get("amp", True)) and device.type == "cuda"
        )
    else:
        print(
            "[VDA] teacher cache ready root={}".format(teacher_cache_root),
            flush=True,
        )
    log_every = max(int(config["training"].get("log_every", 10)), 1)

    current_sequence_id: Optional[str] = None
    current_spool: Optional[_SequencePredictionSpool] = None
    sequence_results: List[Dict[str, Any]] = []
    inference_times = []
    teacher_variants = set()
    processed_clips = 0

    def start_sequence(sequence_id: str, frame_count: int) -> None:
        nonlocal current_spool
        current_spool = _SequencePredictionSpool(
            output_path.parent,
            frame_count,
            evaluation_height,
            evaluation_width,
        )
        print(
            "[VDA] sequence={} prediction spool={} size={:.2f}GiB".format(
                sequence_id, current_spool.path, current_spool.size_gib
            ),
            flush=True,
        )

    def finish_sequence() -> None:
        nonlocal current_spool
        if current_sequence_id is None or current_spool is None:
            return
        current_spool.flush()
        try:
            sequence_result = _evaluate_sequence(
                sequence_by_id[current_sequence_id],
                current_spool,
                gt_channel,
                gt_depths_by_sequence[current_sequence_id],
                require_all_gt=(
                    require_all_gt
                    and limit_clips is None
                    and teacher_cache_root is None
                ),
            )
            sequence_results.append(sequence_result)
            print(
                "[VDA] sequence={} frames={} metrics={}".format(
                    current_sequence_id,
                    sequence_result["matched_frame_count"],
                    sequence_result["metrics"],
                ),
                flush=True,
            )
        finally:
            current_spool.close()
            current_spool = None
            gc.collect()

    def accept_clip(
        sequence_id: str,
        sequence_length: int,
        frame_indices: Sequence[int],
        disparities: np.ndarray,
        source_seconds: float,
    ) -> None:
        nonlocal current_sequence_id, processed_clips
        if current_sequence_id is None:
            current_sequence_id = sequence_id
            start_sequence(sequence_id, sequence_length)
        elif sequence_id != current_sequence_id:
            finish_sequence()
            current_sequence_id = sequence_id
            start_sequence(sequence_id, sequence_length)
        if current_spool is None:
            raise RuntimeError("Prediction spool was not initialized")
        current_spool.add(frame_indices, disparities)
        inference_times.append(source_seconds)
        processed_clips += 1
        if processed_clips == 1 or processed_clips % log_every == 0:
            current_spool.flush()
            time_label = (
                "cache_load"
                if teacher_cache_root is not None
                else "inference"
            )
            message = (
                "[VDA] clip={}/{} sequence={} {}={:.3f}s "
                "spool={:.2f}GiB".format(
                    processed_clips,
                    len(clip_indices),
                    sequence_id,
                    time_label,
                    source_seconds,
                    current_spool.size_gib,
                )
            )
            if teacher_cache_root is None and device.type == "cuda":
                message += " cuda_allocated={:.2f}GiB".format(
                    torch.cuda.memory_allocated(device) / (1024 ** 3)
                )
            print(message, flush=True)

    print(
        "[VDA] starting {} amp={} log_every={} clips".format(
            "cache evaluation"
            if teacher_cache_root is not None
            else "inference",
            amp_enabled,
            log_every,
        ),
        flush=True,
    )
    try:
        if teacher_cache_root is not None:
            for index in clip_indices:
                metadata = clip_metadata(dataset, index)
                started = time.perf_counter()
                frame_indices, disparities, variant = (
                    _load_teacher_cache_clip(
                        cache_paths[index], metadata
                    )
                )
                teacher_variants.add(variant)
                accept_clip(
                    str(metadata["sequence_id"]),
                    int(metadata["sequence_length"]),
                    frame_indices,
                    disparities,
                    time.perf_counter() - started,
                )
                if (
                    limit_clips is not None
                    and processed_clips >= limit_clips
                ):
                    break
        else:
            if loader is None or model is None:
                raise RuntimeError("Student evaluation was not initialized")
            with torch.inference_mode():
                for batch in loader:
                    images = batch["images"].to(
                        device, non_blocking=True
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        prediction = adapt_student_outputs(model(images))
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    depth = prediction["depth"][0].float().cpu().numpy()
                    disparities = _student_depth_to_vda_disparity(depth)
                    accept_clip(
                        str(batch["sequence_id"][0]),
                        int(batch["sequence_length"][0]),
                        [
                            int(index)
                            for index in batch["frame_indices"][0].tolist()
                        ],
                        disparities,
                        time.perf_counter() - started,
                    )
                    if (
                        limit_clips is not None
                        and processed_clips >= limit_clips
                    ):
                        break
        finish_sequence()
    finally:
        if current_spool is not None:
            current_spool.close()
            current_spool = None
    if not sequence_results:
        raise RuntimeError("No sequence was evaluated")

    final_metrics = {
        name: float(
            np.mean(
                [result["metrics"][name] for result in sequence_results]
            )
        )
        for name in VDA_METRIC_NAMES
    }
    all_source_clips_processed = (
        limit_clips is None and processed_clips == len(clip_indices)
    )
    complete_gt_coverage = all(
        item["missing_prediction_count"] == 0
        for item in sequence_results
    )
    if not complete_gt_coverage:
        print(
            "[VDA] warning: all source clips were evaluated, but some GT "
            "frames are not covered by the generated cache clips",
            flush=True,
        )
    result = {
        "protocol": "video-depth-anything-depth",
        "source": (
            "https://github.com/DepthAnything/Video-Depth-Anything/"
            "tree/main/benchmark/eval"
        ),
        "config": str(config_path),
        "input_type": (
            "teacher_cache"
            if teacher_cache_root is not None
            else "student_checkpoint"
        ),
        "checkpoint": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
        "teacher_cache_root": (
            str(teacher_cache_root)
            if teacher_cache_root is not None
            else None
        ),
        "teacher_variants": sorted(teacher_variants),
        "split": split,
        "metrics": final_metrics,
        "sequence_count": len(sequence_results),
        "processed_clip_count": processed_clips,
        "expected_clip_count": len(clip_indices),
        "all_source_clips_processed": all_source_clips_processed,
        "complete_gt_coverage": complete_gt_coverage,
        "full_test_set": (
            all_source_clips_processed and complete_gt_coverage
        ),
        "evaluation_size": [evaluation_width, evaluation_height],
        "evaluation_shape_hxw": [evaluation_height, evaluation_width],
        "mean_clip_inference_seconds": (
            float(np.mean(inference_times))
            if inference_times and teacher_cache_root is None
            else None
        ),
        "mean_clip_cache_load_seconds": (
            float(np.mean(inference_times))
            if inference_times and teacher_cache_root is not None
            else None
        ),
        "skipped_sequences_without_gt": skipped_sequences,
        "model_output_adapter": (
            "cached xyz_local[...,2] depth -> reciprocal disparity; "
            if teacher_cache_root is not None
            else "student xyz_local[...,2] depth -> reciprocal disparity; "
        )
        + (
            "overlapping clip predictions averaged by source frame index"
        ),
        "prediction_storage": "per-sequence disk memmap",
        "streaming_two_pass": True,
        "core_algorithm_modified": False,
        "sequences": sequence_results,
    }
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("wrote VDA evaluation: {}".format(output_path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/student_distillation.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=None,
        help=(
            "VGGT-Omega cache root, with or without the split suffix; "
            "mutually exclusive with --checkpoint"
        ),
    )
    parser.add_argument(
        "--split", choices=("train", "test"), default=None
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=None,
        help="Debug only",
    )
    args = parser.parse_args()
    evaluate(
        args.config,
        args.checkpoint,
        args.split,
        args.output,
        args.limit_clips,
        args.teacher_cache,
    )


if __name__ == "__main__":
    main()
