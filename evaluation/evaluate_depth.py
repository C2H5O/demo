"""Evaluate the distilled student with the official Endo3R SCARED protocol.

Algorithmic constants and depth scoring follow:
https://github.com/wrld/Endo3R/blob/main/eval/depth_evaluation.py

Only project interfaces are adapted: the fixed-length student is run on
overlapping clips, repeated predictions for one frame are averaged, and SCARED
prediction/GT files are aligned by their numeric frame IDs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

from datasets.scared_clip_dataset import make_scared_rgb_dataset
from datasets.scared_dataset import build_scared_dataloader
from evaluation.depth_metrics import METRIC_NAMES, compute_errors
from models.student.dune_model import DUNEViTSmallPointMapStudent
from utils.config import ensure_dir, load_config


# Official Endo3R SCARED depth-evaluation constants.
ENDO3R_WIDTH = 320
ENDO3R_HEIGHT = 256
ENDO3R_MIN_DEPTH = 0.0001
ENDO3R_MAX_DEPTH = 100.0
ENDO3R_GT_SCALE = 1.0 / 1000.0
ENDO3R_GT_DIRECTORY = "data/depthmap_rectified"
ENDO3R_FRAME_SOURCE = "left_rectified"
PROJECT_FRAME_SOURCE = "auto"

SUPPORTED_DEPTH_SUFFIXES = {".png", ".tif", ".tiff", ".npy"}
FRAME_ID_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def _opencv() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise RuntimeError(
            "Endo3R evaluation requires OpenCV. Install opencv-python."
        ) from error


def extract_frame_id(path: str | Path) -> int:
    match = FRAME_ID_PATTERN.search(Path(path).stem)
    if match is None:
        raise ValueError("Cannot extract a numeric frame ID from {}".format(path))
    return int(match.group(1))


def _metric_dict(values: np.ndarray) -> Dict[str, float]:
    return {
        name: float(value) for name, value in zip(METRIC_NAMES, values)
    }


def _distribution_stats(values: np.ndarray) -> Dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot summarize an empty/non-finite depth array")
    percentiles = np.percentile(finite, [0, 1, 5, 50, 95, 99, 100])
    return {
        "count": int(finite.size),
        "min": float(percentiles[0]),
        "p01": float(percentiles[1]),
        "p05": float(percentiles[2]),
        "median": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "max": float(percentiles[6]),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _build_unique_frame_map(
    paths: Iterable[Path],
    label: str,
) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    for path in paths:
        identifier = extract_frame_id(path)
        if identifier in result:
            raise RuntimeError(
                "Duplicate {} frame ID {}: {} and {}".format(
                    label, identifier, result[identifier], path
                )
            )
        result[identifier] = path
    return result


def _keyframe_directory(sequence: Dict[str, Any]) -> Path:
    value = sequence.get("keyframe_directory")
    if value:
        return Path(str(value))
    # Compatibility with older manifests that did not record this field.
    return Path(str(sequence["frame_directory"])).parent.parent


def _find_gt_depths(
    keyframe_directory: Path,
    relative_directory: str = ENDO3R_GT_DIRECTORY,
) -> Tuple[Path, Dict[int, Path]]:
    directory = keyframe_directory / relative_directory
    if not directory.is_dir():
        raise FileNotFoundError(
            "Endo3R GT directory does not exist: {}".format(directory)
        )
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_DEPTH_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(
            "Endo3R GT directory has no supported depth files: {}".format(
                directory
            )
        )
    return directory, _build_unique_frame_map(paths, "GT")


def _load_endo3r_gt_depth(path: Path, channel: int) -> np.ndarray:
    """Load SCARED GT and apply Endo3R's fixed millimetre-to-metre scale."""
    if path.suffix.lower() == ".npy":
        depth = np.load(str(path))
    else:
        cv2 = _opencv()
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError("Failed to read GT depth {}".format(path))
    depth = np.asarray(depth)
    if depth.ndim == 3:
        if not 0 <= channel < depth.shape[-1]:
            raise ValueError(
                "GT depth channel {} is invalid for {} with shape {}".format(
                    channel, path, depth.shape
                )
            )
        depth = depth[..., channel]
    if depth.ndim != 2:
        raise ValueError(
            "Expected a 2D GT depth map at {}, got {}".format(
                path, depth.shape
            )
        )
    return depth.astype(np.float32) * ENDO3R_GT_SCALE


def depth_evaluation(
    gt_depths: Sequence[np.ndarray],
    pred_depths: Sequence[np.ndarray],
    pred_masks: Optional[Sequence[np.ndarray]] = None,
    min_depth: float = ENDO3R_MIN_DEPTH,
    max_depth: float = ENDO3R_MAX_DEPTH,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Endo3R scene-level depth evaluation, kept algorithmically equivalent."""
    if len(gt_depths) != len(pred_depths):
        raise ValueError(
            "GT/prediction frame count mismatch: {} versus {}".format(
                len(gt_depths), len(pred_depths)
            )
        )
    if pred_masks is not None and len(pred_masks) != len(gt_depths):
        raise ValueError("pred_masks must have the same frame count as GT")

    cv2 = _opencv()
    gt_depths_valid: List[np.ndarray] = []
    pred_depths_valid: List[np.ndarray] = []
    for index, (gt_depth, pred_depth) in enumerate(
        zip(gt_depths, pred_depths)
    ):
        mask = (gt_depth > min_depth) * (gt_depth < max_depth)
        gt_height, gt_width = gt_depth.shape[:2]
        if pred_masks is not None:
            pred_mask = cv2.resize(
                pred_masks[index].astype(np.uint8),
                (gt_width, gt_height),
            ) > 0.5
            mask = mask * pred_mask
        if mask.sum() == 0:
            continue
        pred_depth_valid = pred_depth[mask].astype(np.float64)
        gt_depth_valid = gt_depth[mask].astype(np.float64)
        if not np.all(np.isfinite(pred_depth_valid)):
            raise ValueError(
                "Prediction contains NaN or Inf at aligned frame {}".format(
                    index
                )
            )
        pred_depths_valid.append(pred_depth_valid)
        gt_depths_valid.append(gt_depth_valid)

    if not gt_depths_valid:
        raise RuntimeError("No valid depth pixels remained for Endo3R evaluation")
    prediction_median = float(np.median(np.concatenate(pred_depths_valid)))
    if not np.isfinite(prediction_median) or prediction_median <= 0:
        raise RuntimeError(
            "Endo3R scene scaling requires a positive prediction median; "
            "received {}".format(prediction_median)
        )
    ratio = float(
        np.median(np.concatenate(gt_depths_valid)) / prediction_median
    )

    errors = []
    for gt_depth, pred_depth in zip(gt_depths_valid, pred_depths_valid):
        # Work on a copy because Endo3R scales and clips predictions in-place.
        aligned_prediction = pred_depth.copy()
        aligned_prediction *= ratio
        aligned_prediction[aligned_prediction < min_depth] = min_depth
        aligned_prediction[aligned_prediction > max_depth] = max_depth
        errors.append(compute_errors(gt_depth, aligned_prediction))
    error_array = np.asarray(errors, dtype=np.float64)
    return error_array, error_array.mean(0), ratio


def _load_student(
    checkpoint_path: Path,
    fallback_config: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    checkpoint_config = (
        checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    )
    student_config = dict(
        checkpoint_config.get("student", fallback_config["student"])
    )
    student_config["encoder_checkpoint"] = None
    model = DUNEViTSmallPointMapStudent(student_config)
    state = (
        checkpoint.get("model", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def _evaluate_sequence(
    sequence: Dict[str, Any],
    depth_sums: Dict[int, np.ndarray],
    depth_counts: Dict[int, int],
    gt_channel: int,
    gt_relative_directory: str,
    require_all_gt: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    cv2 = _opencv()
    gt_directory, gt_by_id = _find_gt_depths(
        _keyframe_directory(sequence),
        relative_directory=gt_relative_directory,
    )
    prediction_ids = set(depth_sums)
    gt_ids = set(gt_by_id)
    matched_ids = sorted(prediction_ids & gt_ids)
    missing_prediction_ids = sorted(gt_ids - prediction_ids)
    prediction_without_gt_ids = sorted(prediction_ids - gt_ids)
    if require_all_gt and missing_prediction_ids:
        raise RuntimeError(
            "Endo3R evaluation requires predictions for every GT frame in {}. "
            "Missing IDs: {}".format(
                sequence["sequence_id"], missing_prediction_ids[:20]
            )
        )
    if not matched_ids:
        raise RuntimeError(
            "No prediction/GT frame IDs match for {}".format(
                sequence["sequence_id"]
            )
        )

    gt_depths: List[np.ndarray] = []
    pred_depths: List[np.ndarray] = []
    for identifier in matched_ids:
        gt_depth = _load_endo3r_gt_depth(
            gt_by_id[identifier], gt_channel
        )
        pred_depth = depth_sums[identifier] / float(depth_counts[identifier])
        # These are the exact Endo3R SCARED evaluation size and interpolation.
        gt_depths.append(
            cv2.resize(
                gt_depth,
                (ENDO3R_WIDTH, ENDO3R_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        pred_depths.append(
            cv2.resize(
                pred_depth.astype(np.float32),
                (ENDO3R_WIDTH, ENDO3R_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            )
        )

    errors, mean_errors, ratio = depth_evaluation(
        gt_depths,
        pred_depths,
        min_depth=ENDO3R_MIN_DEPTH,
        max_depth=ENDO3R_MAX_DEPTH,
    )
    valid_masks = [
        (depth > ENDO3R_MIN_DEPTH) & (depth < ENDO3R_MAX_DEPTH)
        for depth in gt_depths
    ]
    empty_gt_after_resize_ids = [
        identifier
        for identifier, mask in zip(matched_ids, valid_masks)
        if not np.any(mask)
    ]
    gt_valid = np.concatenate(
        [depth[mask].astype(np.float64) for depth, mask in zip(gt_depths, valid_masks)]
    )
    pred_valid = np.concatenate(
        [depth[mask].astype(np.float64) for depth, mask in zip(pred_depths, valid_masks)]
    )
    aligned_valid = np.clip(
        pred_valid * ratio,
        ENDO3R_MIN_DEPTH,
        ENDO3R_MAX_DEPTH,
    )
    gt_scene_median = float(np.median(gt_valid))
    constant_errors = np.asarray(
        [
            compute_errors(
                depth[mask].astype(np.float64),
                np.full(int(mask.sum()), gt_scene_median, dtype=np.float64),
            )
            for depth, mask in zip(gt_depths, valid_masks)
            if np.any(mask)
        ],
        dtype=np.float64,
    )
    result = {
        "sequence_id": str(sequence["sequence_id"]),
        "gt_directory": str(gt_directory),
        "predicted_frame_count": len(prediction_ids),
        "gt_frame_count": len(gt_ids),
        "matched_frame_count": len(matched_ids),
        "evaluated_frame_count": len(errors),
        "missing_prediction_count": len(missing_prediction_ids),
        "prediction_without_gt_count": len(prediction_without_gt_ids),
        "missing_prediction_ids_preview": missing_prediction_ids[:20],
        "prediction_without_gt_ids_preview": prediction_without_gt_ids[:20],
        "scene_median_scaling_ratio": ratio,
        "empty_gt_after_resize_frame_count": len(empty_gt_after_resize_ids),
        "empty_gt_after_resize_frame_ids": empty_gt_after_resize_ids,
        "depth_diagnostics": {
            "units": {
                "gt": "metres after fixed Endo3R /1000 conversion",
                "prediction_raw": "student native scale",
                "prediction_aligned": "metres after scene median scaling",
            },
            "gt_valid_m": _distribution_stats(gt_valid),
            "prediction_raw": _distribution_stats(pred_valid),
            "prediction_aligned_m": _distribution_stats(aligned_valid),
            "oracle_constant_scene_median_baseline": {
                "description": (
                    "Diagnostic only: every valid pixel is predicted as the "
                    "scene GT median; not part of the Endo3R score"
                ),
                "constant_depth_m": gt_scene_median,
                "metrics": _metric_dict(constant_errors.mean(0)),
            },
        },
        "metrics": _metric_dict(mean_errors),
    }
    return errors, result


def evaluate(
    config_path: Path,
    checkpoint_override: Optional[Path],
    split_override: Optional[str],
    output_override: Optional[Path],
    limit_clips: Optional[int],
) -> Dict[str, Any]:
    config = load_config(config_path)
    eval_config = dict(config.get("evaluation", {}))
    checkpoint_path = checkpoint_override or Path(
        eval_config.get(
            "checkpoint",
            Path(config["training"]["output_dir"]) / "last.pt",
        )
    )
    split = split_override or str(eval_config.get("split", "test"))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Student checkpoint not found: {}".format(checkpoint_path)
        )
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    dataset_config = dict(config["dataset"])
    dataset_config["frame_source"] = str(
        eval_config.get("frame_source", PROJECT_FRAME_SOURCE)
    )
    # Include the final complete window so the fixed-length student covers the
    # full scene, matching Endo3R's sequence evaluation.
    dataset_config["drop_incomplete_clip"] = False
    dataset = make_scared_rgb_dataset(dataset_config, split)
    gt_channel = int(eval_config.get("gt_depth_channel", 0))
    gt_relative_directory = str(
        eval_config.get("gt_relative_directory", ENDO3R_GT_DIRECTORY)
    )
    require_all_gt = bool(eval_config.get("require_all_gt", True))

    sequence_by_id = {
        str(sequence["sequence_id"]): sequence
        for sequence in dataset.sequences
    }
    selected_frame_sources = sorted(
        {
            Path(str(sequence["frame_directory"])).name
            for sequence in dataset.sequences
        }
    )
    evaluable_sequence_ids = set()
    skipped_sequences_without_gt: List[Dict[str, str]] = []
    for sequence_id, sequence in sequence_by_id.items():
        try:
            _find_gt_depths(
                _keyframe_directory(sequence),
                relative_directory=gt_relative_directory,
            )
        except FileNotFoundError as error:
            skipped_sequences_without_gt.append(
                {"sequence_id": sequence_id, "reason": str(error)}
            )
            print(
                "Skipping sequence without Endo3R GT: {} ({})".format(
                    sequence_id, error
                )
            )
            continue
        evaluable_sequence_ids.add(sequence_id)

    evaluable_clip_indices = [
        index
        for index, clip in enumerate(dataset.clips)
        if str(clip.sequence["sequence_id"]) in evaluable_sequence_ids
    ]
    if not evaluable_clip_indices:
        raise RuntimeError(
            "No evaluable {} clips have Endo3R depthmap_rectified GT".format(
                split
            )
        )
    print(
        "Endo3R preflight: evaluable_sequences={} skipped_without_gt={} "
        "evaluable_clips={} frame_source={} selected_sources={}".format(
            len(evaluable_sequence_ids),
            len(skipped_sequences_without_gt),
            len(evaluable_clip_indices),
            dataset_config["frame_source"],
            ",".join(selected_frame_sources),
        )
    )

    loader = build_scared_dataloader(
        Subset(dataset, evaluable_clip_indices),
        batch_size=1,
        shuffle=False,
        num_workers=int(
            eval_config.get(
                "num_workers",
                config["dataloader"].get("num_workers", 0),
            )
        ),
        pin_memory=bool(eval_config.get("pin_memory", True)),
        persistent_workers=bool(
            eval_config.get("persistent_workers", True)
        ),
        prefetch_factor=int(
            config["dataloader"].get("prefetch_factor", 2)
        ),
        drop_last=False,
        seed=int(config.get("seed", 42)),
    )
    model = _load_student(checkpoint_path, config, device)
    amp_enabled = (
        bool(eval_config.get("amp", True)) and device.type == "cuda"
    )

    current_sequence_id: Optional[str] = None
    depth_sums: Dict[int, np.ndarray] = {}
    depth_counts: Dict[int, int] = {}
    sequence_results: List[Dict[str, Any]] = []
    inference_times: List[float] = []
    processed_clips = 0
    weighted_error_sum = np.zeros(len(METRIC_NAMES), dtype=np.float64)
    total_scene_length = 0

    def finish_sequence() -> None:
        nonlocal depth_sums, depth_counts, weighted_error_sum
        nonlocal total_scene_length
        if current_sequence_id is None:
            return
        _, sequence_result = _evaluate_sequence(
            sequence_by_id[current_sequence_id],
            depth_sums,
            depth_counts,
            gt_channel,
            gt_relative_directory,
            require_all_gt=require_all_gt and limit_clips is None,
        )
        # Endo3R weights each scene mean by the scene GT sequence length.
        scene_length = (
            sequence_result["gt_frame_count"]
            if limit_clips is None
            else sequence_result["evaluated_frame_count"]
        )
        sequence_metrics = np.asarray(
            [
                sequence_result["metrics"][name]
                for name in METRIC_NAMES
            ],
            dtype=np.float64,
        )
        weighted_error_sum += sequence_metrics * scene_length
        total_scene_length += int(scene_length)
        sequence_result["official_aggregation_weight"] = int(scene_length)
        sequence_results.append(sequence_result)
        print(
            "evaluated sequence={} matched={} evaluated={} scale={:.6f}".format(
                current_sequence_id,
                sequence_result["matched_frame_count"],
                sequence_result["evaluated_frame_count"],
                sequence_result["scene_median_scaling_ratio"],
            )
        )
        depth_sums, depth_counts = {}, {}

    with torch.inference_mode():
        for batch in loader:
            sequence_id = str(batch["sequence_id"][0])
            if (
                current_sequence_id is not None
                and sequence_id != current_sequence_id
            ):
                finish_sequence()
            current_sequence_id = sequence_id
            images = batch["images"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                prediction = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_times.append(time.perf_counter() - start)

            depths = (
                prediction["xyz_local"][0, ..., 2]
                .float()
                .cpu()
                .numpy()
            )
            for offset, frame_name in enumerate(batch["frame_names"][0]):
                identifier = extract_frame_id(frame_name)
                if identifier in depth_sums:
                    depth_sums[identifier] += depths[offset]
                    depth_counts[identifier] += 1
                else:
                    depth_sums[identifier] = depths[offset].copy()
                    depth_counts[identifier] = 1
            processed_clips += 1
            if (
                limit_clips is not None
                and processed_clips >= limit_clips
            ):
                break
    finish_sequence()

    if total_scene_length <= 0:
        raise RuntimeError("Endo3R evaluation produced no valid scene metrics")
    mean_errors = weighted_error_sum / float(total_scene_length)
    mean_inference_seconds = float(np.mean(inference_times))
    frames_per_clip = int(config["dataset"]["clip_length"])
    result = {
        "protocol": "Official Endo3R SCARED depth evaluation",
        "source_reference": (
            "https://github.com/wrld/Endo3R/blob/main/"
            "eval/depth_evaluation.py"
        ),
        "project_adaptations": [
            "student xyz_local[...,2] supplies predicted depth",
            "overlapping fixed-length clip predictions are averaged per frame",
            "prediction and GT are paired by numeric frame ID",
        ],
        "checkpoint": str(checkpoint_path),
        "split": split,
        "frame_source_selection": dataset_config["frame_source"],
        "selected_frame_sources": selected_frame_sources,
        "endo3r_official_frame_source": ENDO3R_FRAME_SOURCE,
        "gt_directory": gt_relative_directory,
        "min_depth": ENDO3R_MIN_DEPTH,
        "max_depth": ENDO3R_MAX_DEPTH,
        "gt_depth_scale": ENDO3R_GT_SCALE,
        "gt_depth_channel": gt_channel,
        "evaluation_resolution": [ENDO3R_WIDTH, ENDO3R_HEIGHT],
        "resize_interpolation": "cv2.INTER_NEAREST",
        "scale_alignment": "one median scaling ratio per scene",
        "scene_aggregation": "scene mean weighted by GT sequence length",
        "metric_units": {
            "abs_rel": "dimensionless",
            "sq_rel": "metres",
            "rmse": "metres",
            "rmse_log": "dimensionless",
            "a1_a2_a3": "fractions",
        },
        "processed_clip_count": processed_clips,
        "available_evaluable_clip_count": len(evaluable_clip_indices),
        "evaluated_sequence_count": len(sequence_results),
        "evaluated_frame_count": int(
            sum(item["evaluated_frame_count"] for item in sequence_results)
        ),
        "official_total_scene_length": total_scene_length,
        "skipped_sequence_without_gt_count": len(
            skipped_sequences_without_gt
        ),
        "skipped_sequences_without_gt": skipped_sequences_without_gt,
        "metrics": _metric_dict(mean_errors),
        "average_inference_ms_per_clip": mean_inference_seconds * 1000.0,
        "average_inference_ms_per_input_frame": (
            mean_inference_seconds * 1000.0 / frames_per_clip
        ),
        "input_frames_per_second": (
            frames_per_clip / mean_inference_seconds
        ),
        "sequences": sequence_results,
    }
    output_path = output_override or Path(
        eval_config.get(
            "output",
            checkpoint_path.parent
            / "evaluation_{}_endo3r.json".format(split),
        )
    )
    ensure_dir(output_path.parent)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "\n "
        + ("{:>8} | " * 7).format(
            "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
        )
    )
    print(
        ("&{: 8.3f} " * 7).format(*mean_errors.tolist()) + "\\\\"
    )
    print(
        "unit check: rmse={:.6f} m ({:.3f} mm), sq_rel={:.6f} m".format(
            mean_errors[2],
            mean_errors[2] * 1000.0,
            mean_errors[1],
        )
    )
    print(
        "average inference: {:.1f} ms/clip, {:.1f} ms/input-frame".format(
            result["average_inference_ms_per_clip"],
            result["average_inference_ms_per_input_frame"],
        )
    )
    print("wrote Endo3R evaluation: {}".format(output_path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/student_distillation.yaml"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--split", choices=("train", "test"), default=None
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--limit-clips", type=int, default=None, help="Debug only"
    )
    args = parser.parse_args()
    evaluate(
        Path(args.config),
        args.checkpoint,
        args.split,
        args.output,
        args.limit_clips,
    )


if __name__ == "__main__":
    main()
