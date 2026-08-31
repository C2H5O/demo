"""VDA-default and Endo3R evaluation for the 16-frame cross-clip student."""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import evaluation.evaluate_vda as vda_core
from datasets.crossclip_teacher_dataset import (
    CROSSCLIP_CACHE_PROTOCOL,
    make_crossclip_rgb_dataset,
)
from evaluation.depth_metrics import METRIC_NAMES
from evaluation.evaluate_depth import (
    _evaluate_sequence as evaluate_endo3r_sequence,
    extract_frame_id,
)
from models.student.da3_small_student import DA3SmallStudent
from utils.checkpoint import require_student_cache_protocol
from utils.config import ensure_dir, load_config


def select_protocol(config: Dict[str, Any], override: Optional[str] = None) -> str:
    value = override or str(config.get("evaluation", {}).get("protocol", "vda"))
    protocol = value.strip().lower()
    if protocol == "video-depth-anything-depth":
        protocol = "vda"
    if protocol not in {"vda", "endo3r"}:
        raise ValueError("Evaluation protocol must be 'vda' or 'endo3r'")
    return protocol


def _load_model(
    checkpoint_path: Path, config: Dict[str, Any], device: torch.device
) -> DA3SmallStudent:
    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True
        )
    except (TypeError, RuntimeError):
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("Cross-clip checkpoint must contain model and config")
    require_student_cache_protocol(checkpoint, CROSSCLIP_CACHE_PROTOCOL)
    model_config = checkpoint.get("config", {}).get("student", config["student"])
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Cross-clip checkpoint has no model state")
    model = DA3SmallStudent(model_config, device=device)
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except TypeError:
        model.load_state_dict(state, strict=True)
    checkpoint.clear()
    del state, checkpoint
    gc.collect()
    return model.eval().to(device)


def _clip_depths(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    prediction = model(images)
    depth_all = prediction["depth"]
    if tuple(depth_all.shape[:2]) != (1, 16) or tuple(depth_all.shape[-2:]) != (448, 560):
        raise RuntimeError("Expected DA3 depth [1,16,448,560]")
    depth = depth_all[0].float()
    if not bool(torch.isfinite(depth).all()):
        raise FloatingPointError("Cross-clip student produced non-finite depth")
    return depth


def _dataset_and_ground_truth(
    config: Dict[str, Any], eval_config: Dict[str, Any], split: str
) -> Tuple[Any, Dict[str, Dict[str, Any]], Dict[str, Tuple[Path, Dict[int, Path]]], List[Dict[str, str]]]:
    dataset_config = dict(config["dataset"])
    dataset_config["frame_source"] = str(
        eval_config.get("frame_source", dataset_config.get("frame_source", "auto"))
    )
    # Add one final complete 16-frame window when a sequence length is not
    # aligned to stride eight, so every available RGB/GT tail frame is scored.
    dataset_config["drop_incomplete_clip"] = False
    # Detection/inpainting is a training-only auxiliary and does not alter RGB.
    dataset_config["highlight"] = {"enabled": False}
    dataset = make_crossclip_rgb_dataset(dataset_config, split)
    sequences = {str(item["sequence_id"]): item for item in dataset.sequences}
    gt_by_sequence: Dict[str, Tuple[Path, Dict[int, Path]]] = {}
    skipped: List[Dict[str, str]] = []
    for sequence_id, sequence in sequences.items():
        try:
            gt_by_sequence[sequence_id] = vda_core._find_sequence_gt_depths(
                sequence, eval_config, dataset_config
            )
        except FileNotFoundError as error:
            skipped.append({"sequence_id": sequence_id, "reason": str(error)})
    return dataset, sequences, gt_by_sequence, skipped


def _indices_by_sequence(dataset: Any, allowed: Sequence[str]) -> Dict[str, List[int]]:
    allowed_set = set(allowed)
    result: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(dataset.clips):
        sequence_id = str(record.sequence["sequence_id"])
        if sequence_id in allowed_set:
            result[sequence_id].append(index)
    return result


def evaluate_vda(
    config_path: Path,
    checkpoint_override: Optional[Path] = None,
    split_override: Optional[str] = None,
    output_override: Optional[Path] = None,
    limit_clips: Optional[int] = None,
) -> Dict[str, Any]:
    """Average overlapping clip disparities, then use the unchanged VDA core."""
    config = load_config(config_path)
    eval_config = dict(config.get("vda_evaluation", {}))
    split = split_override or str(eval_config.get("split", "test"))
    checkpoint = checkpoint_override or Path(str(eval_config["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError("Student checkpoint not found: {}".format(checkpoint))
    output = output_override or Path(str(eval_config["output"]))
    ensure_dir(output.parent)
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    dataset, sequences, gt_depths, skipped = _dataset_and_ground_truth(
        config, eval_config, split
    )
    by_sequence = _indices_by_sequence(dataset, gt_depths)
    expected = sum(len(value) for value in by_sequence.values())
    if expected == 0:
        raise RuntimeError("No cross-clip sequences contain configured depth GT")
    model = _load_model(checkpoint, config, device)
    amp = bool(eval_config.get("amp", True)) and device.type == "cuda"
    height = int(config["dataset"]["image_height"])
    width = int(config["dataset"]["image_width"])
    remaining = limit_clips
    processed = 0
    times: List[float] = []
    sequence_results: List[Dict[str, Any]] = []
    for sequence_id, sequence in sequences.items():
        indices = by_sequence.get(sequence_id, [])
        if not indices or remaining == 0:
            continue
        if remaining is not None:
            indices = indices[:remaining]
        spool = vda_core._SequencePredictionSpool(
            output.parent, int(sequence["sequence_length"]), height, width
        )
        try:
            with torch.inference_mode():
                for index in indices:
                    sample = dataset[index]
                    images = sample["images"].unsqueeze(0).to(device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.cuda.amp.autocast(enabled=amp):
                        depth = _clip_depths(model, images)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    times.append(time.perf_counter() - started)
                    disparities = np.stack(
                        [
                            vda_core._student_depth_to_vda_disparity(item)
                            for item in depth.cpu().numpy()
                        ]
                    )
                    # The cache's frame_indices are absolute source IDs.  The
                    # spool is indexed by sequence-local position so a
                    # canonical sequence may preserve non-zero source IDs.
                    spool.add(dataset.clips[index].frame_indices, disparities)
                    processed += 1
            spool.flush()
            sequence_results.append(
                vda_core._evaluate_sequence(
                    sequence,
                    spool,
                    int(eval_config.get("gt_depth_channel", 0)),
                    gt_depths[sequence_id],
                    require_all_gt=(
                        bool(eval_config.get("require_all_gt", True))
                        and limit_clips is None
                    ),
                )
            )
        finally:
            spool.close()
        if remaining is not None:
            remaining -= len(indices)
    if not sequence_results:
        raise RuntimeError("No sequence was evaluated")
    metrics = {
        name: float(np.mean([item["metrics"][name] for item in sequence_results]))
        for name in vda_core.VDA_METRIC_NAMES
    }
    complete = all(item["missing_prediction_count"] == 0 for item in sequence_results)
    result = {
        "protocol": "video-depth-anything-depth",
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "split": split,
        "metrics": metrics,
        "sequence_count": len(sequence_results),
        "processed_clip_count": processed,
        "expected_clip_count": expected,
        "all_source_clips_processed": limit_clips is None and processed == expected,
        "complete_gt_coverage": complete,
        "full_test_set": limit_clips is None and processed == expected and complete,
        "clip_length": 16,
        "clip_stride": 8,
        "overlap_reduction": "mean disparity per absolute frame",
        "prediction_semantics": "joint DA3 depth, then reciprocal disparity",
        "mean_clip_inference_seconds": float(np.mean(times)) if times else None,
        "skipped_sequences_without_gt": skipped,
        "sequences": sequence_results,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote cross-clip VDA evaluation: {}".format(output))
    return result


def evaluate_endo3r(
    config_path: Path,
    checkpoint_override: Optional[Path] = None,
    split_override: Optional[str] = None,
    output_override: Optional[Path] = None,
    limit_clips: Optional[int] = None,
) -> Dict[str, Any]:
    """Average overlapping camera-local Z maps before Endo3R scoring."""
    config = load_config(config_path)
    eval_config = dict(config.get("endo3r_evaluation", {}))
    split = split_override or str(eval_config.get("split", "test"))
    checkpoint = checkpoint_override or Path(str(eval_config["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError("Student checkpoint not found: {}".format(checkpoint))
    output = output_override or Path(str(eval_config["output"]))
    ensure_dir(output.parent)
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    dataset, sequences, gt_depths, skipped = _dataset_and_ground_truth(
        config, eval_config, split
    )
    by_sequence = _indices_by_sequence(dataset, gt_depths)
    expected = sum(len(value) for value in by_sequence.values())
    model = _load_model(checkpoint, config, device)
    amp = bool(eval_config.get("amp", True)) and device.type == "cuda"
    remaining = limit_clips
    processed = 0
    weighted = np.zeros(len(METRIC_NAMES), dtype=np.float64)
    total_weight = 0
    sequence_results: List[Dict[str, Any]] = []
    for sequence_id, sequence in sequences.items():
        indices = by_sequence.get(sequence_id, [])
        if not indices or remaining == 0:
            continue
        if remaining is not None:
            indices = indices[:remaining]
        sums: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}
        with torch.inference_mode():
            for index in indices:
                sample = dataset[index]
                with torch.cuda.amp.autocast(enabled=amp):
                    depths = _clip_depths(
                        model, sample["images"].unsqueeze(0).to(device)
                    ).cpu().numpy()
                for name, depth in zip(sample["frame_names"], depths):
                    identifier = extract_frame_id(name)
                    if identifier in sums:
                        sums[identifier] += depth
                        counts[identifier] += 1
                    else:
                        sums[identifier] = depth.copy()
                        counts[identifier] = 1
                processed += 1
        _, sequence_result = evaluate_endo3r_sequence(
            sequence,
            sums,
            counts,
            int(eval_config.get("gt_depth_channel", 0)),
            str(gt_depths[sequence_id][0]),
            bool(eval_config.get("require_all_gt", True)) and limit_clips is None,
        )
        weight = int(
            sequence_result["gt_frame_count"]
            if limit_clips is None
            else sequence_result["evaluated_frame_count"]
        )
        weighted += np.asarray(
            [sequence_result["metrics"][name] for name in METRIC_NAMES]
        ) * weight
        total_weight += weight
        sequence_results.append(sequence_result)
        if remaining is not None:
            remaining -= len(indices)
    if total_weight == 0:
        raise RuntimeError("No SCARED sequences were evaluated")
    metrics = {
        name: float(value)
        for name, value in zip(METRIC_NAMES, weighted / total_weight)
    }
    metrics.update({"delta1": metrics["a1"], "delta2": metrics["a2"], "delta3": metrics["a3"]})
    result = {
        "protocol": "Official Endo3R SCARED depth evaluation",
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "split": split,
        "metrics": metrics,
        "processed_clip_count": processed,
        "expected_clip_count": expected,
        "clip_length": 16,
        "clip_stride": 8,
        "overlap_reduction": "mean depth per absolute frame",
        "prediction_semantics": "joint DA3 depth with native camera head",
        "skipped_sequences_without_gt": skipped,
        "sequences": sequence_results,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote cross-clip Endo3R evaluation: {}".format(output))
    return result


def evaluate(
    config_path: Path,
    checkpoint: Optional[Path] = None,
    split: Optional[str] = None,
    output: Optional[Path] = None,
    limit_clips: Optional[int] = None,
    protocol: Optional[str] = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    selected = select_protocol(config, protocol)
    function = evaluate_vda if selected == "vda" else evaluate_endo3r
    return function(config_path, checkpoint, split, output, limit_clips)


__all__ = ["evaluate", "evaluate_endo3r", "evaluate_vda", "select_protocol"]
