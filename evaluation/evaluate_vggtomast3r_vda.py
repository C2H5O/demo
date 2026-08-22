"""Video-Depth-Anything evaluation adapter for the VGG-to-MASt3R V1 model."""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

import evaluation.evaluate_vda as vda_core
from datasets.scared_pair_dataset import make_scared_pair_rgb_dataset
from models.student.dune_mast3r_adapter import DuneMast3RStudent
from utils.checkpoint import require_student_cache_protocol
from utils.config import ensure_dir, load_config


def _select_evaluation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Select the V1 VDA section while retaining legacy fallback behavior."""
    return dict(config.get("vda_evaluation", config.get("evaluation", {})))


def _load_model(
    checkpoint_path: Path,
    fallback_config: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Load only the model state, following the old VDA memory-bounded path."""
    try:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except (TypeError, RuntimeError) as error:
        if "mmap" not in str(error).lower() and not isinstance(error, TypeError):
            raise
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
    require_student_cache_protocol(checkpoint)
    checkpoint_config = (
        checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    )
    model_config = dict(
        checkpoint_config.get("student", fallback_config["student"])
    )
    state = (
        checkpoint.get("model", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if state is not checkpoint:
        checkpoint.clear()
    checkpoint = None
    checkpoint_config = None
    gc.collect()

    model = DuneMast3RStudent(model_config, device=device)
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except TypeError:
        model.load_state_dict(state, strict=True)
    state = None
    gc.collect()
    return model.eval().to(device)


def _pair_reference_disparities(
    model: torch.nn.Module, images: torch.Tensor
) -> np.ndarray:
    """Predict camera-local A/B depth using reference output in both orders."""
    forward = model(images)
    depth_a = forward["pts3d_ref"][0, ..., 2].float().cpu().numpy()
    depth_b = forward["pts3d_other_local"][0, ..., 2].float().cpu().numpy()
    return np.stack(
        (
            vda_core._student_depth_to_vda_disparity(depth_a),
            vda_core._student_depth_to_vda_disparity(depth_b),
        )
    )


def evaluate(
    config_path: Path,
    checkpoint_override: Optional[Path] = None,
    split_override: Optional[str] = None,
    output_override: Optional[Path] = None,
    limit_pairs: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the unchanged VDA alignment/metrics on V1 pair predictions."""
    config = load_config(config_path)
    eval_config = _select_evaluation_config(config)
    protocol = str(eval_config.get("protocol", "vda")).lower()
    if protocol not in ("vda", "video-depth-anything-depth"):
        raise ValueError("vda_evaluation.protocol must be vda")
    split = split_override or str(eval_config.get("split", "test"))
    checkpoint_path = checkpoint_override or Path(
        eval_config.get(
            "checkpoint", Path(config["training"]["output_dir"]) / "last.pt"
        )
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Student checkpoint not found: {}".format(checkpoint_path)
        )
    output_path = output_override or Path(
        eval_config.get(
            "output",
            checkpoint_path.parent / "evaluation_{}_vda.json".format(split),
        )
    )
    ensure_dir(output_path.parent)

    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    dataset_config = dict(config["dataset"])
    dataset_config["frame_source"] = str(
        eval_config.get(
            "frame_source", dataset_config.get("frame_source", "auto")
        )
    )
    dataset = make_scared_pair_rgb_dataset(dataset_config, split)
    evaluation_height = int(dataset_config["image_height"])
    evaluation_width = int(dataset_config["image_width"])
    gt_channel = int(eval_config.get("gt_depth_channel", 0))
    require_all_gt = bool(eval_config.get("require_all_gt", True))

    sequence_by_id = {
        str(sequence["sequence_id"]): sequence for sequence in dataset.sequences
    }
    gt_depths_by_sequence = {}
    skipped_sequences: List[Dict[str, str]] = []
    for sequence_id, sequence in sequence_by_id.items():
        try:
            gt_depths_by_sequence[sequence_id] = (
                vda_core._find_sequence_gt_depths(
                    sequence, eval_config, dataset_config
                )
            )
        except FileNotFoundError as error:
            skipped_sequences.append(
                {"sequence_id": sequence_id, "reason": str(error)}
            )

    pair_indices_by_sequence: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(dataset.clips):
        sequence_id = str(record.sequence["sequence_id"])
        if sequence_id in gt_depths_by_sequence:
            pair_indices_by_sequence[sequence_id].append(index)
    expected_pair_count = sum(
        len(indices) for indices in pair_indices_by_sequence.values()
    )
    if expected_pair_count == 0:
        raise RuntimeError(
            "No VGG-to-MASt3R pair sequences contain configured depth GT"
        )

    model = _load_model(checkpoint_path, config, device)
    amp_enabled = bool(eval_config.get("amp", True)) and device.type == "cuda"
    sequence_results: List[Dict[str, Any]] = []
    inference_times: List[float] = []
    processed_pairs = 0
    remaining = limit_pairs

    for sequence_id, sequence in sequence_by_id.items():
        indices = pair_indices_by_sequence.get(sequence_id, [])
        if not indices or remaining == 0:
            continue
        if remaining is not None:
            indices = indices[:remaining]
        spool = vda_core._SequencePredictionSpool(
            output_path.parent,
            int(sequence["sequence_length"]),
            evaluation_height,
            evaluation_width,
        )
        try:
            with torch.inference_mode():
                for index in indices:
                    sample = dataset[index]
                    images = sample["images"].unsqueeze(0).to(device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        disparities = _pair_reference_disparities(model, images)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_times.append(time.perf_counter() - started)
                    frame_indices: Sequence[int] = [
                        int(value) for value in sample["frame_indices"].tolist()
                    ]
                    spool.add(frame_indices, disparities)
                    processed_pairs += 1
            spool.flush()
            sequence_result = vda_core._evaluate_sequence(
                sequence,
                spool,
                gt_channel,
                gt_depths_by_sequence[sequence_id],
                require_all_gt=(require_all_gt and limit_pairs is None),
            )
            sequence_results.append(sequence_result)
            print(
                "[VDA/V1] sequence={} pairs={} frames={} metrics={}".format(
                    sequence_id,
                    len(indices),
                    sequence_result["matched_frame_count"],
                    sequence_result["metrics"],
                ),
                flush=True,
            )
        finally:
            spool.close()
        if remaining is not None:
            remaining -= len(indices)

    if not sequence_results:
        raise RuntimeError("No sequence was evaluated")
    metrics = {
        name: float(
            np.mean([result["metrics"][name] for result in sequence_results])
        )
        for name in vda_core.VDA_METRIC_NAMES
    }
    all_pairs_processed = (
        limit_pairs is None and processed_pairs == expected_pair_count
    )
    complete_gt_coverage = all(
        item["missing_prediction_count"] == 0 for item in sequence_results
    )
    result = {
        "protocol": "video-depth-anything-depth",
        "source": (
            "https://github.com/DepthAnything/Video-Depth-Anything/"
            "tree/main/benchmark/eval"
        ),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": split,
        "metrics": metrics,
        "sequence_count": len(sequence_results),
        "processed_pair_count": processed_pairs,
        "expected_pair_count": expected_pair_count,
        "all_source_pairs_processed": all_pairs_processed,
        "complete_gt_coverage": complete_gt_coverage,
        "full_test_set": all_pairs_processed and complete_gt_coverage,
        "evaluation_size": [evaluation_width, evaluation_height],
        "evaluation_shape_hxw": [evaluation_height, evaluation_width],
        "mean_pair_inference_seconds": (
            float(np.mean(inference_times)) if inference_times else None
        ),
        "skipped_sequences_without_gt": skipped_sequences,
        "prediction_semantics": (
            "A/B camera-local depth uses pts3d_ref/pts3d_other_local Z, "
            "then reciprocal disparity"
        ),
        "other_output_semantics": (
            "pts3d_other_local is camera-B local and is not fused with camera A"
        ),
        "prediction_storage": "per-sequence disk memmap",
        "streaming_two_pass": True,
        "core_algorithm_modified": False,
        "sequences": sequence_results,
    }
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("wrote VGG-to-MASt3R VDA evaluation: {}".format(output_path))
    return result


__all__ = [
    "_pair_reference_disparities",
    "_select_evaluation_config",
    "evaluate",
]
