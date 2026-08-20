"""Endo3R-compatible depth evaluation using reference-view pointmap Z only."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from datasets.ground_truth import frame_id
from datasets.scared_pair_dataset import make_scared_pair_rgb_dataset
from evaluation.depth_metrics import METRIC_NAMES
from evaluation.evaluate_depth import (
    ENDO3R_MAX_DEPTH,
    ENDO3R_MIN_DEPTH,
    _evaluate_sequence,
    _find_gt_depths,
    _keyframe_directory,
    _load_endo3r_gt_depth,
    _opencv,
)
from evaluation.vggtomast3r_metrics import patch_boundary_artifact
from models.student.dune_mast3r_adapter import DuneMast3RStudent
from utils.config import load_config


def _add_depth(
    sums: Dict[int, np.ndarray],
    counts: Dict[int, int],
    name: str,
    depth: torch.Tensor,
) -> None:
    identifier = frame_id(name)
    array = depth.detach().float().cpu().numpy()
    if identifier in sums:
        sums[identifier] += array
        counts[identifier] += 1
    else:
        sums[identifier] = array.copy()
        counts[identifier] = 1


def evaluate(
    config_path: Path,
    checkpoint_override: Optional[Path] = None,
    split_override: Optional[str] = None,
    output_override: Optional[Path] = None,
    limit_pairs: Optional[int] = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    eval_config = dict(config.get("evaluation", {}))
    if str(eval_config.get("protocol", "endo3r")).lower() != "endo3r":
        raise ValueError("vggtomast3r V1 evaluation.protocol must be endo3r")
    split = split_override or str(eval_config.get("split", "test"))
    checkpoint_path = checkpoint_override or Path(eval_config["checkpoint"])
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_config = checkpoint.get("config", {}).get("student", config["student"])
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    model = DuneMast3RStudent(model_config, device=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset_config = dict(config["dataset"])
    dataset_config["frame_source"] = str(eval_config.get("frame_source", "auto"))
    dataset = make_scared_pair_rgb_dataset(dataset_config, split)
    gt_relative = str(eval_config.get("gt_relative_directory", "data/depth"))
    evaluable_sequence_ids = set()
    skipped_sequences_without_gt = []
    for sequence in dataset.sequences:
        sequence_id = str(sequence["sequence_id"])
        try:
            _find_gt_depths(
                _keyframe_directory(sequence), relative_directory=gt_relative
            )
        except FileNotFoundError as error:
            skipped_sequences_without_gt.append(
                {"sequence_id": sequence_id, "reason": str(error)}
            )
        else:
            evaluable_sequence_ids.add(sequence_id)
    pairs_by_sequence: Dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.clips):
        sequence_id = str(record.sequence["sequence_id"])
        if sequence_id in evaluable_sequence_ids:
            pairs_by_sequence[sequence_id].append(index)
    available_pair_count = sum(len(indices) for indices in pairs_by_sequence.values())
    if available_pair_count == 0:
        raise RuntimeError("No pair sequences contain Endo3R depth ground truth")
    amp_enabled = bool(eval_config.get("amp", True)) and device.type == "cuda"
    weighted = np.zeros(len(METRIC_NAMES), dtype=np.float64)
    total_weight = 0
    sequences = []
    patch_records = []
    gt_channel = int(eval_config.get("gt_depth_channel", 0))
    require_all = bool(eval_config.get("require_all_gt", True)) and limit_pairs is None
    remaining = limit_pairs
    processed_pairs = 0
    for sequence in dataset.sequences:
        sequence_id = str(sequence["sequence_id"])
        pair_indices = pairs_by_sequence.get(sequence_id, [])
        if not pair_indices or remaining == 0:
            continue
        if remaining is not None:
            pair_indices = pair_indices[:remaining]
        depth_sums: Dict[int, np.ndarray] = {}
        depth_counts: Dict[int, int] = {}
        with torch.inference_mode():
            for index in pair_indices:
                sample = dataset[index]
                images = sample["images"].unsqueeze(0).to(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    forward = model(images)
                    reverse = model(images.flip(1))
                # Never use pts3d_other_in_ref[...,2] as second-camera depth.
                depth_a = forward["pts3d_ref"][0, ..., 2]
                depth_b = reverse["pts3d_ref"][0, ..., 2]
                _add_depth(depth_sums, depth_counts, sample["frame_names"][0], depth_a)
                _add_depth(depth_sums, depth_counts, sample["frame_names"][1], depth_b)
        processed_pairs += len(pair_indices)
        if remaining is not None:
            remaining -= len(pair_indices)
        _, sequence_result = _evaluate_sequence(
            sequence,
            depth_sums,
            depth_counts,
            gt_channel,
            gt_relative,
            require_all,
        )
        weight = sequence_result["gt_frame_count"] if limit_pairs is None else sequence_result["evaluated_frame_count"]
        weighted += np.asarray([sequence_result["metrics"][name] for name in METRIC_NAMES]) * weight
        total_weight += int(weight)
        sequences.append(sequence_result)
        if bool(eval_config.get("patch_artifact", True)):
            _, gt_by_id = _find_gt_depths(
                _keyframe_directory(sequence), relative_directory=gt_relative
            )
            cv2 = _opencv()
            for identifier in sorted(set(depth_sums) & set(gt_by_id)):
                averaged = depth_sums[identifier] / float(depth_counts[identifier])
                gt = _load_endo3r_gt_depth(gt_by_id[identifier], gt_channel)
                gt = cv2.resize(
                    gt,
                    (averaged.shape[1], averaged.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                valid = np.isfinite(gt) & (gt > ENDO3R_MIN_DEPTH) & (gt < ENDO3R_MAX_DEPTH)
                try:
                    patch_record = patch_boundary_artifact(
                        torch.from_numpy(averaged),
                        int(eval_config.get("patch_size", 14)),
                        torch.from_numpy(valid),
                    )
                except ValueError:
                    continue
                patch_records.append(patch_record)
    if total_weight == 0:
        raise RuntimeError("No SCARED sequences were evaluated")
    metrics = {name: float(value) for name, value in zip(METRIC_NAMES, weighted / total_weight)}
    metrics.update(
        {"delta1": metrics["a1"], "delta2": metrics["a2"], "delta3": metrics["a3"]}
    )
    patch_summary = None
    if patch_records:
        patch_summary = {
            key: float(np.mean([record[key] for record in patch_records]))
            for key in patch_records[0]
        }
    result = {
        "protocol": "Official Endo3R SCARED depth evaluation",
        "prediction_semantics": "each frame depth is pts3d_ref[...,2]; second frames use reverse pair inference",
        "forbidden_semantics": "pts3d_other_in_ref[...,2] is not second-camera depth",
        "checkpoint": str(checkpoint_path),
        "split": split,
        "processed_pair_count": processed_pairs,
        "available_evaluable_pair_count": available_pair_count,
        "skipped_sequences_without_gt": skipped_sequences_without_gt,
        "pair_stride": int(config["dataset"]["pair_stride"]),
        "metrics": metrics,
        "patch_artifact": patch_summary,
        "sequences": sequences,
    }
    output_path = output_override or Path(eval_config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    return result


__all__ = ["evaluate"]
