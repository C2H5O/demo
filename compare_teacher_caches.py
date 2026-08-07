"""Diagnostic comparison of teacher caches on SCARED datasets 8 and 9.

The comparison is paired: only clips, frames, ground-truth pixels, and teacher
valid pixels shared by both cache roots are evaluated.  Overlapping temporal
clips are averaged per frame.  This is a teacher-collapse diagnostic, not the
project's benchmark evaluator; ``evaluate.py`` implements the official Endo3R
protocol.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from datasets.ground_truth import (
    frame_id,
    index_depth_directory,
    load_depth,
)
from datasets.scared_clip_dataset import make_scared_rgb_dataset
from evaluation.depth_metrics import METRIC_NAMES, compute_errors
from utils.config import load_config


@dataclass
class FrameAccumulator:
    depth_sum: np.ndarray
    confidence_sum: np.ndarray
    valid_count: np.ndarray

    @classmethod
    def empty(cls, shape: Tuple[int, int]) -> "FrameAccumulator":
        return cls(
            depth_sum=np.zeros(shape, dtype=np.float64),
            confidence_sum=np.zeros(shape, dtype=np.float64),
            valid_count=np.zeros(shape, dtype=np.uint32),
        )

    def add(
        self,
        depth: np.ndarray,
        confidence: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        if depth.shape != self.depth_sum.shape:
            raise ValueError(
                "Overlapping cache entries have inconsistent shapes: {} and {}".format(
                    depth.shape, self.depth_sum.shape
                )
            )
        finite = (
            valid.astype(bool)
            & np.isfinite(depth)
            & np.isfinite(confidence)
            & (depth > 0)
        )
        self.depth_sum[finite] += depth[finite]
        self.confidence_sum[finite] += confidence[finite]
        self.valid_count[finite] += 1

    def averaged(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid = self.valid_count > 0
        denominator = np.maximum(self.valid_count, 1).astype(np.float64)
        return (
            self.depth_sum / denominator,
            self.confidence_sum / denominator,
            valid,
        )


FrameKey = Tuple[int, str, int]
SequenceKey = Tuple[int, str]


def _split_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    test_directory = path / "test"
    return test_directory if test_directory.is_dir() else path


def _cache_files(
    root: Path,
    dataset_ids: Sequence[int],
) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    allowed = {
        "dataset_{:02d}".format(identifier) for identifier in dataset_ids
    } | {"dataset_{}".format(identifier) for identifier in dataset_ids}
    for path in root.rglob("*.npz"):
        relative = path.relative_to(root)
        if not any(part.lower() in allowed for part in relative.parts):
            continue
        key = relative.as_posix()
        if key in result:
            raise RuntimeError("Duplicate relative cache path: {}".format(key))
        result[key] = path
    if not result:
        raise FileNotFoundError(
            "No dataset 8/9 NPZ caches found under {}".format(root)
        )
    return result


def _scalar(cache: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in cache:
        raise KeyError("Cache is missing required field {!r}".format(name))
    return np.asarray(cache[name]).item()


def _metadata(
    cache: Mapping[str, np.ndarray],
    path: Path,
) -> Tuple[int, str]:
    dataset_id = int(_scalar(cache, "dataset_id"))
    if "metadata_json" in cache:
        metadata = json.loads(str(_scalar(cache, "metadata_json")))
        keyframe_id = str(metadata["keyframe_id"])
        metadata_dataset_id = int(metadata["dataset_id"])
        if metadata_dataset_id != dataset_id:
            raise RuntimeError(
                "Conflicting dataset IDs in {}: {} and {}".format(
                    path, dataset_id, metadata_dataset_id
                )
            )
    else:
        keyframe_id = path.parent.name
    return dataset_id, keyframe_id


def _load_cache_pair(
    base_path: Path,
    finetuned_path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int, str]:
    required = (
        "xyz_local",
        "conf_local",
        "valid_mask",
        "frame_names",
        "frame_indices",
        "dataset_id",
    )
    with np.load(str(base_path), allow_pickle=False) as base_file:
        missing = [name for name in required if name not in base_file]
        if missing:
            raise RuntimeError("{} is missing {}".format(base_path, missing))
        base = {name: np.asarray(base_file[name]) for name in base_file.files}
    with np.load(str(finetuned_path), allow_pickle=False) as fine_file:
        missing = [name for name in required if name not in fine_file]
        if missing:
            raise RuntimeError("{} is missing {}".format(finetuned_path, missing))
        fine = {name: np.asarray(fine_file[name]) for name in fine_file.files}

    base_identity = _metadata(base, base_path)
    fine_identity = _metadata(fine, finetuned_path)
    if base_identity != fine_identity:
        raise RuntimeError(
            "Cache identity mismatch: {} has {}, {} has {}".format(
                base_path, base_identity, finetuned_path, fine_identity
            )
        )
    for name in ("frame_names", "frame_indices"):
        if not np.array_equal(base[name], fine[name]):
            raise RuntimeError(
                "Paired caches disagree on {}: {} and {}".format(
                    name, base_path, finetuned_path
                )
            )
    if base["xyz_local"].shape != fine["xyz_local"].shape:
        raise RuntimeError(
            "Paired xyz_local shapes differ: {} and {}".format(
                base["xyz_local"].shape, fine["xyz_local"].shape
            )
        )
    return base, fine, base_identity[0], base_identity[1]


def _accumulate_cache(
    bank: Dict[FrameKey, FrameAccumulator],
    cache: Mapping[str, np.ndarray],
    dataset_id: int,
    keyframe_id: str,
) -> None:
    points = np.asarray(cache["xyz_local"], dtype=np.float32)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("xyz_local must have shape [T,H,W,3], got {}".format(points.shape))
    depth = points[..., 2]
    confidence = np.asarray(cache["conf_local"], dtype=np.float32)
    valid = np.asarray(cache["valid_mask"], dtype=bool)
    if confidence.shape != depth.shape or valid.shape != depth.shape:
        raise ValueError(
            "Cache depth/confidence/valid shapes differ: {}, {}, {}".format(
                depth.shape, confidence.shape, valid.shape
            )
        )
    names = [str(name) for name in np.asarray(cache["frame_names"]).tolist()]
    if len(names) != depth.shape[0]:
        raise ValueError("frame_names length does not match cached frame count")
    for index, name in enumerate(names):
        key = (dataset_id, keyframe_id, frame_id(name))
        accumulator = bank.get(key)
        if accumulator is None:
            accumulator = FrameAccumulator.empty(tuple(depth[index].shape))
            bank[key] = accumulator
        accumulator.add(depth[index], confidence[index], valid[index])


def _sequence_map(
    config: Dict[str, Any],
    dataset_root: Path | None,
) -> Dict[SequenceKey, Dict[str, Any]]:
    dataset_config = dict(config["dataset"])
    if dataset_root is not None:
        dataset_config["root"] = str(dataset_root)
    dataset = make_scared_rgb_dataset(dataset_config, "test")
    result = {}
    for sequence in dataset.sequences:
        key = (int(sequence["dataset_id"]), str(sequence["keyframe_id"]))
        if key in result:
            raise RuntimeError("Duplicate discovered SCARED sequence {}".format(key))
        result[key] = sequence
    return result


def _candidate_gt_directories(
    sequence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> List[Path]:
    values: List[Path] = []
    ground_truth = config.get("dataset", {}).get("ground_truth", {})
    for name in ground_truth.get(
        "directory_keys", ("depth_directory", "scene_points_directory")
    ):
        value = sequence.get(str(name))
        if value:
            values.append(Path(str(value)))
    keyframe_directory = Path(str(sequence["frame_directory"])).parent.parent
    relatives: Iterable[str] = config.get("evaluation", {}).get(
        "gt_directory_candidates",
        ground_truth.get(
            "relative_directories",
            ("data/depth", "data/scene_points"),
        ),
    )
    values.extend(keyframe_directory / str(relative) for relative in relatives)
    unique: List[Path] = []
    seen = set()
    for path in values:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _gt_index(
    sequence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[Path, Dict[int, Path]]:
    checked = []
    for directory in _candidate_gt_directories(sequence, config):
        checked.append(directory)
        indexed = index_depth_directory(directory)
        if indexed:
            return directory, indexed
    raise FileNotFoundError(
        "No ground-truth depth for {}. Checked {}".format(
            sequence.get("sequence_id"), [str(path) for path in checked]
        )
    )


def _resize(
    array: np.ndarray,
    shape: Tuple[int, int],
    mode: str,
) -> np.ndarray:
    if tuple(array.shape) == shape:
        return array
    tensor = torch.from_numpy(np.asarray(array)).float()[None, None]
    if mode == "nearest":
        resized = F.interpolate(tensor, size=shape, mode=mode)
    else:
        resized = F.interpolate(
            tensor, size=shape, mode=mode, align_corners=False
        )
    return resized[0, 0].numpy()


def _normalized_gradient(depth: np.ndarray, valid: np.ndarray) -> float:
    values = depth[valid]
    if values.size == 0:
        return float("nan")
    scale = float(np.median(values))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    normalized = depth / scale
    valid_x = valid[:, 1:] & valid[:, :-1]
    valid_y = valid[1:, :] & valid[:-1, :]
    gradients: List[np.ndarray] = []
    if np.any(valid_x):
        gradients.append(np.abs(normalized[:, 1:] - normalized[:, :-1])[valid_x])
    if np.any(valid_y):
        gradients.append(np.abs(normalized[1:, :] - normalized[:-1, :])[valid_y])
    if not gradients:
        return float("nan")
    return float(np.concatenate(gradients).mean())


def _evaluate_frame(
    gt_depth: np.ndarray,
    base: FrameAccumulator,
    finetuned: FrameAccumulator,
    min_depth: float,
    max_depth: float,
    min_valid_pixels: int,
) -> Dict[str, Any] | None:
    base_depth, base_confidence, base_valid = base.averaged()
    fine_depth, fine_confidence, fine_valid = finetuned.averaged()
    shape = tuple(gt_depth.shape)
    base_depth = _resize(base_depth.astype(np.float32), shape, "bilinear")
    fine_depth = _resize(fine_depth.astype(np.float32), shape, "bilinear")
    base_confidence = _resize(
        base_confidence.astype(np.float32), shape, "bilinear"
    )
    fine_confidence = _resize(
        fine_confidence.astype(np.float32), shape, "bilinear"
    )
    base_valid = _resize(base_valid.astype(np.uint8), shape, "nearest").astype(bool)
    fine_valid = _resize(fine_valid.astype(np.uint8), shape, "nearest").astype(bool)

    gt_valid = (
        np.isfinite(gt_depth)
        & (gt_depth > min_depth)
        & (gt_depth < max_depth)
    )
    base_support = base_valid & np.isfinite(base_depth) & (base_depth > 0)
    fine_support = fine_valid & np.isfinite(fine_depth) & (fine_depth > 0)
    common = gt_valid & base_support & fine_support
    common_count = int(common.sum())
    if common_count < min_valid_pixels:
        return None

    gt_values = gt_depth[common].astype(np.float64)
    base_values = base_depth[common].astype(np.float64)
    fine_values = fine_depth[common].astype(np.float64)
    base_ratio = float(np.median(gt_values) / np.median(base_values))
    fine_ratio = float(np.median(gt_values) / np.median(fine_values))
    if not np.isfinite(base_ratio) or not np.isfinite(fine_ratio):
        return None
    base_scaled = np.clip(base_values * base_ratio, min_depth, max_depth)
    fine_scaled = np.clip(fine_values * fine_ratio, min_depth, max_depth)

    base_mean = float(np.mean(base_values))
    fine_mean = float(np.mean(fine_values))
    gt_count = max(int(gt_valid.sum()), 1)
    return {
        "base_errors": compute_errors(gt_values, base_scaled),
        "finetuned_errors": compute_errors(gt_values, fine_scaled),
        "base_scale_ratio": base_ratio,
        "finetuned_scale_ratio": fine_ratio,
        "base_coverage": float((gt_valid & base_support).sum() / gt_count),
        "finetuned_coverage": float((gt_valid & fine_support).sum() / gt_count),
        "common_coverage": float(common_count / gt_count),
        "base_confidence": float(np.mean(base_confidence[common])),
        "finetuned_confidence": float(np.mean(fine_confidence[common])),
        "base_depth_cv": float(np.std(base_values) / max(abs(base_mean), 1e-12)),
        "finetuned_depth_cv": float(
            np.std(fine_values) / max(abs(fine_mean), 1e-12)
        ),
        "base_normalized_gradient": _normalized_gradient(base_depth, common),
        "finetuned_normalized_gradient": _normalized_gradient(fine_depth, common),
        "valid_pixels": common_count,
    }


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    base_errors = np.stack([record["base_errors"] for record in records])
    fine_errors = np.stack([record["finetuned_errors"] for record in records])
    base_metrics = {
        name: float(value) for name, value in zip(METRIC_NAMES, base_errors.mean(0))
    }
    fine_metrics = {
        name: float(value) for name, value in zip(METRIC_NAMES, fine_errors.mean(0))
    }
    higher_is_better = {"a1", "a2", "a3"}
    improvement = {
        name: (
            fine_metrics[name] - base_metrics[name]
            if name in higher_is_better
            else base_metrics[name] - fine_metrics[name]
        )
        for name in METRIC_NAMES
    }
    diagnostic_names = (
        "base_scale_ratio",
        "finetuned_scale_ratio",
        "base_coverage",
        "finetuned_coverage",
        "common_coverage",
        "base_confidence",
        "finetuned_confidence",
        "base_depth_cv",
        "finetuned_depth_cv",
        "base_normalized_gradient",
        "finetuned_normalized_gradient",
    )
    diagnostics = {
        name: _finite_mean(float(record[name]) for record in records)
        for name in diagnostic_names
    }
    return {
        "evaluated_frames": len(records),
        "evaluated_pixels": int(sum(int(record["valid_pixels"]) for record in records)),
        "base": base_metrics,
        "finetuned": fine_metrics,
        "finetuned_improvement": improvement,
        "finetuned_abs_rel_win_rate": float(
            np.mean(fine_errors[:, 0] < base_errors[:, 0])
        ),
        "diagnostics": diagnostics,
    }


def _print_summary(label: str, summary: Mapping[str, Any]) -> None:
    print(
        "\n{}: frames={} common_pixels={} fine_AbsRel_win_rate={:.2%}".format(
            label,
            summary["evaluated_frames"],
            summary["evaluated_pixels"],
            summary["finetuned_abs_rel_win_rate"],
        )
    )
    print(
        "{:<12} {}".format(
            "model", " ".join("{:>10}".format(name) for name in METRIC_NAMES)
        )
    )
    for model_name in ("base", "finetuned"):
        metrics = summary[model_name]
        print(
            "{:<12} {}".format(
                model_name,
                " ".join(
                    "{:10.5f}".format(float(metrics[name]))
                    for name in METRIC_NAMES
                ),
            )
        )
    print(
        "{:<12} {}".format(
            "improvement",
            " ".join(
                "{:+10.5f}".format(
                    float(summary["finetuned_improvement"][name])
                )
                for name in METRIC_NAMES
            ),
        )
    )
    diagnostics = summary["diagnostics"]
    print(
        "coverage base/fine/common={:.4f}/{:.4f}/{:.4f}  "
        "confidence base/fine={:.4f}/{:.4f}".format(
            diagnostics["base_coverage"],
            diagnostics["finetuned_coverage"],
            diagnostics["common_coverage"],
            diagnostics["base_confidence"],
            diagnostics["finetuned_confidence"],
        )
    )
    print(
        "depth_cv base/fine={:.6f}/{:.6f}  "
        "normalized_gradient base/fine={:.6f}/{:.6f}".format(
            diagnostics["base_depth_cv"],
            diagnostics["finetuned_depth_cv"],
            diagnostics["base_normalized_gradient"],
            diagnostics["finetuned_normalized_gradient"],
        )
    )


def compare(
    config_path: Path,
    base_cache_root: Path,
    finetuned_cache_root: Path,
    output_path: Path,
    dataset_root: Path | None = None,
    allow_partial_cache: bool = False,
    limit_clips: int | None = None,
    min_valid_pixels: int = 100,
) -> Dict[str, Any]:
    config = load_config(config_path)
    base_root = _split_root(base_cache_root)
    fine_root = _split_root(finetuned_cache_root)
    dataset_ids = (8, 9)
    base_files = _cache_files(base_root, dataset_ids)
    fine_files = _cache_files(fine_root, dataset_ids)
    common_paths = sorted(set(base_files) & set(fine_files))
    missing_finetuned = sorted(set(base_files) - set(fine_files))
    missing_base = sorted(set(fine_files) - set(base_files))
    if not allow_partial_cache and (missing_finetuned or missing_base):
        raise RuntimeError(
            "Cache clip sets differ. Missing fine-tuned={} missing base={}. "
            "Regenerate both test caches with identical clip settings or pass "
            "--allow-partial-cache.".format(
                missing_finetuned[:10], missing_base[:10]
            )
        )
    if limit_clips is not None:
        if limit_clips <= 0:
            raise ValueError("--limit-clips must be positive")
        common_paths = common_paths[:limit_clips]
    if not common_paths:
        raise RuntimeError("The two cache roots have no matching test clips")

    base_bank: Dict[FrameKey, FrameAccumulator] = {}
    fine_bank: Dict[FrameKey, FrameAccumulator] = {}
    for index, relative in enumerate(common_paths, start=1):
        base, fine, dataset_id, keyframe_id = _load_cache_pair(
            base_files[relative], fine_files[relative]
        )
        if dataset_id not in dataset_ids:
            continue
        _accumulate_cache(base_bank, base, dataset_id, keyframe_id)
        _accumulate_cache(fine_bank, fine, dataset_id, keyframe_id)
        if index % 100 == 0 or index == len(common_paths):
            print("loaded paired clips: {}/{}".format(index, len(common_paths)))

    if set(base_bank) != set(fine_bank):
        raise RuntimeError("Paired cache accumulation produced different frame sets")
    sequences = _sequence_map(config, dataset_root)
    evaluation = config.get("evaluation", {})
    ground_truth = config.get("dataset", {}).get("ground_truth", {})
    min_depth = float(evaluation.get("min_depth", 1e-3))
    max_depth = float(evaluation.get("max_depth", 150.0))
    gt_scale = float(
        evaluation.get("gt_depth_scale", ground_truth.get("scale", 1.0))
    )
    gt_channel = int(
        evaluation.get("gt_depth_channel", ground_truth.get("channel", 0))
    )

    gt_cache: Dict[SequenceKey, Tuple[Path, Dict[int, Path]]] = {}
    records_by_dataset: Dict[int, List[Dict[str, Any]]] = {8: [], 9: []}
    skipped: List[Dict[str, Any]] = []
    evaluated_sequences = set()
    for key in sorted(base_bank):
        dataset_id, keyframe_id, identifier = key
        sequence_key = (dataset_id, keyframe_id)
        sequence = sequences.get(sequence_key)
        if sequence is None:
            skipped.append(
                {
                    "frame": list(key),
                    "reason": "sequence not discovered from configured test dataset",
                }
            )
            continue
        if sequence_key not in gt_cache:
            try:
                gt_cache[sequence_key] = _gt_index(sequence, config)
            except FileNotFoundError as error:
                skipped.append(
                    {
                        "sequence": [dataset_id, keyframe_id],
                        "reason": str(error),
                    }
                )
                gt_cache[sequence_key] = (Path(), {})
        gt_directory, indexed_gt = gt_cache[sequence_key]
        gt_path = indexed_gt.get(identifier)
        if gt_path is None:
            skipped.append(
                {
                    "frame": list(key),
                    "reason": "ground-truth frame ID not found",
                    "gt_directory": str(gt_directory),
                }
            )
            continue
        gt_depth = load_depth(gt_path, gt_scale, gt_channel)
        record = _evaluate_frame(
            gt_depth,
            base_bank[key],
            fine_bank[key],
            min_depth,
            max_depth,
            min_valid_pixels,
        )
        if record is None:
            skipped.append(
                {
                    "frame": list(key),
                    "reason": "insufficient shared valid pixels",
                }
            )
            continue
        record["dataset_id"] = dataset_id
        record["keyframe_id"] = keyframe_id
        record["frame_id"] = identifier
        records_by_dataset[dataset_id].append(record)
        evaluated_sequences.add(sequence_key)

    empty = [identifier for identifier, records in records_by_dataset.items() if not records]
    if empty:
        raise RuntimeError(
            "No evaluable paired frames remained for dataset IDs {}".format(empty)
        )
    summaries = {
        "dataset_{}".format(identifier): _summarize(records)
        for identifier, records in records_by_dataset.items()
    }
    all_records = records_by_dataset[8] + records_by_dataset[9]
    summaries["overall"] = _summarize(all_records)
    result = {
        "protocol": {
            "dataset_ids": list(dataset_ids),
            "paired_common_valid_pixels": True,
            "overlapping_clips_averaged_per_frame": True,
            "per_frame_median_scaling": True,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "gt_scale": gt_scale,
            "gt_channel": gt_channel,
            "min_valid_pixels": min_valid_pixels,
        },
        "paths": {
            "config": str(config_path.resolve()),
            "dataset_root_override": (
                str(dataset_root.resolve()) if dataset_root is not None else None
            ),
            "base_cache": str(base_root),
            "finetuned_cache": str(fine_root),
        },
        "cache_pairing": {
            "base_clip_count": len(base_files),
            "finetuned_clip_count": len(fine_files),
            "paired_clip_count": len(common_paths),
            "missing_finetuned_count": len(missing_finetuned),
            "missing_base_count": len(missing_base),
            "missing_finetuned_preview": missing_finetuned[:20],
            "missing_base_preview": missing_base[:20],
        },
        "evaluated_sequence_count": len(evaluated_sequences),
        "skipped_count": len(skipped),
        "skipped_preview": skipped[:100],
        "results": summaries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for label in ("dataset_8", "dataset_9", "overall"):
        _print_summary(label, summaries[label])
    print("\nwrote comparison report: {}".format(output_path.resolve()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/student_distillation.yaml"),
    )
    parser.add_argument(
        "--base-cache",
        type=Path,
        required=True,
        help="Unmodified/base teacher cache root (with or without a test/ suffix)",
    )
    parser.add_argument(
        "--finetuned-cache",
        type=Path,
        required=True,
        help="Fine-tuned teacher cache root (with or without a test/ suffix)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional override for dataset.root in the YAML config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/teacher_cache_comparison_test_8_9.json"),
    )
    parser.add_argument(
        "--allow-partial-cache",
        action="store_true",
        help="Evaluate the clip intersection instead of requiring identical cache sets",
    )
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--min-valid-pixels", type=int, default=100)
    args = parser.parse_args()
    if args.min_valid_pixels <= 0:
        parser.error("--min-valid-pixels must be positive")
    compare(
        config_path=args.config,
        base_cache_root=args.base_cache,
        finetuned_cache_root=args.finetuned_cache,
        output_path=args.output,
        dataset_root=args.dataset_root,
        allow_partial_cache=args.allow_partial_cache,
        limit_clips=args.limit_clips,
        min_valid_pixels=args.min_valid_pixels,
    )


if __name__ == "__main__":
    main()
