"""Audit and optionally align frozen teacher clip scales entirely offline."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from datasets.crossclip_teacher_dataset import (
    crossclip_teacher_cache_path,
    make_crossclip_rgb_dataset,
    validate_crossclip_teacher_cache,
)
from datasets.scared_clip_dataset import clip_metadata
from utils.config import load_config


def estimate_teacher_overlap_scale(
    previous_depth: np.ndarray,
    current_depth: np.ndarray,
    previous_valid: np.ndarray,
    current_valid: np.ndarray,
    previous_confidence: Optional[np.ndarray] = None,
    current_confidence: Optional[np.ndarray] = None,
    previous_highlight: Optional[np.ndarray] = None,
    current_highlight: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.0,
    minimum_valid_pixels: int = 64,
    eps: float = 1e-6,
) -> Tuple[float, List[float]]:
    """Return scale multiplying current raw geometry to match previous aligned geometry."""
    if previous_depth.shape != current_depth.shape or previous_depth.shape[0] != 15:
        raise ValueError("Teacher overlap depth must have matching [15,H,W] shapes")
    frame_ratios: List[float] = []
    for frame_index in range(15):
        previous = previous_depth[frame_index]
        current = current_depth[frame_index]
        valid = (
            previous_valid[frame_index].astype(bool)
            & current_valid[frame_index].astype(bool)
            & np.isfinite(previous)
            & np.isfinite(current)
            & (previous > eps)
            & (current > eps)
        )
        if previous_confidence is not None and current_confidence is not None:
            valid &= previous_confidence[frame_index] >= confidence_threshold
            valid &= current_confidence[frame_index] >= confidence_threshold
        if previous_highlight is not None and current_highlight is not None:
            valid &= ~previous_highlight[frame_index].astype(bool)
            valid &= ~current_highlight[frame_index].astype(bool)
        if int(valid.sum()) < minimum_valid_pixels:
            continue
        ratio = np.median(previous[valid] / current[valid])
        if np.isfinite(ratio) and ratio > eps:
            frame_ratios.append(float(ratio))
    if not frame_ratios:
        raise RuntimeError("No overlap frame had enough valid pixels for teacher scale audit")
    return float(np.median(np.asarray(frame_ratios))), frame_ratios


def _atomic_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    temporary.replace(path)


def _load_raw(
    path: Path,
    metadata: Dict[str, Any],
    shape: Tuple[int, int],
    base_checkpoint: str,
) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError("Raw cross-clip cache missing: {}".format(path))
    with np.load(str(path), allow_pickle=False) as cache:
        validate_crossclip_teacher_cache(
            cache, metadata, shape, base_checkpoint, "raw"
        )
        return {key: cache[key].copy() for key in cache.files}


def _aligned_arrays(
    raw: Dict[str, np.ndarray],
    scale: float,
    reference_start: int,
    frame_ratios: Sequence[float],
) -> Dict[str, np.ndarray]:
    arrays = {key: value.copy() for key, value in raw.items()}
    for key in ("depth", "xyz_local", "xyz_global"):
        arrays[key] = (arrays[key].astype(np.float32) * scale).astype(np.float32)
    extrinsics = arrays["extrinsics"].astype(np.float32).copy()
    # For X_cam = R X_world + t, scaling both point gauges requires t' = s*t.
    extrinsics[:, :3, 3] *= scale
    arrays["extrinsics"] = extrinsics
    arrays["cache_stage"] = np.asarray("aligned", dtype=np.str_)
    arrays["alignment_scale"] = np.asarray(scale, dtype=np.float32)
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.update(
        {
            "cache_stage": "aligned",
            "alignment_scale": scale,
            "alignment_reference_clip_start": reference_start,
            "overlap_frame_scale_ratios": list(frame_ratios),
        }
    )
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False), dtype=np.str_
    )
    return arrays


def align_crossclip_teacher_cache(
    config_path: Path,
    split: str,
    audit_only: bool = False,
    overwrite: bool = False,
    raw_root_override: Optional[Path] = None,
    aligned_root_override: Optional[Path] = None,
    report_override: Optional[Path] = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    teacher_config = dict(config["teacher"])
    alignment = dict(teacher_config.get("scale_alignment", {}))
    dataset = make_crossclip_rgb_dataset(config["dataset"], split)
    raw_root = (
        Path(raw_root_override)
        if raw_root_override is not None
        else Path(str(teacher_config["raw_cache_root"]))
    ) / split
    aligned_root = (
        Path(aligned_root_override)
        if aligned_root_override is not None
        else Path(str(teacher_config["aligned_cache_root"]))
    ) / split
    if raw_root.resolve() == aligned_root.resolve():
        raise ValueError("Raw and aligned cache roots must be different")
    shape = (
        int(config["dataset"]["image_height"]),
        int(config["dataset"]["image_width"]),
    )
    base_checkpoint = str(teacher_config["pretrained_checkpoint"])
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(dataset.clips):
        groups[str(record.sequence["sequence_id"])].append(index)

    report: Dict[str, Any] = {
        "split": split,
        "raw_cache_root": str(raw_root),
        "aligned_cache_root": str(aligned_root),
        "audit_only": bool(audit_only),
        "sequences": [],
    }
    for sequence_id, indices in groups.items():
        indices.sort(key=lambda value: int(dataset.clips[value].clip_start))
        previous_aligned: Optional[Dict[str, np.ndarray]] = None
        previous_start: Optional[int] = None
        sequence_records = []
        for index in indices:
            metadata = clip_metadata(dataset, index)
            start = int(metadata["clip_start"])
            raw_path = crossclip_teacher_cache_path(raw_root, metadata)
            raw = _load_raw(raw_path, metadata, shape, base_checkpoint)
            if previous_aligned is None:
                scale, frame_ratios, reference_start = 1.0, [], start
            else:
                if start != int(previous_start) + 1:
                    raise RuntimeError(
                        "Stride-one cache sequence has a clip-start gap: {} -> {}".format(
                            previous_start, start
                        )
                    )
                scale, frame_ratios = estimate_teacher_overlap_scale(
                    previous_aligned["depth"][1:16],
                    raw["depth"][0:15],
                    previous_aligned["valid_mask"][1:16],
                    raw["valid_mask"][0:15],
                    previous_aligned["confidence"][1:16],
                    raw["confidence"][0:15],
                    previous_aligned["highlight_mask"][1:16]
                    if bool(alignment.get("ignore_highlight", True))
                    else None,
                    raw["highlight_mask"][0:15]
                    if bool(alignment.get("ignore_highlight", True))
                    else None,
                    confidence_threshold=float(alignment.get("confidence_threshold", 0.0)),
                    minimum_valid_pixels=int(alignment.get("minimum_valid_pixels", 64)),
                    eps=float(alignment.get("eps", 1e-6)),
                )
                reference_start = int(previous_start)
            aligned = _aligned_arrays(raw, scale, reference_start, frame_ratios)
            output_path = crossclip_teacher_cache_path(aligned_root, metadata)
            if not audit_only:
                if output_path.is_file() and not overwrite:
                    with np.load(str(output_path), allow_pickle=False) as existing:
                        validate_crossclip_teacher_cache(
                            existing, metadata, shape, base_checkpoint, "aligned"
                        )
                else:
                    _atomic_npz(output_path, aligned)
            sequence_records.append(
                {
                    "clip_start": start,
                    "reference_clip_start": reference_start,
                    "alignment_scale": scale,
                    "overlap_frame_ratios": frame_ratios,
                    "raw_cache": str(raw_path),
                    "aligned_cache": None if audit_only else str(output_path),
                }
            )
            previous_aligned = aligned
            previous_start = start
        report["sequences"].append(
            {"sequence_id": sequence_id, "clips": sequence_records}
        )
    report_path = report_override
    if report_path is None:
        report_value = alignment.get("report")
        if report_value:
            report_path = Path(str(report_value).format(split=split))
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report


__all__ = ["align_crossclip_teacher_cache", "estimate_teacher_overlap_scale"]
