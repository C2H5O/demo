"""Offline scale alignment for overlapping raw VGGT-Omega teacher clips."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ALIGNMENT_FORMAT_VERSION = "teacher-clip-alignment-v1"
ALIGNMENT_METHOD = "adjacent_overlap_median_scale"
ALIGNMENT_ANCHOR = "first_clip"


@dataclass(frozen=True)
class RawTeacherClip:
    path: Path
    relative_path: str
    sequence_id: str
    clip_start: int
    absolute_frame_ids: Tuple[int, ...]


def _positive_finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("{} must be positive and finite; got {}".format(name, value))
    return result


def _required_scalar(cache: "np.lib.npyio.NpzFile", key: str, path: Path) -> Any:
    if key not in cache:
        raise RuntimeError("Raw teacher cache {} is missing {!r}".format(path, key))
    try:
        return cache[key].item()
    except ValueError as error:
        raise RuntimeError("Raw teacher cache {} field {!r} must be scalar".format(path, key)) from error


def discover_raw_teacher_clips(cache_root: Union[str, Path]) -> Dict[str, list[RawTeacherClip]]:
    """Index raw caches by their exact sequence ID and fail on ambiguous identities."""
    root = Path(cache_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Raw teacher cache root is missing: {}".format(root))
    paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise RuntimeError("No raw teacher cache NPZ files found under {}".format(root))
    grouped: Dict[str, list[RawTeacherClip]] = {}
    identities: set[tuple[str, int]] = set()
    for path in paths:
        with np.load(str(path), allow_pickle=False) as cache:
            sequence_id = str(_required_scalar(cache, "sequence_id", path))
            clip_start = int(_required_scalar(cache, "clip_start", path))
            stage = str(_required_scalar(cache, "cache_stage", path))
            raw_scale = _positive_finite(
                _required_scalar(cache, "alignment_scale", path),
                "raw cache alignment_scale",
            )
            if stage != "raw":
                raise RuntimeError("Alignment input must be raw, but {} has cache_stage={!r}".format(path, stage))
            if raw_scale != 1.0:
                raise RuntimeError(
                    "Raw teacher cache {} already declares alignment_scale={}; expected 1.0"
                    .format(path, raw_scale)
                )
            if "absolute_frame_ids" not in cache:
                raise RuntimeError("Raw teacher cache {} is missing absolute_frame_ids".format(path))
            absolute_ids = tuple(int(value) for value in cache["absolute_frame_ids"].tolist())
            if len(absolute_ids) != 16 or len(set(absolute_ids)) != 16:
                raise RuntimeError("Raw teacher cache {} must contain 16 unique absolute frame IDs".format(path))
            if clip_start < 0:
                raise RuntimeError("Raw teacher cache {} has negative clip_start".format(path))
            if clip_start % 8:
                # A shared cache root can contain dense exploratory windows. Baseline-I
                # consumes only the legal stride-eight training grid, matching the loader.
                continue
            for key in ("depth", "valid_mask"):
                if key not in cache:
                    raise RuntimeError("Raw teacher cache {} is missing {!r}".format(path, key))
            if tuple(cache["depth"].shape[:1]) != (16,) or cache["depth"].ndim != 3:
                raise RuntimeError("Raw teacher cache {} depth must have shape [16,H,W]".format(path))
            if tuple(cache["valid_mask"].shape) != tuple(cache["depth"].shape):
                raise RuntimeError("Raw teacher cache {} valid mask shape differs from depth".format(path))
            if cache["valid_mask"].dtype != np.bool_:
                raise RuntimeError("Raw teacher cache {} valid_mask must be boolean".format(path))
        identity = (sequence_id, clip_start)
        if identity in identities:
            raise RuntimeError("Duplicate raw teacher clip identity {}".format(identity))
        identities.add(identity)
        grouped.setdefault(sequence_id, []).append(
            RawTeacherClip(
                path=path,
                relative_path=path.relative_to(root).as_posix(),
                sequence_id=sequence_id,
                clip_start=clip_start,
                absolute_frame_ids=absolute_ids,
            )
        )
    for sequence_id, clips in grouped.items():
        clips.sort(key=lambda item: item.clip_start)
        starts = [item.clip_start for item in clips]
        if starts != sorted(starts):
            raise RuntimeError("Teacher clips are not sorted for sequence {}".format(sequence_id))
    if not grouped:
        raise RuntimeError("No legal stride-eight raw teacher clips found under {}".format(root))
    return grouped


def compute_overlap_alignment(
    previous_depth: np.ndarray,
    previous_valid: np.ndarray,
    previous_absolute_frame_ids: Sequence[int],
    current_depth: np.ndarray,
    current_valid: np.ndarray,
    current_absolute_frame_ids: Sequence[int],
    *,
    expected_overlap_frames: int = 8,
    eps: float = 1e-6,
    minimum_valid_pixels_per_frame: int = 256,
) -> Dict[str, Any]:
    """Estimate current-to-previous scale with a median of frame medians."""
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be positive and finite")
    if expected_overlap_frames <= 0:
        raise ValueError("expected_overlap_frames must be positive")
    if minimum_valid_pixels_per_frame <= 0:
        raise ValueError("minimum_valid_pixels_per_frame must be positive")
    previous_ids = tuple(int(value) for value in previous_absolute_frame_ids)
    current_ids = tuple(int(value) for value in current_absolute_frame_ids)
    if len(previous_ids) != previous_depth.shape[0] or len(current_ids) != current_depth.shape[0]:
        raise RuntimeError("Depth frame count and absolute_frame_ids length disagree")
    if len(set(previous_ids)) != len(previous_ids) or len(set(current_ids)) != len(current_ids):
        raise RuntimeError("absolute_frame_ids must be unique within each clip")
    if previous_depth.shape[1:] != current_depth.shape[1:]:
        raise RuntimeError(
            "Overlapping teacher clips have different spatial shapes: {} vs {}"
            .format(previous_depth.shape[1:], current_depth.shape[1:])
        )
    if previous_valid.shape != previous_depth.shape or current_valid.shape != current_depth.shape:
        raise RuntimeError("Teacher valid masks must match their depth arrays")
    previous_index = {frame_id: index for index, frame_id in enumerate(previous_ids)}
    current_index = {frame_id: index for index, frame_id in enumerate(current_ids)}
    overlap_ids = sorted(set(previous_index) & set(current_index))
    if len(overlap_ids) != expected_overlap_frames:
        raise RuntimeError(
            "Expected {} overlapping absolute frame IDs, found {}: {}"
            .format(expected_overlap_frames, len(overlap_ids), overlap_ids)
        )

    frame_data: list[tuple[int, int, float, np.ndarray, np.ndarray]] = []
    for frame_id in overlap_ids:
        previous = np.asarray(previous_depth[previous_index[frame_id]], dtype=np.float64)
        current = np.asarray(current_depth[current_index[frame_id]], dtype=np.float64)
        valid = (
            np.asarray(previous_valid[previous_index[frame_id]], dtype=np.bool_)
            & np.asarray(current_valid[current_index[frame_id]], dtype=np.bool_)
            & np.isfinite(previous)
            & np.isfinite(current)
            & (previous > eps)
            & (current > eps)
        )
        valid_count = int(valid.sum())
        if valid_count < minimum_valid_pixels_per_frame:
            raise RuntimeError(
                "Overlap frame {} has {} valid pixels; at least {} are required"
                .format(frame_id, valid_count, minimum_valid_pixels_per_frame)
            )
        previous_values = previous[valid]
        current_values = current[valid]
        frame_scale = _positive_finite(
            np.median(previous_values / current_values),
            "overlap frame {} scale".format(frame_id),
        )
        frame_data.append(
            (frame_id, valid_count, frame_scale, previous_values, current_values)
        )
    relative_scale = _positive_finite(
        np.median([item[2] for item in frame_data]),
        "current-to-previous relative scale",
    )
    per_frame = []
    for frame_id, valid_count, frame_scale, previous_values, current_values in frame_data:
        raw_absrel = float(np.mean(np.abs(previous_values - current_values) / previous_values))
        aligned_absrel = float(
            np.mean(np.abs(previous_values - relative_scale * current_values) / previous_values)
        )
        per_frame.append(
            {
                "absolute_frame_id": frame_id,
                "scale_current_to_previous": frame_scale,
                "valid_pixel_count": valid_count,
                "raw_absrel": raw_absrel,
                "aligned_absrel": aligned_absrel,
            }
        )
    return {
        "overlap_frame_ids": overlap_ids,
        "valid_overlap_pixel_count": int(sum(item[1] for item in frame_data)),
        "relative_scale_to_previous": relative_scale,
        "raw_overlap_absrel": float(np.mean([item["raw_absrel"] for item in per_frame])),
        "aligned_overlap_absrel": float(
            np.mean([item["aligned_absrel"] for item in per_frame])
        ),
        "per_frame_scales": per_frame,
    }


def _load_pair_alignment(
    previous: RawTeacherClip,
    current: RawTeacherClip,
    *,
    expected_overlap_frames: int,
    eps: float,
    minimum_valid_pixels_per_frame: int,
) -> Dict[str, Any]:
    if previous.sequence_id != current.sequence_id:
        raise RuntimeError("Teacher alignment cannot cross sequence IDs")
    if current.clip_start - previous.clip_start != expected_overlap_frames:
        raise RuntimeError(
            "Adjacent clips in sequence {} must differ by {} starts; got {} -> {}"
            .format(
                current.sequence_id,
                expected_overlap_frames,
                previous.clip_start,
                current.clip_start,
            )
        )
    with np.load(str(previous.path), allow_pickle=False) as previous_cache, np.load(
        str(current.path), allow_pickle=False
    ) as current_cache:
        for cache, clip in ((previous_cache, previous), (current_cache, current)):
            for key in ("depth", "valid_mask"):
                if key not in cache:
                    raise RuntimeError("Raw teacher cache {} is missing {!r}".format(clip.path, key))
            if str(_required_scalar(cache, "sequence_id", clip.path)) != clip.sequence_id:
                raise RuntimeError("Raw teacher sequence_id changed after discovery: {}".format(clip.path))
            if int(_required_scalar(cache, "clip_start", clip.path)) != clip.clip_start:
                raise RuntimeError("Raw teacher clip_start changed after discovery: {}".format(clip.path))
        return compute_overlap_alignment(
            previous_cache["depth"],
            previous_cache["valid_mask"],
            previous.absolute_frame_ids,
            current_cache["depth"],
            current_cache["valid_mask"],
            current.absolute_frame_ids,
            expected_overlap_frames=expected_overlap_frames,
            eps=eps,
            minimum_valid_pixels_per_frame=minimum_valid_pixels_per_frame,
        )


def _mean(values: Iterable[float]) -> Optional[float]:
    items = [float(value) for value in values]
    return float(np.mean(items)) if items else None


def _sequence_summary(clip_records: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = [record for record in clip_records.values() if record["previous_clip_start"] is not None]
    relative = [float(record["relative_scale_to_previous"]) for record in pairs]
    raw = [float(record["raw_overlap_absrel"]) for record in pairs]
    aligned = [float(record["aligned_overlap_absrel"]) for record in pairs]
    mean_raw = _mean(raw)
    mean_aligned = _mean(aligned)
    improvement = (
        100.0 * (mean_raw - mean_aligned) / mean_raw
        if mean_raw is not None and mean_raw > 0.0 and mean_aligned is not None
        else None
    )
    return {
        "clip_count": len(clip_records),
        "pair_count": len(pairs),
        "mean_relative_scale": _mean(relative),
        "std_relative_scale": float(np.std(relative)) if relative else None,
        "min_relative_scale": min(relative) if relative else None,
        "max_relative_scale": max(relative) if relative else None,
        "mean_raw_overlap_absrel": mean_raw,
        "mean_aligned_overlap_absrel": mean_aligned,
        "improvement_percentage": improvement,
    }


def _scale_warnings(relative_scale: float, cumulative_scale: float) -> list[str]:
    messages = []
    for name, value in (
        ("relative_scale_to_previous", relative_scale),
        ("alignment_scale", cumulative_scale),
    ):
        if value < 0.5 or value > 2.0:
            messages.append("{}={} is outside diagnostic range [0.5, 2.0]".format(name, value))
    return messages


def build_teacher_clip_alignment(
    cache_root: Union[str, Path],
    *,
    expected_overlap_frames: int = 8,
    eps: float = 1e-6,
    minimum_valid_pixels_per_frame: int = 256,
) -> Dict[str, Any]:
    """Compute fixed sequence-gauge scales without modifying any cache file."""
    if expected_overlap_frames != 8:
        raise ValueError("Current 16-frame, stride-8 protocol requires exactly 8 overlap frames")
    grouped = discover_raw_teacher_clips(cache_root)
    sequences: Dict[str, Any] = {}
    for sequence_id in sorted(grouped):
        clips = grouped[sequence_id]
        anchor_start = min(item.clip_start for item in clips)
        records: Dict[str, Any] = {}
        cumulative_scale = 1.0
        for index, clip in enumerate(clips):
            if index == 0:
                record = {
                    "clip_start": clip.clip_start,
                    "absolute_frame_ids": list(clip.absolute_frame_ids),
                    "cache_relative_path": clip.relative_path,
                    "previous_clip_start": None,
                    "overlap_frame_ids": [],
                    "valid_overlap_pixel_count": 0,
                    "relative_scale_to_previous": None,
                    "alignment_scale": 1.0,
                    "raw_overlap_absrel": None,
                    "aligned_overlap_absrel": None,
                    "per_frame_scales": [],
                    "warnings": [],
                }
            else:
                previous = clips[index - 1]
                pair = _load_pair_alignment(
                    previous,
                    clip,
                    expected_overlap_frames=expected_overlap_frames,
                    eps=eps,
                    minimum_valid_pixels_per_frame=minimum_valid_pixels_per_frame,
                )
                cumulative_scale = _positive_finite(
                    cumulative_scale * float(pair["relative_scale_to_previous"]),
                    "cumulative alignment scale",
                )
                record = {
                    "clip_start": clip.clip_start,
                    "absolute_frame_ids": list(clip.absolute_frame_ids),
                    "cache_relative_path": clip.relative_path,
                    "previous_clip_start": previous.clip_start,
                    **pair,
                    "alignment_scale": cumulative_scale,
                    "warnings": _scale_warnings(
                        float(pair["relative_scale_to_previous"]), cumulative_scale
                    ),
                }
            records[str(clip.clip_start)] = record
        sequences[sequence_id] = {
            "sequence_id": sequence_id,
            "anchor_clip_start": anchor_start,
            "method": ALIGNMENT_METHOD,
            "anchor": ALIGNMENT_ANCHOR,
            "clip_length": 16,
            "window_stride": 8,
            "clips": records,
            "summary": _sequence_summary(records),
        }
    summaries = [value["summary"] for value in sequences.values()]
    mean_raw = _mean(
        value["mean_raw_overlap_absrel"]
        for value in summaries
        if value["mean_raw_overlap_absrel"] is not None
    )
    mean_aligned = _mean(
        value["mean_aligned_overlap_absrel"]
        for value in summaries
        if value["mean_aligned_overlap_absrel"] is not None
    )
    macro_improvement = (
        100.0 * (mean_raw - mean_aligned) / mean_raw
        if mean_raw is not None and mean_raw > 0.0 and mean_aligned is not None
        else None
    )
    macro_relative = [
        float(value["mean_relative_scale"])
        for value in summaries
        if value["mean_relative_scale"] is not None
    ]
    return {
        "format_version": ALIGNMENT_FORMAT_VERSION,
        "method": ALIGNMENT_METHOD,
        "anchor": ALIGNMENT_ANCHOR,
        "clip_length": 16,
        "window_stride": 8,
        "expected_overlap_frames": expected_overlap_frames,
        "eps": eps,
        "minimum_valid_pixels_per_frame": minimum_valid_pixels_per_frame,
        "extreme_scale_warning_range": [0.5, 2.0],
        "source_cache_root": str(Path(cache_root).expanduser().resolve()),
        "sequences": sequences,
        "macro_summary": {
            "sequence_count": len(sequences),
            "clip_count": sum(value["clip_count"] for value in summaries),
            "pair_count": sum(value["pair_count"] for value in summaries),
            "mean_relative_scale": _mean(
                macro_relative
            ),
            "std_relative_scale": float(np.std(macro_relative)) if macro_relative else None,
            "min_relative_scale": min(macro_relative) if macro_relative else None,
            "max_relative_scale": max(macro_relative) if macro_relative else None,
            "mean_raw_overlap_absrel": mean_raw,
            "mean_aligned_overlap_absrel": mean_aligned,
            "improvement_percentage": macro_improvement,
        },
    }


def write_teacher_clip_alignment(metadata: Mapping[str, Any], output_path: Union[str, Path]) -> Path:
    """Atomically write deterministic, human-auditable JSON metadata."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)
    return path


def _require_close(actual: Any, expected: Any, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise RuntimeError("Alignment audit mismatch for {}: {} != {}".format(label, actual, expected))
        return
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError("Alignment audit mismatch for {}: {} != {}".format(label, actual, expected))


def audit_teacher_clip_alignment(
    cache_root: Union[str, Path], metadata_path: Union[str, Path]
) -> Dict[str, Any]:
    """Recompute alignment from raw caches and verify every saved scale and audit metric."""
    saved = TeacherClipAlignmentIndex.from_json(metadata_path).metadata
    requested_root = Path(cache_root).expanduser().resolve()
    if Path(str(saved["source_cache_root"])).expanduser().resolve() != requested_root:
        raise RuntimeError("Alignment audit cache root differs from metadata source_cache_root")
    rebuilt = build_teacher_clip_alignment(
        cache_root,
        expected_overlap_frames=int(saved["expected_overlap_frames"]),
        eps=float(saved["eps"]),
        minimum_valid_pixels_per_frame=int(saved["minimum_valid_pixels_per_frame"]),
    )
    if set(saved["sequences"]) != set(rebuilt["sequences"]):
        raise RuntimeError("Alignment audit sequence set differs from raw caches")
    for sequence_id, saved_sequence in saved["sequences"].items():
        rebuilt_sequence = rebuilt["sequences"][sequence_id]
        if set(saved_sequence["clips"]) != set(rebuilt_sequence["clips"]):
            raise RuntimeError("Alignment audit clip set differs for {}".format(sequence_id))
        for start, saved_clip in saved_sequence["clips"].items():
            rebuilt_clip = rebuilt_sequence["clips"][start]
            for key in (
                "clip_start",
                "absolute_frame_ids",
                "previous_clip_start",
                "overlap_frame_ids",
                "valid_overlap_pixel_count",
                "cache_relative_path",
            ):
                if saved_clip[key] != rebuilt_clip[key]:
                    raise RuntimeError(
                        "Alignment audit mismatch for {}/{}/{}".format(sequence_id, start, key)
                    )
            for key in (
                "relative_scale_to_previous",
                "alignment_scale",
                "raw_overlap_absrel",
                "aligned_overlap_absrel",
            ):
                _require_close(
                    saved_clip[key], rebuilt_clip[key], "{}/{}/{}".format(sequence_id, start, key)
                )
            if saved_clip["per_frame_scales"] != rebuilt_clip["per_frame_scales"]:
                raise RuntimeError(
                    "Alignment audit per-frame details differ for {}/{}".format(sequence_id, start)
                )
        if saved_sequence.get("summary") != rebuilt_sequence["summary"]:
            raise RuntimeError("Alignment audit summary differs for {}".format(sequence_id))
    if saved.get("macro_summary") != rebuilt["macro_summary"]:
        raise RuntimeError("Alignment audit macro summary differs")
    return rebuilt


def print_alignment_audit(metadata: Mapping[str, Any]) -> None:
    """Print pair, per-sequence, and dataset macro diagnostics."""
    for sequence_id, sequence in metadata["sequences"].items():
        for record in sequence["clips"].values():
            if record["previous_clip_start"] is None:
                continue
            print(
                "alignment pair sequence={} prev_start={} cur_start={} relative_scale={:.9g} "
                "cumulative_scale={:.9g} raw_overlap_absrel={:.6%} aligned_overlap_absrel={:.6%}"
                .format(
                    sequence_id,
                    record["previous_clip_start"],
                    record["clip_start"],
                    record["relative_scale_to_previous"],
                    record["alignment_scale"],
                    record["raw_overlap_absrel"],
                    record["aligned_overlap_absrel"],
                )
            )
            for message in record["warnings"]:
                warnings.warn("{} start {}: {}".format(sequence_id, record["clip_start"], message))
        summary = sequence["summary"]
        print(
            "alignment sequence={} clips={} pairs={} relative_mean={} relative_std={} "
            "relative_min={} relative_max={} raw_absrel_mean={} aligned_absrel_mean={} "
            "improvement_percentage={}".format(
                sequence_id,
                summary["clip_count"],
                summary["pair_count"],
                summary["mean_relative_scale"],
                summary["std_relative_scale"],
                summary["min_relative_scale"],
                summary["max_relative_scale"],
                summary["mean_raw_overlap_absrel"],
                summary["mean_aligned_overlap_absrel"],
                summary["improvement_percentage"],
            )
        )
    print("alignment macro summary: {}".format(json.dumps(metadata["macro_summary"], sort_keys=True)))


class TeacherClipAlignmentIndex:
    """Validated lookup table keyed only by sequence_id and clip_start."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self.metadata = dict(metadata)
        self._validate()

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TeacherClipAlignmentIndex":
        metadata_path = Path(path).expanduser().resolve()
        if not metadata_path.is_file():
            raise FileNotFoundError("Teacher clip alignment metadata is missing: {}".format(metadata_path))
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except json.JSONDecodeError as error:
            raise RuntimeError("Teacher clip alignment metadata is invalid JSON: {}".format(metadata_path)) from error
        index = cls(metadata)
        index.path = metadata_path
        return index

    def _validate(self) -> None:
        required = {
            "format_version",
            "method",
            "anchor",
            "clip_length",
            "window_stride",
            "expected_overlap_frames",
            "eps",
            "minimum_valid_pixels_per_frame",
            "source_cache_root",
            "sequences",
        }
        missing = sorted(required - set(self.metadata))
        if missing:
            raise RuntimeError("Teacher clip alignment metadata is missing {}".format(missing))
        if self.metadata["format_version"] != ALIGNMENT_FORMAT_VERSION:
            raise RuntimeError("Unsupported teacher clip alignment format")
        if self.metadata["method"] != ALIGNMENT_METHOD or self.metadata["anchor"] != ALIGNMENT_ANCHOR:
            raise RuntimeError("Teacher clip alignment method/anchor mismatch")
        if (
            int(self.metadata["clip_length"]),
            int(self.metadata["window_stride"]),
            int(self.metadata["expected_overlap_frames"]),
        ) != (16, 8, 8):
            raise RuntimeError("Teacher clip alignment metadata violates the 16/8/8 protocol")
        _positive_finite(self.metadata["eps"], "alignment metadata eps")
        if int(self.metadata["minimum_valid_pixels_per_frame"]) <= 0:
            raise RuntimeError("Alignment minimum valid pixels must be positive")
        sequences = self.metadata["sequences"]
        if not isinstance(sequences, dict) or not sequences:
            raise RuntimeError("Teacher clip alignment metadata contains no sequences")
        for sequence_id, sequence in sequences.items():
            if sequence.get("sequence_id") != sequence_id:
                raise RuntimeError("Teacher alignment sequence key and sequence_id differ")
            if sequence.get("method") != ALIGNMENT_METHOD or sequence.get("anchor") != ALIGNMENT_ANCHOR:
                raise RuntimeError("Teacher alignment sequence method/anchor mismatch")
            clips = sequence.get("clips")
            if not isinstance(clips, dict) or not clips:
                raise RuntimeError("Teacher alignment sequence {} has no clips".format(sequence_id))
            starts = sorted(int(value) for value in clips)
            if int(sequence.get("anchor_clip_start")) != starts[0]:
                raise RuntimeError("Teacher alignment anchor is not the first clip")
            previous_scale = 1.0
            previous_record = None
            for index, start in enumerate(starts):
                record = clips[str(start)]
                required_clip = {
                    "clip_start",
                    "absolute_frame_ids",
                    "relative_scale_to_previous",
                    "alignment_scale",
                    "previous_clip_start",
                    "overlap_frame_ids",
                    "valid_overlap_pixel_count",
                    "raw_overlap_absrel",
                    "aligned_overlap_absrel",
                    "per_frame_scales",
                    "cache_relative_path",
                }
                missing_clip = sorted(required_clip - set(record))
                if missing_clip:
                    raise RuntimeError(
                        "Teacher alignment clip {}/{} is missing {}".format(sequence_id, start, missing_clip)
                    )
                if int(record["clip_start"]) != start:
                    raise RuntimeError("Teacher alignment clip key/start mismatch")
                ids = [int(value) for value in record["absolute_frame_ids"]]
                if len(ids) != 16 or len(set(ids)) != 16:
                    raise RuntimeError("Teacher alignment clip must have 16 unique absolute IDs")
                scale = _positive_finite(record["alignment_scale"], "alignment_scale")
                if index == 0:
                    if record["previous_clip_start"] is not None or record["relative_scale_to_previous"] is not None:
                        raise RuntimeError("Teacher alignment anchor cannot have a previous clip")
                    if scale != 1.0:
                        raise RuntimeError("Teacher alignment anchor scale must be 1.0")
                else:
                    if start - starts[index - 1] != 8 or int(record["previous_clip_start"]) != starts[index - 1]:
                        raise RuntimeError("Teacher alignment chain is not contiguous at {}/{}".format(sequence_id, start))
                    relative = _positive_finite(
                        record["relative_scale_to_previous"], "relative_scale_to_previous"
                    )
                    if not math.isclose(scale, previous_scale * relative, rel_tol=1e-10, abs_tol=1e-12):
                        raise RuntimeError("Teacher alignment cumulative scale is inconsistent")
                    overlap = sorted(set(ids) & set(previous_record["absolute_frame_ids"]))
                    if overlap != record["overlap_frame_ids"] or len(overlap) != 8:
                        raise RuntimeError("Teacher alignment overlap IDs are inconsistent")
                    if int(record["valid_overlap_pixel_count"]) <= 0:
                        raise RuntimeError("Teacher alignment pair has no valid overlap pixels")
                previous_scale = scale
                previous_record = record

    def lookup(
        self, sequence_id: str, clip_start: int, absolute_frame_ids: Sequence[int]
    ) -> Dict[str, Any]:
        try:
            record = self.metadata["sequences"][str(sequence_id)]["clips"][str(int(clip_start))]
        except KeyError as error:
            raise RuntimeError(
                "No teacher alignment entry for sequence={!r} clip_start={}"
                .format(sequence_id, clip_start)
            ) from error
        actual_ids = [int(value) for value in absolute_frame_ids]
        if actual_ids != [int(value) for value in record["absolute_frame_ids"]]:
            raise RuntimeError(
                "Teacher alignment absolute_frame_ids mismatch for sequence={!r} clip_start={}"
                .format(sequence_id, clip_start)
            )
        return dict(record)


__all__ = [
    "ALIGNMENT_ANCHOR",
    "ALIGNMENT_FORMAT_VERSION",
    "ALIGNMENT_METHOD",
    "RawTeacherClip",
    "TeacherClipAlignmentIndex",
    "audit_teacher_clip_alignment",
    "build_teacher_clip_alignment",
    "compute_overlap_alignment",
    "discover_raw_teacher_clips",
    "print_alignment_audit",
    "write_teacher_clip_alignment",
]
