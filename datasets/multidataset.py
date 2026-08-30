"""Canonical processed-dataset discovery and separate student/teacher RGB reads.

The canonical preprocessing contract keeps a student-grid PNG and a teacher-grid
PNG for the same source frame.  This module intentionally leaves legacy SCARED
discovery untouched while exposing the same clip-shaped interface to the
cross-clip code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from datasets.highlight import HighlightDetectionConfig, SpecularHighlightProcessor
from datasets.scared_discovery import (
    expected_dataset_ids,
    extract_dataset_id,
    natural_sort_key,
)
from datasets.scared_dataset import ClipRecord, _validate_temporal_config
from datasets.transforms import (
    load_precomputed_student_rgb_tensor,
    load_teacher_rgb_tensor,
    unnormalize_image,
)


CANONICAL_TRAIN_DATASETS = ("C3VD", "StereoMIS", "AutoLaparo", "EndoVis18")
CANONICAL_EVALUATION_DATASETS = ("Hamlyn",)
CANONICAL_STUDENT_SIZE = (448, 560)
CANONICAL_TEACHER_SIZE = (1024, 1280)


def _frame_entries(metadata: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """Accept the two documented metadata spellings without inventing IDs."""
    entries = metadata.get("frames", metadata.get("frame_mapping"))
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("canonical metadata must contain a non-empty frames list")
    result: Dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeError("canonical frame metadata entry must be an object")
        student = entry.get("student_rgb_file")
        teacher = entry.get("teacher_rgb_file")
        if not student or not teacher:
            raise RuntimeError("canonical frame mapping needs student_rgb_file and teacher_rgb_file")
        name = Path(str(student)).name
        if name in result:
            raise RuntimeError("duplicate canonical student frame name {}".format(name))
        result[name] = entry
    return result


def _require_matching_pngs(sequence_root: Path, entries: Mapping[str, Mapping[str, Any]]) -> Tuple[List[Path], List[Path], List[int]]:
    student_root, teacher_root = sequence_root / "student_rgb", sequence_root / "teacher_rgb"
    if not student_root.is_dir() or not teacher_root.is_dir():
        raise RuntimeError("canonical sequence {} needs teacher_rgb and student_rgb".format(sequence_root))
    student_paths = sorted(student_root.glob("*.png"), key=lambda path: path.name)
    teacher_paths = sorted(teacher_root.glob("*.png"), key=lambda path: path.name)
    if not student_paths or [path.name for path in student_paths] != [path.name for path in teacher_paths]:
        raise RuntimeError("teacher/student PNG names differ in {}".format(sequence_root))
    if {path.name for path in student_paths} != set(entries):
        raise RuntimeError("metadata frame mapping differs from PNG names in {}".format(sequence_root))
    ids: List[int] = []
    for student, teacher in zip(student_paths, teacher_paths):
        entry = entries[student.name]
        if Path(str(entry["teacher_rgb_file"])).name != teacher.name:
            raise RuntimeError("teacher mapping does not match {}".format(student.name))
        value = entry.get("source_frame_id", entry.get("absolute_frame_id", entry.get("processed_index")))
        if value is None:
            raise RuntimeError("canonical frame mapping has no source_frame_id")
        ids.append(int(value))
    if len(set(ids)) != len(ids):
        raise RuntimeError("canonical sequence has duplicate absolute frame IDs")
    if any(right != left + 1 for left, right in zip(ids, ids[1:])):
        raise RuntimeError("canonical sequence has a source-frame gap; preprocess it into separate subsequences")
    return student_paths, teacher_paths, ids


def discover_canonical_sequences(root: str | Path, split: str) -> List[Dict[str, Any]]:
    """Discover complete processed sequences, excluding Hamlyn from training."""
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    wanted = CANONICAL_TRAIN_DATASETS if split.lower() == "train" else CANONICAL_EVALUATION_DATASETS
    sequences: List[Dict[str, Any]] = []
    for dataset_name in wanted:
        dataset_root = root / dataset_name
        if not dataset_root.is_dir():
            continue
        for sequence_root in sorted((path for path in dataset_root.iterdir() if path.is_dir()), key=lambda path: path.name):
            complete = sequence_root / "_preprocess_complete.json"
            metadata_path = sequence_root / "metadata.json"
            if not complete.is_file() or not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                marker = json.loads(complete.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError("invalid canonical metadata in {}".format(sequence_root)) from error
            if dataset_name == "Hamlyn" and not bool(metadata.get("evaluation_only", marker.get("evaluation_only", False))):
                raise RuntimeError("Hamlyn sequence must declare evaluation_only=true: {}".format(sequence_root))
            if dataset_name != "Hamlyn" and bool(metadata.get("evaluation_only", False)):
                continue
            entries = _frame_entries(metadata)
            student, teacher, ids = _require_matching_pngs(sequence_root, entries)
            if len(student) < 16:
                continue
            sequence_id = str(metadata.get("sequence_id", sequence_root.name))
            sequences.append({
                "dataset_name": dataset_name,
                "dataset_id": -1,
                "keyframe_id": sequence_root.name,
                "sequence_id": "{}/{}".format(dataset_name, sequence_id),
                "sequence_length": len(student),
                "frame_paths": [str(path) for path in student],
                "teacher_frame_paths": [str(path) for path in teacher],
                "absolute_frame_ids": ids,
                "frame_directory": str(student[0].parent),
                "keyframe_directory": str(sequence_root),
                "depth_directory": str(sequence_root / "data" / "depth") if (sequence_root / "data" / "depth").is_dir() else None,
                "preprocessing_identity": metadata.get("preprocessing_identity", metadata.get("preprocess_version", marker.get("version", "unknown"))),
                "canonical": True,
                "evaluation_only": dataset_name == "Hamlyn",
            })
    return sequences


def discover_processed_scared_sequences(root: str | Path, split: str) -> List[Dict[str, Any]]:
    """Discover processed SCARED keyframes with separate student/teacher RGB."""
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    wanted = set(expected_dataset_ids(split))
    sequences: List[Dict[str, Any]] = []
    dataset_directories = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        dataset_id = extract_dataset_id(candidate.name)
        if dataset_id is not None and dataset_id in wanted:
            dataset_directories.append((dataset_id, candidate))
    for dataset_id, dataset_root in sorted(dataset_directories):
        keyframes = sorted(
            (path for path in dataset_root.iterdir() if path.is_dir()),
            key=lambda path: natural_sort_key(path.name),
        )
        for sequence_root in keyframes:
            complete = sequence_root / "_preprocess_complete.json"
            metadata_path = sequence_root / "metadata.json"
            if not complete.is_file() or not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                marker = json.loads(complete.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "invalid processed SCARED metadata in {}".format(sequence_root)
                ) from error
            entries = _frame_entries(metadata)
            student, teacher, ids = _require_matching_pngs(sequence_root, entries)
            if len(student) < 16:
                continue
            sequence_id = str(
                metadata.get(
                    "sequence_id",
                    "dataset_{}/{}".format(dataset_id, sequence_root.name),
                )
            )
            sequences.append({
                "dataset_name": "SCARED",
                "dataset_id": dataset_id,
                "keyframe_id": sequence_root.name,
                "sequence_id": sequence_id,
                "sequence_length": len(student),
                "frame_paths": [str(path) for path in student],
                "teacher_frame_paths": [str(path) for path in teacher],
                "absolute_frame_ids": ids,
                "frame_directory": str(student[0].parent),
                "keyframe_directory": str(sequence_root),
                "depth_directory": None,
                "preprocessing_identity": metadata.get(
                    "preprocessing_identity",
                    metadata.get("preprocess_version", marker.get("version", "unknown")),
                ),
                "canonical": True,
                "evaluation_only": False,
            })
    return sequences


class CanonicalTemporalRGBDataset(Dataset):
    """Read already-materialized student PNGs without a second spatial resize."""

    def __init__(self, sequences: Sequence[Dict[str, Any]], *, clip_length: int, sample_stride: int, window_stride: int, normalize_mode: str, highlight: Dict[str, Any] | None = None) -> None:
        _validate_temporal_config(clip_length, sample_stride, window_stride)
        self.sequences = list(sequences)
        self.clip_length, self.sample_stride, self.window_stride = clip_length, sample_stride, window_stride
        self.image_height, self.image_width = CANONICAL_STUDENT_SIZE
        self.resize_mode, self.normalize_mode = "precomputed", normalize_mode
        options = dict(highlight or {})
        self.highlight_processor = None
        if bool(options.pop("enabled", False)):
            options["enabled"] = True
            self.highlight_processor = SpecularHighlightProcessor(HighlightDetectionConfig(**options))
        self.clips: List[ClipRecord] = []
        span = (clip_length - 1) * sample_stride
        for sequence in self.sequences:
            last = int(sequence["sequence_length"]) - span - 1
            for start in range(0, max(last + 1, 0), window_stride):
                self.clips.append(ClipRecord(sequence, tuple(start + step * sample_stride for step in range(clip_length)), start))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.clips[index]
        sequence = record.sequence
        paths = [str(sequence["frame_paths"][item]) for item in record.frame_indices]
        images = torch.stack([load_precomputed_student_rgb_tensor(path, self.normalize_mode) for path in paths])
        absolute = sequence.get("absolute_frame_ids", list(range(int(sequence["sequence_length"]))))
        sample: Dict[str, Any] = {
            "images": images,
            "frame_paths": paths,
            "frame_names": [Path(path).name for path in paths],
            "frame_indices": torch.tensor([int(absolute[item]) for item in record.frame_indices], dtype=torch.long),
            "dataset_id": torch.tensor(int(sequence.get("dataset_id", -1)), dtype=torch.long),
            "keyframe_id": str(sequence["keyframe_id"]),
            "sequence_id": str(sequence["sequence_id"]),
            "dataset_name": str(sequence["dataset_name"]),
            "sequence_length": torch.tensor(int(sequence["sequence_length"]), dtype=torch.long),
            "clip_start": torch.tensor(record.clip_start, dtype=torch.long),
            "clip_length": torch.tensor(self.clip_length, dtype=torch.long),
            "sample_stride": torch.tensor(self.sample_stride, dtype=torch.long),
            "frame_directory": str(sequence["frame_directory"]),
            "keyframe_directory": str(sequence["keyframe_directory"]),
            "depth_directory": sequence.get("depth_directory"),
            "teacher_frame_paths": [str(sequence["teacher_frame_paths"][item]) for item in record.frame_indices],
            "preprocessing_identity": str(sequence.get("preprocessing_identity", "unknown")),
        }
        if self.highlight_processor is not None:
            processed = [self.highlight_processor(unnormalize_image(image, self.normalize_mode)) for image in images]
            sample["highlight_masks"] = torch.stack([item["highlight_mask"] for item in processed])
            sample["inpainted_images"] = torch.stack([item["inpainted_image"] for item in processed])
        return sample


class TeacherClipInputDataset(Dataset):
    """Pair an RGB dataset's student-grid highlights with strict teacher RGB."""

    def __init__(self, rgb_dataset: Any) -> None:
        self.rgb_dataset = rgb_dataset
        self.normalize_mode = "zero_one"

    def __len__(self) -> int:
        return len(self.rgb_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        student = self.rgb_dataset[index]
        record = self.rgb_dataset.clips[index]
        sequence = record.sequence
        teacher_paths = sequence.get("teacher_frame_paths", sequence["frame_paths"])
        paths = [str(teacher_paths[item]) for item in record.frame_indices]
        result = dict(student)
        result["images"] = torch.stack([load_teacher_rgb_tensor(path) for path in paths])
        result["teacher_frame_paths"] = paths
        return result


class MultiSourceTemporalRGBDataset(Dataset):
    """Flatten legacy SCARED and canonical datasets without losing boundaries."""

    def __init__(self, datasets: Iterable[Any]) -> None:
        self.datasets = list(datasets)
        self.sequences: List[Dict[str, Any]] = []
        self.clips: List[ClipRecord] = []
        self._owners: List[Tuple[Any, int]] = []
        for dataset in self.datasets:
            self.sequences.extend(dataset.sequences)
            self.clips.extend(dataset.clips)
            self._owners.extend((dataset, index) for index in range(len(dataset)))
        if not self.clips:
            raise RuntimeError("No complete temporal clips were discovered")
        first = self.datasets[0]
        self.clip_length, self.sample_stride, self.window_stride = first.clip_length, first.sample_stride, first.window_stride
        self.image_height, self.image_width = 448, 560
        self.normalize_mode = first.normalize_mode
        self.resize_mode = "mixed"

    def __len__(self) -> int:
        return len(self._owners)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        owner, local_index = self._owners[index]
        return owner[local_index]


__all__ = [
    "CANONICAL_EVALUATION_DATASETS", "CANONICAL_STUDENT_SIZE", "CANONICAL_TEACHER_SIZE",
    "CANONICAL_TRAIN_DATASETS", "CanonicalTemporalRGBDataset", "MultiSourceTemporalRGBDataset", "TeacherClipInputDataset",
    "discover_canonical_sequences", "discover_processed_scared_sequences",
]
