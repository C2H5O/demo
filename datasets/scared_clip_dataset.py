"""SCARED RGB clip construction and stable clip metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from datasets.multidataset import (
    CanonicalTemporalRGBDataset,
    MultiSourceTemporalRGBDataset,
    discover_canonical_sequences,
    discover_processed_scared_sequences,
)
from datasets.scared_dataset import ScaredTemporalRGBDataset


def _absolute_path(root: Path, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def make_scared_rgb_dataset(
    dataset_config: Dict[str, Any], split: str
) -> Any:
    """Build legacy SCARED plus complete canonical sequences for a split.

    The public name is kept for compatibility with existing entrypoints.
    Canonical datasets are discovered only from the configured processed root;
    Hamlyn is returned only for non-training evaluation splits.
    """
    config = dict(dataset_config)
    config.pop("name", None)
    canonical_root = config.pop("canonical_root", None)
    legacy_root = config.pop("legacy_scared_root", config.get("root"))
    config.pop("train_manifest_path", None)
    config.pop("test_manifest_path", None)
    config["split"] = split
    config["manifest_path"] = dataset_config.get("{}_manifest_path".format(split))
    config["root"] = legacy_root
    datasets = []
    processed_scared = discover_processed_scared_sequences(legacy_root, split)
    if processed_scared:
        dataset = CanonicalTemporalRGBDataset(
            processed_scared,
            clip_length=int(config.get("clip_length", 16)),
            sample_stride=int(config.get("sample_stride", 1)),
            window_stride=int(config.get("window_stride", 1)),
            normalize_mode=str(dataset_config.get("normalize_mode", "minus_one_one")),
            highlight=dict(dataset_config.get("highlight", {})),
        )
    else:
        try:
            dataset = ScaredTemporalRGBDataset(**config)
        except FileNotFoundError:
            # A canonical-only evaluation machine need not mount legacy SCARED.
            if not canonical_root:
                raise
            dataset = None
    if dataset is not None:
        for sequence in dataset.sequences:
            sequence["dataset_name"] = "SCARED"
            sequence.setdefault(
                "preprocessing_identity",
                "processed_scared" if processed_scared else "legacy_scared",
            )
            sequence.setdefault("absolute_frame_ids", list(range(int(sequence["sequence_length"]))))
        datasets.append(dataset)

        if not processed_scared:
            for sequence in dataset.sequences:
                sequence["frame_paths"] = [_absolute_path(dataset.root, str(path)) for path in sequence["frame_paths"]]
                for key in ("keyframe_directory", "frame_directory", "calibration_path", "depth_directory", "disparity_directory", "frame_data_directory", "reprojection_directory", "scene_points_directory", "point_cloud_path", "video_path"):
                    if key in sequence:
                        sequence[key] = _absolute_path(dataset.root, sequence.get(key))
    if canonical_root:
        canonical = discover_canonical_sequences(canonical_root, split)
        if canonical:
            datasets.append(CanonicalTemporalRGBDataset(
                canonical,
                clip_length=int(config.get("clip_length", 16)),
                sample_stride=int(config.get("sample_stride", 1)),
                window_stride=int(config.get("window_stride", 1)),
                normalize_mode=str(dataset_config.get("normalize_mode", "minus_one_one")),
                highlight=dict(dataset_config.get("highlight", {})),
            ))
    if not datasets:
        raise RuntimeError("No legacy SCARED or complete canonical sequences were discovered")
    return datasets[0] if len(datasets) == 1 else MultiSourceTemporalRGBDataset(datasets)


def clip_metadata(dataset: ScaredTemporalRGBDataset, index: int) -> Dict[str, Any]:
    """Read one clip identity without decoding RGB images."""
    record = dataset.clips[index]
    sequence = record.sequence
    frame_paths = [
        str(sequence["frame_paths"][frame_index])
        for frame_index in record.frame_indices
    ]
    return {
        "dataset_name": str(sequence.get("dataset_name", "SCARED")),
        "dataset_id": int(sequence["dataset_id"]),
        "keyframe_id": str(sequence["keyframe_id"]),
        "sequence_id": str(sequence["sequence_id"]),
        "sequence_length": int(sequence["sequence_length"]),
        "clip_start": int(record.clip_start),
        "clip_length": int(dataset.clip_length),
        "sample_stride": int(dataset.sample_stride),
        "frame_indices": [int(sequence.get("absolute_frame_ids", range(int(sequence["sequence_length"])))[item]) for item in record.frame_indices],
        "frame_paths": frame_paths,
        "frame_names": [Path(path).name for path in frame_paths],
        "teacher_frame_paths": [str(sequence.get("teacher_frame_paths", sequence["frame_paths"])[frame_index]) for frame_index in record.frame_indices],
        "preprocessing_identity": str(sequence.get("preprocessing_identity", "legacy_scared")),
    }


__all__ = ["clip_metadata", "make_scared_rgb_dataset"]
