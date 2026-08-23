"""SCARED RGB clip construction and stable clip metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from datasets.scared_dataset import ScaredTemporalRGBDataset


def _absolute_path(root: Path, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def make_scared_rgb_dataset(
    dataset_config: Dict[str, Any], split: str
) -> ScaredTemporalRGBDataset:
    """Build the RGB dataset and resolve discovered paths once."""
    config = dict(dataset_config)
    config.pop("name", None)
    config.pop("train_manifest_path", None)
    config.pop("test_manifest_path", None)
    config["split"] = split
    config["manifest_path"] = dataset_config.get("{}_manifest_path".format(split))
    dataset = ScaredTemporalRGBDataset(**config)

    for sequence in dataset.sequences:
        sequence["frame_paths"] = [
            _absolute_path(dataset.root, str(path)) for path in sequence["frame_paths"]
        ]
        for key in (
            "keyframe_directory",
            "frame_directory",
            "calibration_path",
            "depth_directory",
            "disparity_directory",
            "frame_data_directory",
            "reprojection_directory",
            "scene_points_directory",
            "point_cloud_path",
            "video_path",
        ):
            if key in sequence:
                sequence[key] = _absolute_path(dataset.root, sequence.get(key))
    return dataset


def clip_metadata(dataset: ScaredTemporalRGBDataset, index: int) -> Dict[str, Any]:
    """Read one clip identity without decoding RGB images."""
    record = dataset.clips[index]
    sequence = record.sequence
    frame_paths = [
        str(sequence["frame_paths"][frame_index])
        for frame_index in record.frame_indices
    ]
    return {
        "dataset_id": int(sequence["dataset_id"]),
        "keyframe_id": str(sequence["keyframe_id"]),
        "sequence_id": str(sequence["sequence_id"]),
        "sequence_length": int(sequence["sequence_length"]),
        "clip_start": int(record.clip_start),
        "clip_length": int(dataset.clip_length),
        "sample_stride": int(dataset.sample_stride),
        "frame_indices": list(record.frame_indices),
        "frame_paths": frame_paths,
        "frame_names": [Path(path).name for path in frame_paths],
    }


__all__ = ["clip_metadata", "make_scared_rgb_dataset"]
