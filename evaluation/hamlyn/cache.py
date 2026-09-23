"""Strict per-sequence prediction caches for resumable Hamlyn evaluation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from evaluation.hamlyn.constants import (
    EVALUATION_RESOLUTION_HW,
    INFERENCE_RESOLUTIONS_HW,
    PREDICTION_REPRESENTATIONS,
)
from evaluation.hamlyn.data import SequenceRecord, index_by_frame_id


CACHE_SCHEMA_VERSION = 1


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def sequence_cache_directory(output_root: Path, method: str, sequence_id: int) -> Path:
    return output_root / method / "predictions" / "sequence_{:02d}".format(sequence_id)


def expected_metadata(
    method: str,
    record: SequenceRecord,
    prediction_shape_hw: Tuple[int, int] | None = None,
) -> Dict[str, Any]:
    prediction_shape_hw = prediction_shape_hw or INFERENCE_RESOLUTIONS_HW[method]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "method": method,
        "sequence_id": record.sequence_id,
        "frame_ids": list(record.frame_ids),
        "frame_count": record.frame_count,
        "prediction_representation": PREDICTION_REPRESENTATIONS[method],
        "inference_resolution_hw": list(INFERENCE_RESOLUTIONS_HW[method]),
        "prediction_resolution_hw": list(prediction_shape_hw),
        "evaluation_resolution_hw": list(EVALUATION_RESOLUTION_HW),
    }


def prediction_file(directory: Path, identifier: int) -> Path:
    return directory / "frame_{:08d}.npy".format(identifier)


def prediction_index(directory: Path) -> Dict[int, Path]:
    return index_by_frame_id(sorted(directory.glob("frame_*.npy")), "prediction")


def _metadata_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def cache_is_complete(
    output_root: Path,
    method: str,
    record: SequenceRecord,
    prediction_shape_hw: Tuple[int, int] | None = None,
) -> bool:
    directory = sequence_cache_directory(output_root, method, record.sequence_id)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = expected_metadata(method, record, prediction_shape_hw)
    if not isinstance(metadata, dict) or not _metadata_matches(metadata, expected):
        return False
    files = prediction_index(directory)
    if set(files) != set(record.frame_ids):
        return False
    expected_shape = tuple(expected["prediction_resolution_hw"])
    for path in files.values():
        try:
            value = np.load(str(path), mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            return False
        if tuple(value.shape) != expected_shape or value.dtype != np.float32:
            return False
    return True


def prepare_cache(
    output_root: Path, method: str, record: SequenceRecord, force: bool
) -> Path:
    directory = sequence_cache_directory(output_root, method, record.sequence_id)
    method_prediction_root = (output_root / method / "predictions").resolve()
    resolved = directory.resolve()
    try:
        relative = resolved.relative_to(method_prediction_root)
    except ValueError as error:
        raise RuntimeError("Refusing to reset cache outside prediction root") from error
    if not relative.parts:
        raise RuntimeError("Refusing to reset the method prediction root")
    if force or directory.exists():
        shutil.rmtree(str(directory), ignore_errors=False)
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def save_prediction(directory: Path, identifier: int, value: np.ndarray) -> Path:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(
            "Prediction for frame {} must be finite HxW, found {}".format(
                identifier, array.shape
            )
        )
    path = prediction_file(directory, identifier)
    np.save(str(path), array, allow_pickle=False)
    return path


def finalize_cache(
    directory: Path,
    method: str,
    record: SequenceRecord,
    prediction_shape_hw: Tuple[int, int],
    inference: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = {
        **expected_metadata(method, record, prediction_shape_hw),
        "inference": dict(inference),
    }
    files = prediction_index(directory)
    if set(files) != set(record.frame_ids):
        raise RuntimeError(
            "Prediction frame IDs do not match RGB/GT for sequence {}: missing {}; extra {}"
            .format(
                record.sequence_id,
                sorted(set(record.frame_ids) - set(files))[:20],
                sorted(set(files) - set(record.frame_ids))[:20],
            )
        )
    for path in files.values():
        value = np.load(str(path), mmap_mode="r", allow_pickle=False)
        if tuple(value.shape) != tuple(prediction_shape_hw):
            raise RuntimeError(
                "Prediction shape mismatch in {}: expected {}, found {}".format(
                    path, prediction_shape_hw, value.shape
                )
            )
    atomic_write_json(directory / "metadata.json", metadata)
    return metadata


def load_cache(
    output_root: Path, method: str, record: SequenceRecord
) -> Tuple[Dict[str, Any], Dict[int, Path]]:
    directory = sequence_cache_directory(output_root, method, record.sequence_id)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("Prediction cache metadata is missing: {}".format(metadata_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not cache_is_complete(output_root, method, record):
        raise RuntimeError(
            "Prediction cache is incomplete or incompatible for {} sequence {}. "
            "Use FORCE_INFERENCE=1 to rebuild it.".format(method, record.sequence_id)
        )
    return metadata, prediction_index(directory)
