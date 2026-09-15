"""Full-sequence official EndoDAV inference and unified evaluation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from endodaveval.config import (
    EVALUATION_RESOLUTION_HW,
    INTERNAL_MODEL_RESOLUTION_HW,
    atomic_write_json,
    load_config,
    project_path,
)
from endodaveval.data import (
    SequenceRecord,
    discover_sequences,
    index_by_frame_id,
    matched_frame_ids,
)
from endodaveval.endodav import (
    AUDITED_OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    infer_official_full_video,
    load_official_model,
    load_raw_rgb,
    validate_official_repository,
)
from endodaveval.temporal_alignment import VDA_TAE_METADATA, evaluate_tae, preflight_tae
from endodaveval.vda import METRIC_NAMES, evaluate_prediction_files


STAGES = ("preflight", "infer", "evaluate", "all")


def _safe_name(record: SequenceRecord) -> str:
    return record.sequence_id.replace("/", "_").replace("\\", "_")


def prediction_directory(output_root: Path, record: SequenceRecord) -> Path:
    return output_root / "predictions" / _safe_name(record) / "depth"


def prediction_index(directory: Path) -> Dict[int, Path]:
    paths = [path for path in directory.glob("*.npy") if path.is_file()]
    return index_by_frame_id(paths)


def _git_head(repository: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def preflight(config: Mapping[str, Any], records: List[SequenceRecord]) -> Dict[str, Any]:
    endodav = config["endodav"]
    repository = project_path(endodav["repository"], config)
    checkpoint = project_path(endodav["checkpoint"], config)
    pretrained_path = project_path(endodav["pretrained_path"], config)
    sources = validate_official_repository(repository)
    missing = [
        str(path)
        for path in (checkpoint, pretrained_path / "video_depth_anything_vits.pth")
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Required EndoDAV checkpoints are missing: {}".format(missing))
    for record in records:
        matched_frame_ids(record, bool(config["evaluation"].get("require_all_frames", True)))
        preflight_tae(record, config["evaluation"].get("tae", {}))
    return {
        "status": "ok",
        "official_repository": OFFICIAL_REPOSITORY,
        "audited_official_commit": AUDITED_OFFICIAL_COMMIT,
        "checkout_head": _git_head(repository),
        "official_source_files": sources,
        "official_source_modified": False,
        "checkpoint": str(checkpoint),
        "pretrained_checkpoint": str(pretrained_path / "video_depth_anything_vits.pth"),
    }


def _save_native_depths(
    directory: Path, frame_ids: Tuple[int, ...], depth: np.ndarray
) -> Dict[int, Path]:
    if len(frame_ids) != len(depth):
        raise RuntimeError("Official EndoDAV output count differs from RGB frame IDs")
    directory.mkdir(parents=True, exist_ok=True)
    expected_names = {"depth_{:06d}.npy".format(identifier) for identifier in frame_ids}
    existing_names = {path.name for path in directory.glob("*.npy")}
    extras = sorted(existing_names - expected_names)
    if extras:
        raise RuntimeError("Prediction directory contains stale frame IDs: {}".format(extras[:20]))
    result = {}
    for identifier, value in zip(frame_ids, depth):
        path = directory / "depth_{:06d}.npy".format(identifier)
        np.save(str(path), np.asarray(value, dtype=np.float32))
        result[identifier] = path
    return result


def _require_prediction_ids(record: SequenceRecord, predictions: Mapping[int, Path]):
    expected = set(record.rgb_by_id)
    actual = set(predictions)
    if actual != expected:
        raise RuntimeError(
            "Prediction IDs do not match RGB IDs for {}: missing {}; extra {}".format(
                record.sequence_id, sorted(expected - actual)[:20], sorted(actual - expected)[:20]
            )
        )


def run_pipeline(
    config_path: Path,
    stage: str = "all",
    limit_sequences: Optional[int] = None,
) -> Dict[str, Any]:
    if stage not in STAGES:
        raise ValueError("stage must be one of {}".format(STAGES))
    if limit_sequences is not None and limit_sequences <= 0:
        raise ValueError("limit_sequences must be positive")
    config = load_config(config_path)
    dataset = config["dataset"]
    endodav = config["endodav"]
    configured_cuda = endodav.get("cuda_visible_devices")
    if configured_cuda is not None:
        # Official infer_video_depth uses its default logical CUDA device.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(configured_cuda)
    evaluation = config["evaluation"]
    output_root = project_path(config["output_root"], config)
    records, skipped = discover_sequences(
        project_path(dataset["root"], config),
        [int(value) for value in dataset["dataset_ids"]],
        [str(value) for value in dataset["frame_sources"]],
        str(dataset["ground_truth_directory"]),
    )
    if limit_sequences is not None:
        records = records[:limit_sequences]
    health = preflight(config, records)
    output_root.mkdir(parents=True, exist_ok=True)
    if stage == "preflight":
        result = {"stage": stage, "preflight": health, "sequence_count": len(records), "skipped_sequences": skipped}
        atomic_write_json(output_root / "preflight.json", result)
        return result

    model = None
    if stage in ("infer", "all"):
        model = load_official_model(
            project_path(endodav["repository"], config),
            project_path(endodav["checkpoint"], config),
            project_path(endodav["pretrained_path"], config),
            str(endodav.get("device", "cuda:0")),
        )

    sequence_results = []
    input_shapes = set()
    native_shapes = set()
    for record in records:
        frame_ids = matched_frame_ids(
            record, bool(evaluation.get("require_all_frames", True))
        )
        directory = prediction_directory(output_root, record)
        timing: Dict[str, Any] = {}
        if stage in ("infer", "all"):
            frames = load_raw_rgb([record.rgb_by_id[identifier] for identifier in frame_ids])
            input_shapes.add(tuple(frames.shape[1:3]))
            native_depth, timing = infer_official_full_video(
                model, frames, str(endodav.get("device", "cuda:0"))
            )
            predictions = _save_native_depths(directory, frame_ids, native_depth)
            atomic_write_json(directory.parent / "timing.json", timing)
        else:
            predictions = prediction_index(directory)
            _require_prediction_ids(record, predictions)
            timing_path = directory.parent / "timing.json"
            if timing_path.is_file():
                import json

                timing = json.loads(timing_path.read_text(encoding="utf-8"))
        _require_prediction_ids(record, predictions)
        first_prediction = np.load(str(predictions[frame_ids[0]]), mmap_mode="r")
        native_shape = tuple(first_prediction.shape)
        native_shapes.add(native_shape)
        rgb_probe = load_raw_rgb([record.rgb_by_id[frame_ids[0]]])
        rgb_shape = tuple(rgb_probe.shape[1:3])
        input_shapes.add(rgb_shape)
        if native_shape != rgb_shape:
            raise RuntimeError(
                "Official prediction must remain at input-frame resolution; found {} vs {}".format(
                    native_shape, rgb_shape
                )
            )
        sequence_state: Dict[str, Any] = {
            **record.to_dict(),
            "prediction_directory": str(directory),
            "model_input_resolution_hw": list(rgb_shape),
            "internal_model_resolution_hw": list(INTERNAL_MODEL_RESOLUTION_HW),
            "native_prediction_resolution_hw": list(native_shape),
            "evaluation_resolution_hw": list(EVALUATION_RESOLUTION_HW),
            "inference": timing,
        }
        if stage in ("evaluate", "all"):
            spatial, aligned_depth = evaluate_prediction_files(
                predictions,
                record.ground_truth_by_id,
                frame_ids,
                float(dataset["ground_truth_scale"]),
                int(dataset.get("ground_truth_channel", 0)),
            )
            temporal = evaluate_tae(
                record,
                aligned_depth,
                frame_ids,
                evaluation.get("tae", {}),
                device=str(evaluation.get("device", "cpu")),
            )
            spatial["temporal"] = temporal
            spatial["metrics"]["tae"] = temporal["tae"]
            sequence_state.update(spatial)
        sequence_results.append(sequence_state)

    result: Dict[str, Any] = {
        "schema_version": 1,
        "model": "EndoDAV",
        "dataset": "SCARED",
        "dataset_ids": [8, 9],
        "stage": stage,
        "protocol": "unified-vda-spatial+vda-tae-scared-256x320-v1",
        **VDA_TAE_METADATA,
        "official_sources": health,
        "official_inference": "one model.infer_video_depth(frames) call per complete sequence",
        "external_windowing": False,
        "ground_truth_used_for_inference": False,
        "internal_model_resolution_hw": list(INTERNAL_MODEL_RESOLUTION_HW),
        "model_input_resolution_hw": list(next(iter(input_shapes))) if len(input_shapes) == 1 else None,
        "model_input_resolutions_hw": [list(shape) for shape in sorted(input_shapes)],
        "native_prediction_resolution_hw": list(next(iter(native_shapes))) if len(native_shapes) == 1 else None,
        "native_prediction_resolutions_hw": [list(shape) for shape in sorted(native_shapes)],
        "evaluation_resolution_hw": list(EVALUATION_RESOLUTION_HW),
        "prediction_representation": "official normalized disparity -> official 0.1..150m depth -> reciprocal disparity",
        "prediction_interpolation": "bilinear disparity to 256x320 after reciprocal",
        "ground_truth_interpolation": "nearest-neighbor depth to 256x320",
        "metric_aggregation": "per-frame within sequence, then macro mean over evaluated sequences",
        "sequence_count": len(sequence_results),
        "skipped_sequences": skipped,
        "sequences": sequence_results,
    }
    if stage in ("evaluate", "all"):
        result["metrics"] = {
            name: float(np.mean([item["metrics"][name] for item in sequence_results]))
            for name in (*METRIC_NAMES, "tae")
        }
    forward_values = [item["inference"].get("model_forward_seconds") for item in sequence_results if item["inference"].get("model_forward_seconds") is not None]
    pipeline_values = [item["inference"].get("sequence_pipeline_seconds") for item in sequence_results if item["inference"].get("sequence_pipeline_seconds") is not None]
    result["timing"] = {
        "model_forward_seconds": float(sum(forward_values)) if forward_values else None,
        "sequence_pipeline_seconds": float(sum(pipeline_values)) if pipeline_values else None,
        "evaluation_resize_in_model_forward_timing": False,
    }
    result_file = output_root / str(
        evaluation.get("result_file", "evaluation_vda.json")
        if stage in ("evaluate", "all")
        else "inference_manifest.json"
    )
    atomic_write_json(result_file, result)
    return result
