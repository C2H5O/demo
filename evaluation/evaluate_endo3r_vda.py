"""Run the official Endo3R inference and score it with this project's VDA protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from datasets.scared_discovery import SequenceRecord, discover_scared_sequences
from evaluation.evaluate_depth import extract_frame_id
from evaluation.evaluate_vda import (
    VDA_METRIC_NAMES,
    _SequencePredictionSpool,
    _evaluate_sequence,
    _find_sequence_gt_depths,
    _student_depth_to_vda_disparity,
)
from utils.config import ensure_dir, load_config


OFFICIAL_ENDO3R_REPOSITORY = "https://github.com/wrld/Endo3R"
DEFAULT_DUST3R_NAME = "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
DEFAULT_RAFT_NAME = "raft-things.pth"


def _absolute(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else base / value).resolve()


def _sequence_dict(record: SequenceRecord) -> Dict[str, Any]:
    return {
        "dataset_id": record.dataset_id,
        "keyframe_id": record.keyframe_id,
        "sequence_id": record.sequence_id,
        "keyframe_directory": str(record.keyframe_directory),
        "frame_directory": str(record.frame_directory),
        "frame_paths": [str(path) for path in record.frame_paths],
        "sequence_length": record.sequence_length,
        "depth_directory": (
            str(record.depth_directory) if record.depth_directory else None
        ),
        "scene_points_directory": (
            str(record.scene_points_directory)
            if record.scene_points_directory
            else None
        ),
    }


def _safe_sequence_name(sequence_id: str) -> str:
    return sequence_id.replace("/", "_").replace("\\", "_")


def _index_npy_depths(directory: Path) -> Dict[int, Path]:
    if not directory.is_dir():
        return {}
    indexed: Dict[int, Path] = {}
    for path in sorted(directory.glob("*.npy")):
        frame_id = extract_frame_id(path)
        if frame_id in indexed:
            raise RuntimeError(
                "Duplicate Endo3R prediction frame ID {} in {}".format(
                    frame_id, directory
                )
            )
        indexed[frame_id] = path
    return indexed


def _load_prediction_depth(path: Path) -> np.ndarray:
    try:
        depth = np.asarray(np.load(str(path), allow_pickle=False))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "Failed to load Endo3R depth {}: {}".format(path, error)
        ) from error
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(
            "Endo3R depth {} must be 2D after squeeze, got {}".format(
                path, depth.shape
            )
        )
    depth = depth.astype(np.float32, copy=False)
    if not np.all(np.isfinite(depth)):
        raise ValueError("Endo3R depth contains NaN/Inf: {}".format(path))
    return depth


def _prediction_directory(
    output_root: Path, record: SequenceRecord
) -> Tuple[Path, Path]:
    scene_root = output_root / "predictions" / _safe_sequence_name(
        record.sequence_id
    )
    # Endo3R demo.py appends basename(demo_path) before writing depth/*.npy.
    return scene_root, scene_root / record.frame_directory.name / "depth"


def _build_demo_command(
    python_executable: str,
    endo3r_root: Path,
    record: SequenceRecord,
    scene_root: Path,
    checkpoint: Path,
    device: str,
    resolution: int,
    kf_every: int,
) -> List[str]:
    return [
        python_executable,
        str(endo3r_root / "demo.py"),
        "--demo_path",
        str(record.frame_directory),
        "--kf_every",
        str(kf_every),
        "--save_path",
        str(scene_root),
        "--ckpt_path",
        str(checkpoint),
        "--device",
        device,
        "--resolution",
        str(resolution),
        "--save_result",
    ]


def _prepare_runtime_checkpoints(
    runtime_root: Path, aliases: Mapping[str, Path]
) -> Dict[str, str]:
    checkpoint_directory = ensure_dir(runtime_root / "checkpoints")
    prepared: Dict[str, str] = {}
    for alias, source in aliases.items():
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(
                "Required Endo3R runtime checkpoint is missing: {} ({})".format(
                    source, alias
                )
            )
        destination = checkpoint_directory / alias
        if destination.is_symlink() or destination.exists():
            if destination.resolve() != source:
                raise RuntimeError(
                    "Checkpoint alias already targets another file: {}".format(
                        destination
                    )
                )
        else:
            try:
                destination.symlink_to(source)
            except OSError as error:
                raise RuntimeError(
                    "Could not create checkpoint symlink {} -> {}: {}".format(
                        destination, source, error
                    )
                ) from error
        prepared[alias] = str(source)
    return prepared


def _prediction_is_complete(
    record: SequenceRecord, prediction_directory: Path
) -> bool:
    expected = {extract_frame_id(path) for path in record.frame_paths}
    return expected == set(_index_npy_depths(prediction_directory))


def _run_inference(
    command: Sequence[str], runtime_root: Path, cuda_visible_devices: Optional[str]
) -> float:
    environment = os.environ.copy()
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    started = time.perf_counter()
    subprocess.run(
        list(command), cwd=str(runtime_root), env=environment, check=True
    )
    return time.perf_counter() - started


def _evaluate_sequence_predictions(
    sequence: Dict[str, Any],
    prediction_directory: Path,
    spool_directory: Path,
    dataset_config: Dict[str, Any],
    evaluation_config: Dict[str, Any],
) -> Dict[str, Any]:
    predictions = _index_npy_depths(prediction_directory)
    if not predictions:
        raise FileNotFoundError(
            "No Endo3R .npy depth predictions found in {}".format(
                prediction_directory
            )
        )
    height = int(evaluation_config["image_height"])
    width = int(evaluation_config["image_width"])
    spool = _SequencePredictionSpool(
        spool_directory,
        int(sequence["sequence_length"]),
        height,
        width,
    )
    rgb_index_by_id = {
        extract_frame_id(path): index
        for index, path in enumerate(sequence["frame_paths"])
    }
    unknown_prediction_ids = sorted(set(predictions) - set(rgb_index_by_id))
    if unknown_prediction_ids:
        spool.close()
        raise RuntimeError(
            "Endo3R produced frame IDs absent from RGB sequence {}: {}".format(
                sequence["sequence_id"], unknown_prediction_ids[:20]
            )
        )
    try:
        for frame_id, prediction_path in predictions.items():
            depth = _load_prediction_depth(prediction_path)
            spool.add(
                [rgb_index_by_id[frame_id]],
                _student_depth_to_vda_disparity(depth)[None],
            )
        gt_depths = _find_sequence_gt_depths(
            sequence, evaluation_config, dataset_config
        )
        result = _evaluate_sequence(
            sequence,
            spool,
            int(evaluation_config.get("gt_depth_channel", 0)),
            gt_depths,
            require_all_gt=bool(
                evaluation_config.get("require_all_gt", True)
            ),
        )
    finally:
        spool.close()
    result["prediction_directory"] = str(prediction_directory)
    result["prediction_frame_count"] = len(predictions)
    return result


def _checkpoint_aliases(
    endo3r_config: Dict[str, Any], base: Path
) -> Tuple[Path, Dict[str, Path]]:
    checkpoint = _absolute(endo3r_config["checkpoint"], base)
    aliases = {
        str(endo3r_config.get("checkpoint_alias", "endo3r.pth")): checkpoint,
        str(
            endo3r_config.get("dust3r_alias", DEFAULT_DUST3R_NAME)
        ): _absolute(endo3r_config["dust3r_checkpoint"], base),
    }
    raft_value = endo3r_config.get("raft_checkpoint")
    if raft_value:
        aliases[
            str(endo3r_config.get("raft_alias", DEFAULT_RAFT_NAME))
        ] = _absolute(raft_value, base)
    return checkpoint, aliases


def evaluate(
    config_path: Path,
    *,
    skip_inference: bool = False,
    force_inference: bool = False,
    limit_sequences: Optional[int] = None,
) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    base = config_path.parent.parent
    dataset_config = dict(config["dataset"])
    endo3r_config = dict(config["endo3r"])
    evaluation_config = dict(config["evaluation"])

    dataset_root = _absolute(dataset_config["root"], base)
    endo3r_root = _absolute(endo3r_config["root"], base)
    output_root = _absolute(config["output_root"], base)
    ensure_dir(output_root)
    if not (endo3r_root / "demo.py").is_file():
        raise FileNotFoundError(
            "Endo3R checkout does not contain demo.py: {}".format(endo3r_root)
        )
    checkpoint, aliases = _checkpoint_aliases(endo3r_config, base)
    runtime_root = ensure_dir(output_root / ".runtime")
    prepared_aliases = _prepare_runtime_checkpoints(runtime_root, aliases)

    records, malformed = discover_scared_sequences(
        dataset_root,
        split="test",
        frame_source=str(dataset_config.get("frame_source", "left_rectified")),
        strict=True,
    )
    if limit_sequences is not None:
        records = records[:limit_sequences]
    if not records:
        raise RuntimeError("No SCARED dataset 8/9 sequences were discovered")

    python_executable = str(endo3r_config.get("python", sys.executable))
    device = str(endo3r_config.get("device", "cuda:0"))
    resolution = int(endo3r_config.get("resolution", 320))
    kf_every = int(endo3r_config.get("kf_every", 1))
    cuda_visible_devices = endo3r_config.get("cuda_visible_devices")
    sequence_results: List[Dict[str, Any]] = []
    commands: List[List[str]] = []

    for position, record in enumerate(records, start=1):
        scene_root, prediction_directory = _prediction_directory(
            output_root, record
        )
        command = _build_demo_command(
            python_executable,
            endo3r_root,
            record,
            scene_root,
            checkpoint,
            device,
            resolution,
            kf_every,
        )
        commands.append(command)
        complete = _prediction_is_complete(record, prediction_directory)
        inference_seconds: Optional[float] = None
        if skip_inference:
            if not complete:
                raise RuntimeError(
                    "--skip-inference requested but predictions are incomplete: "
                    "{}".format(prediction_directory)
                )
        elif force_inference or not complete:
            ensure_dir(scene_root)
            print(
                "[Endo3R] sequence={}/{} id={} command={}".format(
                    position, len(records), record.sequence_id, command
                ),
                flush=True,
            )
            inference_seconds = _run_inference(
                command, runtime_root, cuda_visible_devices
            )
            if not _prediction_is_complete(record, prediction_directory):
                raise RuntimeError(
                    "Endo3R inference did not produce one depth for every RGB "
                    "frame: {}".format(prediction_directory)
                )
        else:
            print(
                "[Endo3R] reusing complete predictions for {}".format(
                    record.sequence_id
                ),
                flush=True,
            )
        result = _evaluate_sequence_predictions(
            _sequence_dict(record),
            prediction_directory,
            output_root,
            dataset_config,
            evaluation_config,
        )
        result["inference_seconds"] = inference_seconds
        sequence_results.append(result)

    metrics = {
        name: float(
            np.mean([item["metrics"][name] for item in sequence_results])
        )
        for name in VDA_METRIC_NAMES
    }
    result = {
        "protocol": "video-depth-anything-depth",
        "baseline_model": "Endo3R",
        "model_source": OFFICIAL_ENDO3R_REPOSITORY,
        "config": str(config_path),
        "dataset_root": str(dataset_root),
        "datasets": [8, 9],
        "frame_source": str(dataset_config.get("frame_source")),
        "endo3r_root": str(endo3r_root),
        "checkpoint": str(checkpoint),
        "runtime_checkpoint_aliases": prepared_aliases,
        "metrics": metrics,
        "sequence_count": len(sequence_results),
        "evaluation_shape_hxw": [
            int(evaluation_config["image_height"]),
            int(evaluation_config["image_width"]),
        ],
        "core_algorithm_modified": False,
        "model_output_adapter": (
            "Endo3R per-frame depth .npy -> reciprocal disparity; numeric frame "
            "ID matching; project VDA global disparity scale/shift alignment"
        ),
        "malformed_sequences": malformed,
        "commands": commands,
        "sequences": sequence_results,
    }
    output_path = output_root / str(
        evaluation_config.get("result_file", "evaluation_vda.json")
    )
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("wrote Endo3R VDA baseline: {}".format(output_path), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/endo3r_vda_baseline.yaml"),
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only score an already complete output_root/predictions tree",
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="Rerun Endo3R even when a complete prediction set exists",
    )
    parser.add_argument("--limit-sequences", type=int, default=None)
    args = parser.parse_args()
    if args.skip_inference and args.force_inference:
        parser.error("--skip-inference and --force-inference are mutually exclusive")
    evaluate(
        args.config,
        skip_inference=args.skip_inference,
        force_inference=args.force_inference,
        limit_sequences=args.limit_sequences,
    )


if __name__ == "__main__":
    main()
