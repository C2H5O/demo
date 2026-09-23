"""Subprocess adapter for unmodified official Endo3R inference."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from evaluation.hamlyn.cache import (
    cache_is_complete,
    finalize_cache,
    prepare_cache,
    save_prediction,
)
from evaluation.hamlyn.config import RuntimeConfig
from evaluation.hamlyn.constants import INFERENCE_RESOLUTIONS_HW
from evaluation.hamlyn.data import SequenceRecord, index_by_frame_id


OFFICIAL_REPOSITORY = "https://github.com/wrld/Endo3R"


def clean_environment() -> Dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def build_demo_command(
    runtime: RuntimeConfig, record: SequenceRecord, save_path: Path
) -> list[str]:
    return [
        str(runtime.endo3r_python),
        "-s",
        str(runtime.endo3r_repository / "demo.py"),
        "--demo_path",
        str(record.rgb_directory),
        "--kf_every",
        "1",
        "--save_path",
        str(save_path),
        "--ckpt_path",
        str(runtime.endo3r_checkpoint),
        "--device",
        runtime.device,
        "--resolution",
        "320",
        "--save_result",
    ]


def _run(command: Sequence[str], repository: Path, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=str(repository),
            env=clean_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Could not capture Endo3R output")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            "Official Endo3R demo.py failed with exit {}. Log: {}".format(
                return_code, log_path
            )
        )
    return time.perf_counter() - started


def infer_endo3r_sequences(
    records: Sequence[SequenceRecord],
    runtime: RuntimeConfig,
    force: bool = False,
) -> None:
    method = "endo3r"
    shape = INFERENCE_RESOLUTIONS_HW[method]
    pending = [
        record
        for record in records
        if force or not cache_is_complete(runtime.output_root, method, record, shape)
    ]
    for record in records:
        if record not in pending:
            print("[endo3r] reuse sequence {:02d} cache".format(record.sequence_id), flush=True)
    for position, record in enumerate(pending, start=1):
        print(
            "[endo3r] {}/{} sequence {:02d}".format(
                position, len(pending), record.sequence_id
            ),
            flush=True,
        )
        directory = prepare_cache(runtime.output_root, method, record, force=force)
        staging = directory / ".official"
        staging.mkdir(parents=True)
        command = build_demo_command(runtime, record, staging)
        log_path = runtime.output_root / "logs" / "endo3r_sequence_{:02d}.log".format(
            record.sequence_id
        )
        elapsed = _run(command, runtime.endo3r_repository, log_path)
        official_depth_directory = staging / record.rgb_directory.name / "depth"
        paths = sorted(official_depth_directory.glob("*.npy"))
        predictions = index_by_frame_id(paths, "Endo3R prediction")
        if set(predictions) != set(record.frame_ids):
            raise RuntimeError(
                "Endo3R prediction IDs differ from Hamlyn RGB/GT for sequence {}: "
                "missing {}; extra {}".format(
                    record.sequence_id,
                    sorted(set(record.frame_ids) - set(predictions))[:20],
                    sorted(set(predictions) - set(record.frame_ids))[:20],
                )
            )
        for identifier in record.frame_ids:
            depth = np.load(str(predictions[identifier]), allow_pickle=False)
            if tuple(depth.shape) != shape:
                raise RuntimeError(
                    "Endo3R --resolution 320 must save 256x320 depth; {} has {}".format(
                        predictions[identifier], depth.shape
                    )
                )
            save_prediction(directory, identifier, depth)
        shutil.rmtree(str(staging))
        finalize_cache(
            directory,
            method,
            record,
            shape,
            {
                "official_command": command,
                "resolution_argument": 320,
                "kf_every": 1,
                "save_result": True,
                "sequence_pipeline_seconds": elapsed,
                "prediction_resize": "bilinear depth 256x320 to 224x280 in common evaluator",
                "ground_truth_used_for_inference": False,
            },
        )
