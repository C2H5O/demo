"""Ours and official DA3-Small through the identical baseline-J VDA wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gc
import torch

from datasets.crossclip_teacher_dataset import CROSSCLIP_CACHE_PROTOCOL
from evaluation.evaluate_crossclip_projection import load_official_da3_small
from evaluation.hamlyn.cache import (
    cache_is_complete,
    finalize_cache,
    prepare_cache,
    save_prediction,
)
from evaluation.hamlyn.config import RuntimeConfig
from evaluation.hamlyn.constants import INFERENCE_RESOLUTIONS_HW
from evaluation.hamlyn.data import SequenceRecord
from inference.student_video import SequenceFrames, infer_student_video
from models.student.da3_small_student import DA3SmallStudent
from utils.checkpoint import (
    require_merged_student_checkpoint,
    require_student_cache_protocol,
)
from utils.config import load_config


def _model_config(runtime: RuntimeConfig):
    config = load_config(runtime.project_root / "configs/baselines/J.yaml")
    config["student"] = dict(config["student"])
    config["student"]["checkpoint"] = str(
        runtime.da3_checkpoint_dir / "model.safetensors"
    )
    config["student"]["config_path"] = str(
        runtime.da3_checkpoint_dir / "config.json"
    )
    return config


def _load(method: str, runtime: RuntimeConfig, device: torch.device):
    config = _model_config(runtime)
    if method == "ours":
        try:
            checkpoint = torch.load(
                str(runtime.ours_checkpoint),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except (TypeError, RuntimeError):
            checkpoint = torch.load(
                str(runtime.ours_checkpoint), map_location="cpu", weights_only=False
            )
        if not isinstance(checkpoint, dict):
            raise ValueError("Ours checkpoint must contain model and config")
        require_student_cache_protocol(checkpoint, CROSSCLIP_CACHE_PROTOCOL)
        require_merged_student_checkpoint(checkpoint)
        model_config = dict(
            checkpoint.get("config", {}).get("student", config["student"])
        )
        # Keep checkpoint provenance strict while allowing the documented local
        # official-DA3 asset directory override.
        model_config["checkpoint"] = config["student"]["checkpoint"]
        model_config["config_path"] = config["student"]["config_path"]
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise ValueError("Ours checkpoint has no model state")
        model = DA3SmallStudent(model_config, device=device)
        try:
            model.load_state_dict(state, strict=True, assign=True)
        except TypeError:
            model.load_state_dict(state, strict=True)
        checkpoint.clear()
        del state, checkpoint
        gc.collect()
        return model.eval().to(device)
    if method == "da3":
        return load_official_da3_small(config, device)
    raise ValueError("DA3 adapter only supports ours and da3")


def infer_da3_sequences(
    method: str,
    records: Sequence[SequenceRecord],
    runtime: RuntimeConfig,
    force: bool = False,
) -> None:
    shape = INFERENCE_RESOLUTIONS_HW[method]
    pending = [
        record
        for record in records
        if force or not cache_is_complete(runtime.output_root, method, record, shape)
    ]
    for record in records:
        if record not in pending:
            print(
                "[{}] reuse sequence {:02d} cache".format(method, record.sequence_id),
                flush=True,
            )
    if not pending:
        return
    device = torch.device(runtime.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Hamlyn DA3 inference requires an available CUDA device")
    model = _load(method, runtime, device)
    amp = True
    for position, record in enumerate(pending, start=1):
        print(
            "[{}] {}/{} sequence {:02d}".format(
                method, position, len(pending), record.sequence_id
            ),
            flush=True,
        )
        directory = prepare_cache(runtime.output_root, method, record, force=force)
        frames = SequenceFrames(
            [record.rgb_by_id[identifier] for identifier in record.frame_ids],
            resize_mode="resize",
            height=shape[0],
            width=shape[1],
        )

        def emit(start, disparities, _intrinsics):
            for offset, disparity in enumerate(disparities):
                identifier = record.frame_ids[start + offset]
                save_prediction(directory, identifier, disparity)

        timing = infer_student_video(
            model,
            frames,
            emit,
            device=device,
            amp=amp,
        )
        if timing["output_frame_count"] != record.frame_count:
            raise RuntimeError(
                "{} emitted {} of {} Hamlyn frames".format(
                    method, timing["output_frame_count"], record.frame_count
                )
            )
        finalize_cache(
            directory,
            method,
            record,
            shape,
            {
                **timing,
                "model_input_resolution_hw": list(shape),
                "temporal_pipeline": (
                    "baseline-J infer_student_video: 32-frame windows, anchor "
                    "scale-shift alignment, overlap blending/stitching"
                ),
                "ground_truth_used_for_inference": False,
            },
        )
