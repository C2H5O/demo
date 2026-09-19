"""Full-sequence DA3 inference with VDA spatial metrics and VDA TAE."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import evaluation.evaluate_vda as vda_core
from datasets.crossclip_teacher_dataset import (
    CROSSCLIP_CACHE_PROTOCOL,
)
from datasets.scared_clip_dataset import make_scared_rgb_dataset
from evaluation.temporal_alignment import (
    VDA_TAE_METADATA,
    camera_index,
    evaluate_tae,
)
from inference.kv_sampling import resolve_kv_sampling
from inference.student_video import WINDOW, infer_student_video, sequence_frames

from models.student.da3_small_student import DA3SmallStudent
from utils.checkpoint import (
    require_merged_student_checkpoint,
    require_student_cache_protocol,
)
from utils.config import ensure_dir, load_config
from utils.merge_student_checkpoint import ensure_merged_student_checkpoint


TRAINED_STUDENT_SOURCE = "trained_student_checkpoint"
OFFICIAL_DA3_SMALL_SOURCE = "official_da3_small"


def _evaluation_resolution(
    config: Dict[str, Any], eval_config: Dict[str, Any]
) -> Tuple[int, int, int, int]:
    """Keep model-input and metric grids explicit and independent."""
    inference = config.get("inference", {})
    model_height = int(inference.get("image_height", config["dataset"]["image_height"]))
    model_width = int(inference.get("image_width", config["dataset"]["image_width"]))
    evaluation_height = int(eval_config.get("evaluation_height", model_height))
    evaluation_width = int(eval_config.get("evaluation_width", model_width))
    if min(model_height, model_width, evaluation_height, evaluation_width) <= 0:
        raise ValueError("Model-input and evaluation resolutions must be positive")
    return model_height, model_width, evaluation_height, evaluation_width


def select_protocol(config: Dict[str, Any], override: Optional[str] = None) -> str:
    value = override or str(config.get("evaluation", {}).get("protocol", "vda"))
    protocol = value.strip().lower()
    if protocol == "video-depth-anything-depth":
        protocol = "vda"
    if protocol != "vda":
        raise ValueError("Evaluation protocol must be 'vda'")
    return protocol


def _load_model(
    checkpoint_path: Path, config: Dict[str, Any], device: torch.device
) -> DA3SmallStudent:
    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True
        )
    except (TypeError, RuntimeError):
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("Cross-clip checkpoint must contain model and config")
    require_student_cache_protocol(checkpoint, CROSSCLIP_CACHE_PROTOCOL)
    require_merged_student_checkpoint(checkpoint)
    model_config = checkpoint.get("config", {}).get("student", config["student"])
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Cross-clip checkpoint has no model state")
    model = DA3SmallStudent(model_config, device=device)
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except TypeError:
        model.load_state_dict(state, strict=True)
    checkpoint.clear()
    del state, checkpoint
    gc.collect()
    return model.eval().to(device)


def _official_da3_small_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return an inference-only config with no distillation adapters or weights."""
    model_config = dict(config["student"])
    model_config.update(
        {
            "freeze_backbone": True,
            "use_backbone_lora": False,
            "freeze_depth_head": True,
            "freeze_camera_encoder": True,
            "freeze_camera_decoder": True,
        }
    )
    return model_config


def load_official_da3_small(
    config: Dict[str, Any], device: torch.device
) -> DA3SmallStudent:
    """Strict-load the untouched official DA3-Small safetensors for baseline eval."""
    model = DA3SmallStudent(_official_da3_small_config(config), device=device)
    if model.lora_modules:
        raise RuntimeError("Official DA3-Small baseline must not contain LoRA modules")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Official DA3-Small baseline must be inference-only")
    return model.eval().to(device)


def _require_scared_8_9(sequences: Dict[str, Dict[str, Any]]) -> None:
    dataset_ids = {int(sequence["dataset_id"]) for sequence in sequences.values()}
    if dataset_ids != {8, 9}:
        raise RuntimeError(
            "Official DA3-Small baseline requires exactly SCARED datasets 8 and 9; "
            "discovered {}".format(sorted(dataset_ids))
        )


def _evaluation_section(protocol: str, model_source: str) -> str:
    if model_source == OFFICIAL_DA3_SMALL_SOURCE:
        return "da3_small_baseline_{}_evaluation".format(protocol)
    if model_source == TRAINED_STUDENT_SOURCE:
        return "{}_evaluation".format(protocol)
    raise ValueError("Unsupported evaluation model source {!r}".format(model_source))


def _evaluation_model(
    checkpoint: Optional[Path],
    config: Dict[str, Any],
    device: torch.device,
    model_source: str,
) -> DA3SmallStudent:
    if model_source == OFFICIAL_DA3_SMALL_SOURCE:
        if checkpoint is not None:
            raise ValueError("Official DA3-Small baseline does not accept a training checkpoint")
        return load_official_da3_small(config, device)
    if model_source != TRAINED_STUDENT_SOURCE:
        raise ValueError("Unsupported evaluation model source {!r}".format(model_source))
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError("Student checkpoint not found: {}".format(checkpoint))
    return _load_model(checkpoint, config, device)


def _dataset_and_ground_truth(
    config: Dict[str, Any], eval_config: Dict[str, Any], split: str
) -> Tuple[Any, Dict[str, Dict[str, Any]], Dict[str, Tuple[Path, Dict[int, Path]]], List[Dict[str, str]]]:
    dataset_config = dict(config["dataset"])
    rgb_root = eval_config.get("rgb_root")
    if rgb_root:
        # Evaluation RGB and training's preprocessed student RGB are separate
        # data sources. An explicit evaluation root must take precedence over
        # every legacy/canonical training-root alias.
        dataset_config["root"] = str(rgb_root)
        dataset_config["legacy_scared_root"] = str(rgb_root)
        dataset_config["canonical_root"] = None
    dataset_config["frame_source"] = str(
        eval_config.get("frame_source", dataset_config.get("frame_source", "auto"))
    )
    # Discovery must include short sequences; inference owns window construction.
    dataset_config.update(clip_length=1, sample_stride=1, window_stride=1)
    dataset_config["drop_incomplete_clip"] = False
    # Detection/inpainting is a training-only auxiliary and does not alter RGB.
    dataset_config["highlight"] = {"enabled": False}
    # Evaluation needs RGB only. The cross-clip training factory intentionally
    # forces strict stride-eight tail dropping for teacher-cache compatibility.
    dataset = make_scared_rgb_dataset(dataset_config, split)
    sequences = {str(item["sequence_id"]): item for item in dataset.sequences}
    gt_by_sequence: Dict[str, Tuple[Path, Dict[int, Path]]] = {}
    skipped: List[Dict[str, str]] = []
    for sequence_id, sequence in sequences.items():
        try:
            gt_by_sequence[sequence_id] = vda_core._find_sequence_gt_depths(
                sequence, eval_config, dataset_config
            )
        except FileNotFoundError as error:
            skipped.append({"sequence_id": sequence_id, "reason": str(error)})
    return dataset, sequences, gt_by_sequence, skipped


def evaluate_vda(
    config_path: Path,
    checkpoint_override: Optional[Path] = None,
    split_override: Optional[str] = None,
    output_override: Optional[Path] = None,
    limit_clips: Optional[int] = None,
    model_source: str = TRAINED_STUDENT_SOURCE,
) -> Dict[str, Any]:
    """Infer each complete RGB sequence, then score once per absolute frame.

    limit_clips is retained as a Python compatibility name for a window budget.
    """
    config = load_config(config_path)
    kv_config = resolve_kv_sampling(config)
    kv_config.frame_budget(WINDOW)
    eval_config = dict(config.get(_evaluation_section("vda", model_source), {}))
    split = split_override or str(eval_config.get("split", "test"))
    if model_source == OFFICIAL_DA3_SMALL_SOURCE and split != "test":
        raise ValueError("Official DA3-Small baseline is fixed to SCARED test split 8 and 9")
    checkpoint = (
        None
        if model_source == OFFICIAL_DA3_SMALL_SOURCE
        else checkpoint_override or Path(str(eval_config["checkpoint"]))
    )
    output = output_override or Path(str(eval_config["output"]))
    ensure_dir(output.parent)
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    dataset, sequences, gt_depths, skipped = _dataset_and_ground_truth(
        config, eval_config, split
    )
    if model_source == OFFICIAL_DA3_SMALL_SOURCE:
        _require_scared_8_9(sequences)
    if limit_clips is not None and limit_clips <= 0:
        raise ValueError("Window limit must be positive")
    if not gt_depths:
        raise RuntimeError("No sequences contain configured depth GT")
    tae_config = eval_config.get("tae", {})
    tae_enabled = bool(tae_config.get("enabled", True))
    if tae_enabled:
        if config["dataset"].get("resize_mode", "resize") != "resize":
            raise ValueError("TAE currently requires full-FOV RGB with resize_mode=resize")
        # Fail before costly model inference if a required dataset pose is absent.
        if tae_config.get("require_all_pairs", True):
            from evaluation.scared_gt import extract_frame_id
            for sequence_id in gt_depths:
                sequence = sequences[sequence_id]
                directory, cameras = camera_index(sequence, eval_config)
                missing = [extract_frame_id(p) for p in sequence["frame_paths"]
                           if extract_frame_id(p) not in cameras]
                if missing:
                    raise FileNotFoundError("TAE dataset cameras missing in {}: {}".format(directory, missing[:20]))
    inference_checkpoint = checkpoint
    if model_source == TRAINED_STUDENT_SOURCE:
        assert checkpoint is not None
        inference_checkpoint = ensure_merged_student_checkpoint(checkpoint, config)
    model = _evaluation_model(inference_checkpoint, config, device, model_source)
    amp = bool(eval_config.get("amp", True)) and device.type == "cuda"
    model_height, model_width, evaluation_height, evaluation_width = (
        _evaluation_resolution(config, eval_config)
    )
    remaining = limit_clips
    sequence_results = []
    resolution_audit_printed = False
    for sequence_id, sequence in sequences.items():
        if sequence_id not in gt_depths or remaining == 0:
            continue
        frames = sequence_frames(
            sequence, config["dataset"], raw_rgb=bool(eval_config.get("rgb_root")),
            inference_height=model_height, inference_width=model_width)
        spool = vda_core._SequencePredictionSpool(
            output.parent, len(frames), evaluation_height, evaluation_width
        )
        try:
            def emit(start, disparities, intrinsics):
                spool.add(range(start, start + len(disparities)), disparities)
            timing = infer_student_video(model, frames, emit, device=device, amp=amp,
                                         max_windows=remaining, kv_sampling=kv_config)
            if not resolution_audit_printed and isinstance(model, DA3SmallStudent):
                patch_size = int(model.config.patch_size)
                native = timing["native_prediction_resolution_hw"]
                audit = {
                    "model_input": f"{model_height}x{model_width}",
                    "patch_size": patch_size,
                    "patch_grid": f"{model_height // patch_size}x{model_width // patch_size}",
                    "patches_per_frame": model_height * model_width // patch_size ** 2,
                    "native_prediction": f"{native[0]}x{native[1]}",
                    "evaluation_grid": f"{evaluation_height}x{evaluation_width}",
                    "window_length": WINDOW,
                }
                print("DA3 inference resolution audit: " + json.dumps(audit), flush=True)
                if kv_config.enabled and kv_config.method == "role_layer_spatial_kv":
                    print("H sparse KV audit: " + json.dumps({
                        "full_patches_per_frame": audit["patches_per_frame"],
                        "stride2_patches_per_frame":
                            (model_height // (2 * patch_size)) *
                            (model_width // (2 * patch_size)),
                        "block5": "sparse", "block7": "sparse",
                        "block9": "dense", "block11": "dense",
                    }), flush=True)
                resolution_audit_printed = True
            spool.flush()
            item = vda_core._evaluate_sequence(
                sequence, spool, int(eval_config.get("gt_depth_channel", 0)),
                gt_depths[sequence_id],
                require_all_gt=bool(eval_config.get("require_all_gt", True)) and limit_clips is None)
            if tae_enabled:
                # A debug window limit evaluates temporal pairs only in its inferred prefix.
                temporal_sequence = dict(sequence)
                temporal_sequence["frame_paths"] = sequence["frame_paths"][:timing["output_frame_count"]]
                item["temporal"] = evaluate_tae(
                    temporal_sequence, spool, item, eval_config
                )
                item["metrics"]["tae"] = item["temporal"]["tae"]
            item["inference"] = timing
            sequence_results.append(item)
        finally:
            spool.close()
        if remaining is not None:
            remaining -= timing["window_count"]
    if not sequence_results:
        raise RuntimeError("No sequence was evaluated")
    metrics = {name: float(np.mean([item["metrics"][name] for item in sequence_results]))
               for name in vda_core.VDA_METRIC_NAMES}
    temporal = (
        [
            item["metrics"]["tae"]
            for item in sequence_results
            if item["metrics"]["tae"] is not None
        ]
        if tae_enabled
        else []
    )
    if tae_enabled:
        metrics["tae"] = float(np.mean(temporal)) if temporal else None
    total_frames = sum(item["inference"]["output_frame_count"] for item in sequence_results)
    total_seconds = sum(item["inference"]["model_forward_seconds"] for item in sequence_results)
    pipeline_seconds = sum(item["inference"]["sequence_pipeline_seconds"] for item in sequence_results)
    peak_memory = [item["inference"]["peak_cuda_memory_allocated_bytes"] for item in sequence_results
                   if item["inference"]["peak_cuda_memory_allocated_bytes"] is not None]
    peak_reserved = [item["inference"]["peak_cuda_memory_reserved_bytes"] for item in sequence_results
                     if item["inference"]["peak_cuda_memory_reserved_bytes"] is not None]
    complete = not skipped and len(sequence_results) == len(sequences) and all(
        item["missing_prediction_count"] == 0 and item["inference"]["output_frame_count"] ==
        len(sequences[item["sequence_id"]]["frame_paths"]) for item in sequence_results)
    result = {
        "protocol": (
            "video-depth-anything-depth+video-depth-anything-tae-scared-v2"
            if tae_enabled
            else "video-depth-anything-depth-scared-v2"
        ),
        "config": str(config_path), "model_source": model_source,
        "checkpoint": str(checkpoint) if checkpoint is not None else str(config["student"]["checkpoint"]),
        "inference_checkpoint": (
            str(inference_checkpoint)
            if inference_checkpoint is not None
            else str(config["student"]["checkpoint"])
        ),
        "split": split, "metrics": metrics, "metric_aggregation": "macro mean over evaluated sequences",
        "sequence_count": len(sequence_results), "expected_sequence_count": len(sequences),
        "complete_gt_coverage": complete,
        "full_test_set": split == "test" and limit_clips is None and complete,
        "inference_mode": "complete sequence, VDA 32-view windows with anchor alignment and 8-frame disparity blending",
        "inference_frame_count": total_frames,
        "inference_window_count": sum(item["inference"]["window_count"] for item in sequence_results),
        "model_input_frame_count": sum(item["inference"]["model_input_frame_count"] for item in sequence_results),
        "total_model_inference_seconds": total_seconds,
        "mean_frame_inference_seconds": total_seconds / total_frames,
        "mean_frame_inference_ms": 1000 * total_seconds / total_frames,
        "inference_fps": total_frames / total_seconds if total_seconds else None,
        "kv_sampling": kv_config.as_dict(),
        "total_sequence_pipeline_seconds": pipeline_seconds,
        "pipeline_fps": total_frames / pipeline_seconds if pipeline_seconds else None,
        "peak_cuda_memory_allocated": max(peak_memory) if peak_memory else None,
        "peak_cuda_memory_reserved": max(peak_reserved) if peak_reserved else None,
        # Preserve the established byte-explicit aliases for downstream readers.
        "peak_cuda_memory_allocated_bytes": max(peak_memory) if peak_memory else None,
        "peak_cuda_memory_reserved_bytes": max(peak_reserved) if peak_reserved else None,
        "timing_scope": sequence_results[0]["inference"]["timing_scope"],
        "warmup_excluded": False, "amp": amp, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "input_resolution_hw": [model_height, model_width],
        "model_input_resolution_hw": [model_height, model_width],
        "evaluation_resolution_hw": [evaluation_height, evaluation_width],
        "window_limit": limit_clips,
        "skipped_sequences_without_gt": skipped, "sequences": sequence_results,
    }
    if tae_enabled:
        result.update(
            VDA_TAE_METADATA,
            tae_sequence_count=len(temporal),
            complete_tae_coverage=(
                bool(temporal)
                and complete
                and all(
                    item["temporal"]["status"] == "complete"
                    for item in sequence_results
                )
            ),
        )
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(
        "wrote full-sequence {} evaluation: {}".format(
            "VDA + TAE" if tae_enabled else "VDA", output
        )
    )
    return result


def evaluate(
    config_path: Path,
    checkpoint: Optional[Path] = None,
    split: Optional[str] = None,
    output: Optional[Path] = None,
    limit_clips: Optional[int] = None,
    protocol: Optional[str] = None,
    model_source: str = TRAINED_STUDENT_SOURCE,
) -> Dict[str, Any]:
    config = load_config(config_path)
    select_protocol(config, protocol)
    return evaluate_vda(
        config_path, checkpoint, split, output, limit_clips, model_source
    )


def evaluate_official_da3_small(
    config_path: Path,
    output: Optional[Path] = None,
    limit_clips: Optional[int] = None,
    protocol: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate untouched official DA3-Small on raw SCARED datasets 8 and 9."""
    return evaluate(
        config_path,
        checkpoint=None,
        split="test",
        output=output,
        limit_clips=limit_clips,
        protocol=protocol,
        model_source=OFFICIAL_DA3_SMALL_SOURCE,
    )


__all__ = [
    "OFFICIAL_DA3_SMALL_SOURCE",
    "TRAINED_STUDENT_SOURCE",
    "evaluate",
    "evaluate_official_da3_small",
    "evaluate_vda",
    "load_official_da3_small",
    "select_protocol",
]
