"""Trainer for same-clip VGGT-Omega pseudo-GT -> DA3-Small distillation."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from datasets.crossclip_teacher_dataset import (
    CROSSCLIP_CACHE_PROTOCOL,
    make_teacher_cache_rgb_dataset,
)
from cache.generate_crossclip_teacher_cache import (
    SUPERVISION_SHAPE,
    TEACHER_SHAPE,
    canonicalize_teacher_outputs,
)
from datasets.direct_teacher_distillation_dataset import (
    DirectTeacherDistillationDataset,
    FullOnlineTeacherDistillationDataset,
    build_direct_teacher_distillation_dataloader,
)
from datasets.scared_clip_dataset import make_scared_rgb_dataset
from losses.direct_teacher_distillation_loss import DirectTeacherDistillationLoss
from losses.regularizer_diagnostics import loss_share_logs, regularizer_diagnostics
from losses.attention_distillation_loss import (
    AttentionDistillationConfig,
    CrossFrameAttentionDistillationLoss,
)
from models.student.da3_small_student import DA3SmallStudent
from models.teacher.vggt_omega_wrapper import VGGTOmegaTeacher
from models.teacher.output_adapter import adapt_teacher_outputs
from utils.checkpoint import (
    DIRECT_TEACHER_DISTILLATION_PROTOCOL,
    atomic_torch_save,
    require_student_cache_protocol,
    require_training_objective,
)
from utils.config import ensure_dir, load_config
from utils.seed import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _record_cuda_event(enabled: bool) -> Optional[torch.cuda.Event]:
    if not enabled:
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _elapsed_ms(
    start: Optional[torch.cuda.Event], end: Optional[torch.cuda.Event]
) -> float:
    return start.elapsed_time(end) if start is not None and end is not None else 0.0


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _teacher_is_full_online(config: Mapping[str, Any]) -> bool:
    teacher = config.get("teacher", {})
    return (
        str(teacher.get("mode", "cache")).lower() == "full_online"
        and not bool(teacher.get("use_cache", True))
    )


def _build_dataset(
    config: Dict[str, Any], split: str
) -> DirectTeacherDistillationDataset | FullOnlineTeacherDistillationDataset:
    teacher = config["teacher"]
    if _teacher_is_full_online(config):
        rgb = make_scared_rgb_dataset(config["dataset"], split)
        dataset = FullOnlineTeacherDistillationDataset(
            rgb,
            teacher_input_height=int(teacher.get("input_height", 1024)),
            teacher_input_width=int(teacher.get("input_width", 1280)),
        )
        print(
            "ordinary online sampling: length={} sample_stride={} start_stride={} "
            "clips={} teacher_cache_used=false".format(
                int(dataset.rgb_dataset.clip_length),
                int(dataset.rgb_dataset.sample_stride),
                int(dataset.rgb_dataset.window_stride),
                len(dataset),
            )
        )
        return dataset
    raw_root = teacher.get("raw_cache_root")
    if not raw_root:
        raise ValueError("teacher.raw_cache_root must be configured")
    cache_root = Path(str(raw_root)) / split
    rgb = make_teacher_cache_rgb_dataset(
        config["dataset"], split, cache_root=cache_root
    )
    dataset = DirectTeacherDistillationDataset(
        rgb,
        cache_root,
        expected_base_checkpoint=str(teacher.get("cache_checkpoint_identity", teacher["pretrained_checkpoint"])),
        online_teacher_attention=bool(
            config.get("attention_distill", {}).get("enabled", False)
        ),
    )
    print(
        "same-clip cache sampling: length=16 start_stride=8 first_frames=1,9,17,... "
        "matched={} skipped_off_stride={} skipped_without_cache={} root={}".format(
            len(dataset), dataset.skipped_off_stride, dataset.skipped_without_cache, cache_root
        )
    )
    return dataset


def _print_same_clip_examples(
    dataset: DirectTeacherDistillationDataset | FullOnlineTeacherDistillationDataset,
    limit: int = 3,
) -> None:
    for index in range(min(limit, len(dataset))):
        metadata = dataset.metadata(index)
        if isinstance(dataset, FullOnlineTeacherDistillationDataset):
            frame_ids = [int(value) for value in metadata["frame_indices"]]
            print(
                "full-online clip audit: sequence={} start_frame={} frame_ids={} "
                "teacher_frame_ids_equal_student=true".format(
                    metadata["sequence_id"], metadata["clip_start"], frame_ids
                )
            )
            continue
        with np.load(str(dataset.cache_paths[index]), allow_pickle=False) as cache:
            teacher_start = int(cache["clip_start"].item())
            teacher_ids = [int(value) for value in cache["absolute_frame_ids"].tolist()]
        student_start = int(metadata["clip_start"])
        student_ids = [int(value) for value in metadata["frame_indices"]]
        if student_start != teacher_start or student_ids != teacher_ids:
            raise RuntimeError("Startup same-clip cache audit found a mapping mismatch")
        print(
            "same-clip audit: sequence={} student_start={} teacher_start={} "
            "absolute_ids={} cache={}".format(
                metadata["sequence_id"], student_start, teacher_start,
                student_ids, dataset.cache_paths[index],
            )
        )


def build_direct_distillation_optimizer(
    model: DA3SmallStudent, training_config: Dict[str, Any]
) -> torch.optim.AdamW:
    model.assert_trainability_contract()
    groups = model.parameter_groups()
    if groups["camera_encoder"]:
        raise RuntimeError("Inactive DA3 camera encoder entered the optimizer")
    head_parameters = groups["depth_head"] + groups["camera_decoder"]
    encoder_parameters = groups["backbone"]
    if not groups["depth_head"] or not groups["camera_decoder"]:
        raise RuntimeError("Optimizer requires trainable DA3 depth and camera-decoder parameters")
    head_learning_rate = float(training_config["learning_rate"])
    if head_learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be positive")
    parameter_groups = [
        {"params": head_parameters, "lr": head_learning_rate, "name": "heads"}
    ]
    if model.config.use_backbone_lora:
        if not encoder_parameters:
            raise RuntimeError("DINOv2 LoRA mode has no trainable adapter parameters")
        lora_learning_rate = float(
            training_config.get("lora_learning_rate", head_learning_rate)
        )
        if lora_learning_rate <= 0.0:
            raise ValueError("training.lora_learning_rate must be positive")
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": lora_learning_rate,
                "name": "backbone_lora",
            }
        )
    elif model.config.freeze_backbone:
        if encoder_parameters:
            raise RuntimeError("Frozen DA3 backbone parameters entered the optimizer")
    else:
        encoder_learning_rate = float(
            training_config.get("encoder_learning_rate", head_learning_rate * 0.1)
        )
        if encoder_learning_rate <= 0.0:
            raise ValueError("training.encoder_learning_rate must be positive")
        if not encoder_parameters:
            raise RuntimeError("Joint training has no DA3 backbone parameters")
        parameter_groups.append(
            {"params": encoder_parameters, "lr": encoder_learning_rate, "name": "encoder"}
        )
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(training_config.get("weight_decay", 0.05)),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    initial_learning_rate: float,
    minimum_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if initial_learning_rate <= 0:
        raise ValueError("initial_learning_rate must be positive")
    if not 0.0 <= minimum_learning_rate <= initial_learning_rate:
        raise ValueError("minimum_learning_rate must be between zero and initial_learning_rate")
    minimum_ratio = minimum_learning_rate / initial_learning_rate

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _amp_settings(
    training_config: Dict[str, Any], device: torch.device
) -> Tuple[bool, torch.dtype, bool]:
    enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    requested = str(training_config.get("amp_dtype", "auto")).lower()
    if requested not in {"auto", "float16", "bfloat16"}:
        raise ValueError("training.amp_dtype must be auto, float16, or bfloat16")
    bf16_supported = bool(
        device.type == "cuda"
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    if requested == "bfloat16" and enabled and not bf16_supported:
        raise RuntimeError("training.amp_dtype=bfloat16 but CUDA does not support BF16")
    dtype = (
        torch.bfloat16
        if enabled and (requested == "bfloat16" or (requested == "auto" and bf16_supported))
        else torch.float16
    )
    return enabled, dtype, enabled and dtype == torch.float16


def _teacher_amp_settings(
    teacher_config: Mapping[str, Any], device: torch.device
) -> Tuple[bool, torch.dtype]:
    enabled = bool(teacher_config.get("amp", True)) and device.type == "cuda"
    requested = str(teacher_config.get("amp_dtype", "auto")).lower()
    if requested == "auto":
        requested = (
            "bfloat16"
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else "float16"
        )
    dtypes = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if requested not in dtypes:
        raise ValueError("teacher.amp_dtype must be auto, bfloat16, or float16")
    if enabled and requested in {"bfloat16", "bf16"} and not torch.cuda.is_bf16_supported():
        raise RuntimeError("teacher.amp_dtype=bfloat16 but CUDA does not support BF16")
    return enabled, dtypes[requested]


def _slice_attention_features(
    features: Mapping[int, Mapping[str, Any]], start: int, stop: int
) -> Dict[int, Dict[str, Any]]:
    return {
        int(layer): {
            "q": feature["q"][start:stop],
            "k": feature["k"][start:stop],
            "metadata": dict(feature["metadata"]),
        }
        for layer, feature in features.items()
    }


def _compute_online_teacher_attention_loss(
    teacher_model: VGGTOmegaTeacher,
    teacher_images: torch.Tensor,
    student_features: Mapping[int, Mapping[str, Any]],
    loss_function: CrossFrameAttentionDistillationLoss,
    config: AttentionDistillationConfig,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Run frozen Teacher Q/K and relation loss chunk-by-chunk without caching."""
    first_student = next(iter(student_features.values()))["q"]
    temporal_length = int(first_student.shape[1])
    if tuple(teacher_images.shape[1:]) != (temporal_length, 3, 1024, 1280):
        raise RuntimeError(
            "Online Teacher batch must have shape [B,T,3,1024,1280]; got {}"
            .format(tuple(teacher_images.shape))
        )
    batch_size = int(teacher_images.shape[0])
    if batch_size <= 0:
        raise RuntimeError("Online Teacher batch is empty")
    first_student = next(iter(student_features.values()))["q"]
    if int(first_student.shape[0]) != batch_size:
        raise RuntimeError("Online Teacher RGB and Student attention batch sizes differ")

    total = first_student.new_zeros((), dtype=torch.float32)
    logs: Dict[str, float] = {}
    chunks = 0
    started = time.perf_counter()
    for start in range(0, batch_size, config.online_teacher_batch_size):
        stop = min(batch_size, start + config.online_teacher_batch_size)
        images = teacher_images[start:stop].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=amp_dtype,
        ):
            teacher_features = teacher_model.forward_attention(images)
        if any(
            feature[name].requires_grad
            for feature in teacher_features.values()
            for name in ("q", "k")
        ):
            raise RuntimeError("Online Teacher Q/K unexpectedly require gradients")
        student_chunk = _slice_attention_features(student_features, start, stop)
        # Keep QK^T, softmax, and JS outside the enclosing training autocast.
        # Teacher/Student forward activations remain mixed precision; only the
        # numerically sensitive relation objective is evaluated in FP32.
        with torch.autocast(device_type=device.type, enabled=False):
            chunk_loss, chunk_logs = loss_function(teacher_features, student_chunk)
        weight = float(stop - start) / float(batch_size)
        total = total + weight * chunk_loss
        for name, value in chunk_logs.items():
            logs[name] = logs.get(name, 0.0) + weight * float(value)
        chunks += 1
        del images, teacher_features, student_chunk, chunk_loss
    logs["loss/attention"] = float(total.detach().cpu())
    logs["stats/online_teacher_chunks"] = float(chunks)
    logs["timing/online_teacher_attention_seconds"] = time.perf_counter() - started
    return total, logs


def _forward_full_online_teacher(
    teacher_model: VGGTOmegaTeacher,
    teacher_images: torch.Tensor,
    absolute_frame_ids: torch.Tensor,
    clip_start: torch.Tensor,
    sequence_ids: list[str],
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    teacher_input_shape: tuple[int, int],
    supervision_shape: tuple[int, int],
    min_depth: float,
    max_depth: float,
    minimum_valid_fraction: float,
) -> tuple[
    Dict[str, Any], Optional[Dict[int, Dict[str, Any]]], Dict[str, Any]
]:
    """Run one frozen VGGT-Omega forward and reuse the raw-cache conversion."""
    if tuple(teacher_input_shape) != TEACHER_SHAPE:
        raise ValueError(
            "Full-online Teacher input must preserve the raw-cache native grid {}"
            .format(TEACHER_SHAPE)
        )
    if tuple(supervision_shape) != SUPERVISION_SHAPE:
        raise ValueError(
            "Full-online supervision must preserve the raw-cache grid {}"
            .format(SUPERVISION_SHAPE)
        )
    if absolute_frame_ids.ndim != 2 or int(absolute_frame_ids.shape[1]) < 2:
        raise RuntimeError("Teacher/Student frame IDs must have shape [B,T] with T >= 2")
    temporal_length = int(absolute_frame_ids.shape[1])
    expected = (temporal_length, 3, *teacher_input_shape)
    if tuple(teacher_images.shape[1:]) != expected:
        raise RuntimeError(
            "Full-online Teacher input must have shape [B,{}]; got {}".format(
                ",".join(str(value) for value in expected), tuple(teacher_images.shape)
            )
        )
    if int(teacher_images.shape[0]) != int(absolute_frame_ids.shape[0]):
        raise RuntimeError("Teacher RGB and absolute-frame-ID batch sizes differ")
    if torch.is_grad_enabled():
        raise RuntimeError("Full-online Teacher helper must run with gradients disabled")

    images = teacher_images.to(device, non_blocking=True)
    started = time.perf_counter()
    forward_count_before = getattr(teacher_model, "prediction_forward_count", None)
    with torch.autocast(
        device_type=device.type,
        enabled=amp_enabled,
        dtype=amp_dtype,
    ):
        raw_outputs = teacher_model(images)
    forward_count_after = getattr(teacher_model, "prediction_forward_count", None)
    if (
        forward_count_before is not None
        and forward_count_after is not None
        and int(forward_count_after) - int(forward_count_before) != 1
    ):
        raise RuntimeError("Expected exactly one VGGT-Omega prediction forward")
    attention = raw_outputs.pop("attention", None)
    adapted = adapt_teacher_outputs(
        {
            name: raw_outputs[name].float()
            for name in ("pose_enc", "depth", "depth_conf")
        },
        image_shape=teacher_input_shape,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    canonical = canonicalize_teacher_outputs(adapted)
    teacher = {
        key: canonical[key].detach()
        for key in (
            "depth", "confidence", "valid_mask", "intrinsics", "extrinsics"
        )
    }
    del raw_outputs, adapted, canonical, images
    if any(
        value.requires_grad
        for value in teacher.values()
        if isinstance(value, torch.Tensor)
    ) or (attention is not None and any(
        feature[name].requires_grad
        for feature in attention.values()
        for name in ("q", "k")
    )):
        raise RuntimeError("Full-online Teacher supervision unexpectedly requires gradients")
    if attention is not None and any(
        int(feature["metadata"].get("num_frames", -1)) != temporal_length
        for feature in attention.values()
    ):
        raise RuntimeError("Full-online Teacher attention frame count is incorrect")

    valid_fraction = teacher["valid_mask"].flatten(2).float().mean(2)
    if bool((valid_fraction < float(minimum_valid_fraction)).any()):
        raise RuntimeError(
            "Full-online Teacher valid fraction fell below {}: {}".format(
                minimum_valid_fraction, valid_fraction.detach().cpu().tolist()
            )
        )
    teacher["absolute_frame_ids"] = absolute_frame_ids.detach()
    teacher["clip_start"] = clip_start.detach()
    teacher["sequence_id"] = list(sequence_ids)
    audit = {
        "teacher_forward_count": 1,
        "teacher_input_shape": list(teacher_images.shape),
        "teacher_depth_shape": list(teacher["depth"].shape),
        "teacher_confidence_shape": list(teacher["confidence"].shape),
        "teacher_intrinsics_shape": list(teacher["intrinsics"].shape),
        "teacher_extrinsics_shape": list(teacher["extrinsics"].shape),
        "teacher_attention_layers": (
            sorted(int(layer) for layer in attention) if attention is not None else []
        ),
        "teacher_forward_seconds": time.perf_counter() - started,
    }
    return teacher, attention, audit


def _compute_attention_loss_from_online_features(
    teacher_features: Mapping[int, Mapping[str, Any]],
    student_features: Mapping[int, Mapping[str, Any]],
    loss_function: CrossFrameAttentionDistillationLoss,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Consume Q/K already produced by the same full-online Teacher forward."""
    started = time.perf_counter()
    first_student = next(iter(student_features.values()))["q"]
    first_teacher = next(iter(teacher_features.values()))["q"]
    if first_teacher.shape[:2] != first_student.shape[:2]:
        raise RuntimeError("Teacher/Student attention batch or frame counts differ")
    with torch.autocast(device_type=first_student.device.type, enabled=False):
        loss, logs = loss_function(teacher_features, student_features)
    logs = dict(logs)
    logs["loss/attention"] = float(loss.detach().cpu())
    logs["stats/online_teacher_chunks"] = 1.0
    logs["timing/online_teacher_attention_seconds"] = time.perf_counter() - started
    return loss, logs


def _audit_attention_backward(
    attention_loss: torch.Tensor,
    model: DA3SmallStudent,
    student_features: Mapping[int, Mapping[str, Any]],
    student_layers: tuple[int, ...],
    scaler: torch.cuda.amp.GradScaler,
) -> int:
    """Audit the same backward API used by training, then clear audit gradients.

    ``torch.autograd.grad`` does not execute exactly the same autograd nodes as
    ``backward`` and is a poor probe for retained non-leaf Q/K tensors behind
    many non-reentrant checkpoint regions.  The dry run therefore performs an
    attention-only scaled backward, inspects the gradients that training would
    actually accumulate, clears them, and leaves the retained graph available
    for the subsequent total-loss backward.
    """
    attention_parameters = [
        (name, parameter)
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    ]
    attention_tensors = [
        (
            "layer_{}_{}".format(layer, name),
            student_features[layer][name],
        )
        for layer in student_layers
        for name in ("q", "k")
    ]
    scaler.scale(attention_loss).backward(retain_graph=True)

    parameter_nonzero = [
        name
        for name, parameter in attention_parameters
        if parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool((parameter.grad.abs() > 0).any())
    ]
    qk_nonzero = [
        name
        for name, value in attention_tensors
        if value.grad is not None
        and bool(torch.isfinite(value.grad).all())
        and bool((value.grad.abs() > 0).any())
    ]
    parameter_none = sum(parameter.grad is None for _, parameter in attention_parameters)
    parameter_nonfinite = sum(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for _, parameter in attention_parameters
    )
    parameter_zero = sum(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and not bool((parameter.grad.abs() > 0).any())
        for _, parameter in attention_parameters
    )
    qk_none = sum(value.grad is None for _, value in attention_tensors)
    qk_nonfinite = sum(
        value.grad is not None and not bool(torch.isfinite(value.grad).all())
        for _, value in attention_tensors
    )
    qk_zero = sum(
        value.grad is not None
        and bool(torch.isfinite(value.grad).all())
        and not bool((value.grad.abs() > 0).any())
        for _, value in attention_tensors
    )
    if not parameter_nonzero or len(qk_nonzero) != len(attention_tensors):
        relation_stats: Dict[str, Dict[str, float]] = {}
        with torch.no_grad(), torch.autocast(
            device_type=attention_loss.device.type, enabled=False
        ):
            for layer in student_layers:
                feature = student_features[layer]
                q = feature["q"][:, 0, :, :16].float()
                k = feature["k"][:, 1].float()
                logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                    float(q.shape[-1])
                )
                probability = torch.softmax(logits, dim=-1)
                relation_stats[str(layer)] = {
                    "logit_min": float(logits.min().cpu()),
                    "logit_max": float(logits.max().cpu()),
                    "probability_min": float(probability.min().cpu()),
                    "probability_max": float(probability.max().cpu()),
                    "probability_zero_fraction": float(
                        (probability == 0).float().mean().cpu()
                    ),
                }
        raise RuntimeError(
            "L_attention backward audit failed: loss={:.9g} scale={:.9g} "
            "qk_nonzero={}/{} qk_none={} qk_zero={} qk_nonfinite={} "
            "parameter_nonzero={}/{} parameter_none={} parameter_zero={} "
            "parameter_nonfinite={} student_relation_stats={}".format(
                float(attention_loss.detach().cpu()),
                float(scaler.get_scale()),
                len(qk_nonzero),
                len(attention_tensors),
                qk_none,
                qk_zero,
                qk_nonfinite,
                len(parameter_nonzero),
                len(attention_parameters),
                parameter_none,
                parameter_zero,
                parameter_nonfinite,
                json.dumps(relation_stats, sort_keys=True),
            )
        )

    model.zero_grad(set_to_none=True)
    for _, value in attention_tensors:
        value.grad = None
    return len(parameter_nonzero)


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = dict(batch)
    for key in (
        "images", "clean_images", "highlight_masks", "absolute_frame_ids", "clip_start",
        "teacher_absolute_frame_ids",
    ):
        if key in batch:
            moved[key] = batch[key].to(device, non_blocking=True)
    if "teacher" in batch:
        teacher = dict(batch["teacher"])
        for key in (
            "depth", "confidence", "valid_mask", "intrinsics", "extrinsics",
            "absolute_frame_ids", "clip_start",
        ):
            teacher[key] = batch["teacher"][key].detach().to(device, non_blocking=True)
        moved["teacher"] = teacher
    return moved


def _prediction_is_finite(prediction: Dict[str, Any]) -> bool:
    def finite(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, Mapping):
            return all(finite(item) for item in value.values())
        return True

    return bool(prediction) and finite(prediction)


def _forward_with_fp32_retry(
    model: torch.nn.Module,
    images: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[Dict[str, torch.Tensor], bool, bool]:
    with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
        prediction = model(images, include_global_points=False)
    finite = _prediction_is_finite(prediction)
    if finite or not amp_enabled:
        return prediction, False, finite
    del prediction
    if images.device.type == "cuda":
        torch.cuda.empty_cache()
    with torch.cuda.amp.autocast(enabled=False):
        prediction = model(images.float(), include_global_points=False)
    return prediction, True, _prediction_is_finite(prediction)


def _append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _check_resume_contract(
    checkpoint: Dict[str, Any], config: Dict[str, Any], model: DA3SmallStudent
) -> None:
    require_training_objective(checkpoint, DIRECT_TEACHER_DISTILLATION_PROTOCOL)
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_full_online = _teacher_is_full_online(checkpoint_config)
    current_full_online = _teacher_is_full_online(config)
    if checkpoint_full_online != current_full_online:
        raise ValueError("Resume checkpoint Teacher mode differs from current config")
    if not current_full_online:
        require_student_cache_protocol(checkpoint, CROSSCLIP_CACHE_PROTOCOL)
    else:
        checkpoint_teacher = checkpoint_config.get("teacher", {})
        current_teacher = config.get("teacher", {})
        for field in ("mode", "use_cache", "input_height", "input_width"):
            if checkpoint_teacher.get(field) != current_teacher.get(field):
                raise ValueError(
                    "Resume checkpoint full-online Teacher setting differs: {}".format(field)
                )
        temporal_fields = ("clip_length", "sample_stride", "window_stride")
        for field in temporal_fields:
            if checkpoint_config.get("dataset", {}).get(field) != config.get("dataset", {}).get(field):
                raise ValueError(
                    "Resume checkpoint temporal setting differs: {}".format(field)
                )
    if (
        checkpoint_config.get("experiment", {}).get("objective_protocol")
        != DIRECT_TEACHER_DISTILLATION_PROTOCOL
    ):
        raise ValueError("Resume checkpoint config has an incompatible objective protocol")
    if checkpoint_config.get("loss", {}).get("mode") != "direct_teacher_distillation":
        raise ValueError("Resume checkpoint does not use direct_teacher_distillation loss")
    loss_fields = ("lambda_depth", "lambda_camera", "lambda_highlight", "lambda_smooth", "camera", "eps", "use_confidence_weight")
    loss_mismatches = {key: (checkpoint_config.get("loss", {}).get(key), config["loss"].get(key))
                       for key in loss_fields
                       if checkpoint_config.get("loss", {}).get(key) != config["loss"].get(key)}
    if loss_mismatches:
        raise ValueError("Checkpoint loss settings differ from current config: {}. Start a new run for a loss ablation.".format(loss_mismatches))
    for key, default in (("highlight_mode", "legacy"), ("highlight_cone_full_angle_degrees", 10.0), ("highlight_softness", 0.0005)):
        if checkpoint_config.get("loss", {}).get(key, default) != config["loss"].get(key, default):
            raise ValueError("Checkpoint highlight loss settings differ: {}. Start a new run.".format(key))
    checkpoint_student = checkpoint_config.get("student", {})
    if checkpoint_student.get("architecture") != "da3_small":
        raise ValueError("Resume checkpoint is not a DA3-Small experiment")
    current_student = config["student"]
    fields = (
        "freeze_backbone", "use_backbone_lora", "lora_rank", "lora_alpha",
        "lora_dropout", "lora_expected_modules", "freeze_depth_head",
        "freeze_camera_encoder", "freeze_camera_decoder",
    )
    mismatches = {
        field: (checkpoint_student.get(field), current_student.get(field))
        for field in fields
        if checkpoint_student.get(field) != current_student.get(field)
    }
    if mismatches:
        raise ValueError(
            "Checkpoint DA3/LoRA trainability does not match current config: {}. "
            "Start a new run.".format(mismatches)
        )
    checkpoint_attention = checkpoint_config.get("attention_distill", {})
    current_attention = config.get("attention_distill", {})
    checkpoint_attention_enabled = bool(checkpoint_attention.get("enabled", False))
    current_attention_enabled = bool(current_attention.get("enabled", False))
    attention_fields = (
        "enabled", "teacher_source", "teacher_output_dtype",
        "teacher_layers", "student_layers", "attention_type",
        "spatial_alignment", "common_grid", "head_aggregation", "divergence",
        "temperature_teacher", "temperature_student", "weight", "frame_offsets",
        "query_chunk_size", "eps",
    )
    attention_mismatches = {}
    if checkpoint_attention_enabled != current_attention_enabled:
        attention_mismatches["enabled"] = (
            checkpoint_attention_enabled,
            current_attention_enabled,
        )
    elif current_attention_enabled:
        attention_mismatches = {
            field: (checkpoint_attention.get(field), current_attention.get(field))
            for field in attention_fields
            if checkpoint_attention.get(field) != current_attention.get(field)
        }
    if attention_mismatches:
        raise ValueError(
            "Checkpoint attention-distillation settings differ from current config: {}. "
            "Start a new run.".format(attention_mismatches)
        )
    model.assert_trainability_contract()


def train_direct_teacher_distillation(
    config_path: Path,
    dry_run: bool = False,
    resume_override: Optional[Path] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    if config.get("experiment", {}).get("training_required") is False:
        raise ValueError("This baseline is inference-only; use its documented existing checkpoint")
    objective = config.get("experiment", {}).get("objective_protocol")
    if objective != DIRECT_TEACHER_DISTILLATION_PROTOCOL:
        raise ValueError(
            "experiment.objective_protocol must be {}".format(
                DIRECT_TEACHER_DISTILLATION_PROTOCOL
            )
        )
    teacher = config.get("teacher", {})
    full_online_teacher = _teacher_is_full_online(config)
    teacher_mode = str(teacher.get("mode", "cache")).lower()
    teacher_uses_cache = bool(teacher.get("use_cache", True))
    if (teacher_mode == "full_online") != (not teacher_uses_cache):
        raise ValueError(
            "teacher.mode=full_online and teacher.use_cache=false must be configured together"
        )
    if not full_online_teacher and str(teacher.get("cache_protocol")) != CROSSCLIP_CACHE_PROTOCOL:
        raise ValueError("teacher.cache_protocol must remain crossclip_local_v1")
    if str(teacher.get("variant")) != "base" or not bool(teacher.get("frozen", True)):
        raise ValueError("Direct distillation requires the frozen base teacher")
    attention_config = AttentionDistillationConfig.from_mapping(
        config.get("attention_distill", {})
    )
    if attention_config.enabled:
        configured_teacher_layers = tuple(
            int(value) for value in teacher.get("attention_layers", ())
        )
        if configured_teacher_layers != attention_config.teacher_layers:
            raise ValueError(
                "teacher.attention_layers must match attention_distill.teacher_layers"
            )
    dataset_config = config.get("dataset", {})
    configured_temporal = (
        int(dataset_config.get("clip_length", -1)),
        int(dataset_config.get("sample_stride", -1)),
        int(dataset_config.get("window_stride", -1)),
    )
    if full_online_teacher:
        valid_temporal = configured_temporal == (32, 2, 8)
        expected_temporal = (32, 2, 8)
    else:
        valid_temporal = configured_temporal == (16, 1, 8)
        expected_temporal = (16, 1, 8)
    if not valid_temporal:
        raise ValueError(
            "Dataset temporal settings must be {} for this Teacher mode".format(
                expected_temporal
            )
        )
    training_config = config.get("training", {})
    if full_online_teacher:
        if int(training_config.get("epochs", -1)) != 3:
            raise ValueError("Full-online B/C/D protocol requires training.epochs=3")
        if training_config.get("resume") is not None:
            raise ValueError("Full-online B/C/D protocol starts fresh with resume=null")
        teacher_grid = (
            int(teacher.get("input_height", TEACHER_SHAPE[0])),
            int(teacher.get("input_width", TEACHER_SHAPE[1])),
        )
        student_grid = (
            int(config.get("student", {}).get("image_height", -1)),
            int(config.get("student", {}).get("image_width", -1)),
        )
        if teacher_grid != TEACHER_SHAPE or student_grid != SUPERVISION_SHAPE:
            raise ValueError(
                "Online/cache parity requires Teacher {} and Student supervision {}"
                .format(TEACHER_SHAPE, SUPERVISION_SHAPE)
            )
    seed_everything(int(config.get("seed", 42)))
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    if dry_run and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if attention_config.enabled:
        print("Attention distillation mode: query_chunked")
        print(
            "Attention distillation query_chunk_size: {}".format(
                attention_config.query_chunk_size
            )
        )
    dataset = _build_dataset(config, "train")
    _print_same_clip_examples(dataset, limit=1)
    loader = build_direct_teacher_distillation_dataloader(
        dataset, config["dataloader"], int(config.get("seed", 42)), shuffle=True
    )
    model = DA3SmallStudent(
        config["student"],
        device=device,
        attention_config=config["attention_distill"],
    )
    model.train()
    online_teacher: Optional[VGGTOmegaTeacher] = None
    teacher_amp_enabled = False
    teacher_amp_dtype = torch.float16
    if full_online_teacher or attention_config.enabled:
        online_teacher_config = dict(teacher)
        online_teacher_config.update(
            {
                "save_attention": bool(attention_config.enabled),
                "attention_layers": (
                    list(attention_config.teacher_layers)
                    if attention_config.enabled
                    else list(teacher.get("attention_layers", ()))
                ),
                "attention_cache_dtype": attention_config.teacher_output_dtype,
                "attention_output_device": (
                    "cpu" if full_online_teacher and attention_config.enabled else "source"
                ),
                "attention_only": bool(attention_config.enabled and not full_online_teacher),
            }
        )
        online_teacher = VGGTOmegaTeacher.from_config(
            online_teacher_config, device=device
        )
        online_teacher.eval()
        if any(parameter.requires_grad for parameter in online_teacher.parameters()):
            raise RuntimeError("Online VGGT-Omega Teacher is not fully frozen")
        teacher_amp_enabled, teacher_amp_dtype = _teacher_amp_settings(teacher, device)
    if full_online_teacher:
        if online_teacher is None:
            raise RuntimeError("Full-online Teacher failed to initialize")
    loss_function = DirectTeacherDistillationLoss(config["loss"]).to(device)
    attention_loss_function = (
        CrossFrameAttentionDistillationLoss(attention_config).to(device)
        if attention_config.enabled
        else None
    )
    model.retain_attention_gradients(dry_run and attention_config.enabled)
    training_config = config["training"]
    diagnostics_every = int(training_config.get("regularizer_diagnostics_every", 100))
    if diagnostics_every < 0:
        raise ValueError("regularizer_diagnostics_every cannot be negative")
    timing_config = dict(training_config.get("timing", {}))
    timing_enabled = bool(timing_config.get("enabled", False))
    timing_log_every = int(timing_config.get("log_every_micro_batches", 1))
    if timing_enabled and device.type != "cuda":
        raise ValueError("training.timing requires a CUDA device")
    if timing_log_every <= 0:
        raise ValueError("training.timing.log_every_micro_batches must be positive")
    model.enable_cuda_timing(timing_enabled)
    optimizer = build_direct_distillation_optimizer(model, training_config)
    if online_teacher is not None:
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        teacher_parameter_ids = {id(parameter) for parameter in online_teacher.parameters()}
        if optimizer_parameter_ids & teacher_parameter_ids:
            raise RuntimeError("Frozen Teacher parameters entered the Student optimizer")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    accumulation = int(training_config.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    epochs = int(training_config.get("epochs", 20))
    updates_per_epoch = max(math.ceil(len(loader) / accumulation), 1)
    total_updates = updates_per_epoch * epochs
    warmup_steps = int(training_config.get("warmup_steps", 0))
    if warmup_steps == 0:
        warmup_steps = int(round(float(training_config.get("warmup_fraction", 0.05)) * total_updates))
    scheduler = _build_scheduler(
        optimizer,
        total_steps=total_updates,
        warmup_steps=warmup_steps,
        initial_learning_rate=float(training_config["learning_rate"]),
        minimum_learning_rate=float(training_config.get("min_learning_rate", 0.0)),
    )
    amp_enabled, amp_dtype, scaler_enabled = _amp_settings(training_config, device)
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    output_dir = ensure_dir(training_config["output_dir"])
    start_epoch = global_step = 0
    resume_value = resume_override or training_config.get("resume")
    if resume_value:
        checkpoint = torch.load(
            str(_project_path(resume_value)), map_location="cpu", weights_only=False
        )
        _check_resume_contract(checkpoint, config, model)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if checkpoint.get("loader_generator_state") is not None:
            loader.generator.set_state(checkpoint["loader_generator_state"])

    stats = model.parameter_statistics()
    clip_length = int(dataset_config["clip_length"])
    if full_online_teacher:
        first_metadata = dataset.metadata(0)
        first_frame_ids = [int(value) for value in first_metadata["frame_indices"]]
        baseline_id = str(config.get("experiment", {}).get("baseline_id", "unknown"))
        print(
            "STARTUP AUDIT baseline_id={} teacher_source={} teacher_mode=full_online "
            "teacher_frozen={} teacher_requires_grad={} teacher_in_optimizer=false "
            "clip_length={} sample_stride={} epochs={} "
            "student_rgb_shape=[B,{},3,{},{}] teacher_rgb_shape=[B,{},3,{},{}] "
            "absolute_frame_ids={} teacher_frame_ids_equal_student=true "
            "attention_enabled={} highlight_mode={} output_directory={}".format(
                baseline_id,
                teacher.get("pretrained_checkpoint"),
                bool(online_teacher is not None and not online_teacher.training),
                bool(
                    online_teacher is not None
                    and any(p.requires_grad for p in online_teacher.parameters())
                ),
                int(dataset.rgb_dataset.clip_length),
                int(dataset.rgb_dataset.sample_stride),
                epochs,
                clip_length,
                int(config["student"]["image_height"]),
                int(config["student"]["image_width"]),
                clip_length,
                int(teacher["input_height"]),
                int(teacher["input_width"]),
                first_frame_ids,
                attention_config.enabled,
                config.get("loss", {}).get("highlight_mode", "legacy"),
                output_dir,
            )
        )
    print(
        "VGGT-DA3 direct setup: clips={} batch={} frames={} input=448x560 "
        "sample_stride={} start_stride={} trainable={:,} backbone_trainable={:,} "
        "backbone_lora_trainable={:,} lora_modules={} depth_trainable={:,} "
        "camera_encoder_trainable={:,} camera_decoder_trainable={:,} "
        "ray_trainable={:,} attention_distill={} attention_source={} "
        "online_teacher_batch={} attention_weight={}".format(
            len(dataset), config["dataloader"]["batch_size"], clip_length,
            int(dataset.rgb_dataset.sample_stride), int(dataset.rgb_dataset.window_stride),
            stats["trainable"],
            stats["backbone_trainable"], stats["backbone_lora_trainable"],
            stats["lora_modules"], stats["depth_head_trainable"],
            stats["camera_encoder_trainable"], stats["camera_decoder_trainable"],
            stats["ray_trainable"],
            attention_config.enabled,
            attention_config.teacher_source,
            attention_config.online_teacher_batch_size,
            attention_config.weight,
        )
    )
    optimizer.zero_grad(set_to_none=True)
    last_logs: Dict[str, float] = {}
    if timing_enabled:
        print(
            "CUDA timing enabled; diagnostics are written to {}/timing.jsonl".format(
                output_dir
            )
        )

    def finish_timing(
        events: Dict[str, Optional[torch.cuda.Event]],
        *, epoch: int, batch_index: int, data_wait_seconds: float,
        iteration_start: float, optimizer_step: bool, retried: bool,
    ) -> None:
        if not timing_enabled:
            return
        torch.cuda.synchronize(device)
        forward_parts = model.forward_cuda_timings_ms()
        final_event = events.get("optimizer_end")
        if final_event is None:
            final_event = events.get("backward_end")
        record: Dict[str, Any] = {
            "phase": "timing", "epoch": epoch,
            "micro_batch": epoch * len(loader) + batch_index + 1,
            "batch_index": batch_index, "global_step": global_step,
            "optimizer_step": optimizer_step, "amp_fp32_retry": retried,
            "data_wait_ms": data_wait_seconds * 1000.0,
            "iteration_wall_ms": (time.perf_counter() - iteration_start) * 1000.0,
            "h2d_ms": _elapsed_ms(events.get("h2d_start"), events.get("h2d_end")),
            "forward_ms": _elapsed_ms(events.get("forward_start"), events.get("forward_end")),
            "loss_ms": _elapsed_ms(events.get("loss_start"), events.get("loss_end")),
            "backward_ms": _elapsed_ms(events.get("backward_start"), events.get("backward_end")),
            "optimizer_ms": _elapsed_ms(events.get("optimizer_start"), events.get("optimizer_end")),
            "gpu_pipeline_ms": _elapsed_ms(events.get("h2d_start"), final_event),
            **{"forward_{}_ms".format(name): value for name, value in forward_parts.items()},
        }
        _append_jsonl(output_dir / "timing.jsonl", record)
        if (batch_index + 1) % timing_log_every == 0:
            print(
                "TIMING " + " ".join(
                    "{}={:.3f}".format(key, value) if isinstance(value, float)
                    else "{}={}".format(key, value)
                    for key, value in record.items() if key != "phase"
                ),
                flush=True,
            )

    previous_iteration_end = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch_index, cpu_batch in enumerate(loader):
            iteration_start = time.perf_counter()
            data_wait_seconds = iteration_start - previous_iteration_end
            timing_events: Dict[str, Optional[torch.cuda.Event]] = {}
            timing_events["h2d_start"] = _record_cuda_event(timing_enabled)
            batch = _move_batch(cpu_batch, device)
            timing_events["h2d_end"] = _record_cuda_event(timing_enabled)
            full_online_attention: Optional[Dict[int, Dict[str, Any]]] = None
            full_online_audit: Dict[str, Any] = {}
            if full_online_teacher:
                if online_teacher is None or "teacher_images" not in batch:
                    raise RuntimeError("Full-online Teacher inputs are unavailable")
                if not torch.equal(
                    batch["absolute_frame_ids"], batch["teacher_absolute_frame_ids"]
                ):
                    raise RuntimeError("Full-online Teacher and Student frame IDs differ")
                if batch["sequence_id"] != batch["teacher_sequence_id"]:
                    raise RuntimeError("Full-online Teacher and Student sequence IDs differ")
                with torch.no_grad():
                    online_supervision, full_online_attention, full_online_audit = (
                        _forward_full_online_teacher(
                            online_teacher,
                            batch["teacher_images"],
                            batch["teacher_absolute_frame_ids"],
                            batch["clip_start"],
                            batch["teacher_sequence_id"],
                            device=device,
                            amp_enabled=teacher_amp_enabled,
                            amp_dtype=teacher_amp_dtype,
                            teacher_input_shape=(
                                int(teacher.get("input_height", TEACHER_SHAPE[0])),
                                int(teacher.get("input_width", TEACHER_SHAPE[1])),
                            ),
                            supervision_shape=(
                                int(model.config.image_height),
                                int(model.config.image_width),
                            ),
                            min_depth=float(teacher.get("min_depth", 0.1)),
                            max_depth=float(teacher.get("max_depth", 150.0)),
                            minimum_valid_fraction=float(
                                teacher.get("minimum_valid_fraction", 0.001)
                            ),
                        )
                    )
                batch["teacher"] = online_supervision
                if attention_config.enabled and full_online_attention is None:
                    raise RuntimeError("Enabled attention distillation did not capture Teacher Q/K")
                if not attention_config.enabled and full_online_attention is not None:
                    raise RuntimeError("Disabled attention distillation unexpectedly captured Teacher Q/K")
                if epoch == start_epoch and batch_index == 0:
                    print(
                        "STARTUP BATCH AUDIT sequence_id={} start_frame={} "
                        "frame_ids={} teacher_input_shape={} student_input_shape={} "
                        "teacher_depth_shape={} teacher_confidence_shape={} "
                        "teacher_intrinsics_shape={} teacher_extrinsics_shape={} "
                        "teacher_forward_count=1 attention_enabled={} "
                        "attention_layers={} "
                        "teacher_frame_ids_equal_student=true".format(
                            batch["sequence_id"],
                            batch["clip_start"].detach().cpu().tolist(),
                            batch["absolute_frame_ids"].detach().cpu().tolist(),
                            full_online_audit["teacher_input_shape"],
                            list(batch["images"].shape),
                            full_online_audit["teacher_depth_shape"],
                            full_online_audit["teacher_confidence_shape"],
                            full_online_audit["teacher_intrinsics_shape"],
                            full_online_audit["teacher_extrinsics_shape"],
                            attention_config.enabled,
                            full_online_audit["teacher_attention_layers"],
                        )
                    )
            timing_events["forward_start"] = _record_cuda_event(timing_enabled)
            prediction, retried, finite = _forward_with_fp32_retry(
                model, batch["images"], amp_enabled, amp_dtype
            )
            timing_events["forward_end"] = _record_cuda_event(timing_enabled)
            if not finite:
                raise FloatingPointError(
                    "Student output remained non-finite after FP32 retry at epoch={} batch={}".format(
                        epoch, batch_index
                    )
                )
            timing_events["loss_start"] = _record_cuda_event(timing_enabled)
            with torch.cuda.amp.autocast(enabled=amp_enabled and not retried, dtype=amp_dtype):
                baseline_loss, last_logs = loss_function(prediction, batch)
                attention_loss = baseline_loss.new_zeros(())
                if attention_loss_function is not None:
                    if full_online_teacher:
                        if full_online_attention is None:
                            raise RuntimeError("Same-forward Teacher Q/K are unavailable")
                        attention_loss, attention_logs = (
                            _compute_attention_loss_from_online_features(
                                full_online_attention,
                                prediction["attention"],
                                attention_loss_function,
                            )
                        )
                        attention_logs["stats/teacher_forward_count"] = 1.0
                        attention_logs["timing/full_online_teacher_forward_seconds"] = float(
                            full_online_audit["teacher_forward_seconds"]
                        )
                    else:
                        if online_teacher is None or "teacher_images" not in batch:
                            raise RuntimeError("Online Teacher attention inputs are unavailable")
                        attention_loss, attention_logs = _compute_online_teacher_attention_loss(
                            online_teacher,
                            batch["teacher_images"],
                            prediction["attention"],
                            attention_loss_function,
                            attention_config,
                            device,
                            teacher_amp_enabled,
                            teacher_amp_dtype,
                        )
                    last_logs.update(attention_logs)
                loss = baseline_loss + attention_config.weight * attention_loss
                last_logs["loss/baseline"] = float(baseline_loss.detach().cpu())
                last_logs["loss/attention"] = float(attention_loss.detach().cpu())
                last_logs["loss/attention_weighted"] = float(
                    (attention_config.weight * attention_loss).detach().cpu()
                )
                last_logs["loss/total"] = float(loss.detach().cpu())
            if full_online_attention is not None:
                # Q/K have already been reduced into the unchanged attention
                # loss; release the detached Teacher copies before backward.
                del full_online_attention
                full_online_attention = None
            last_logs.update(loss_share_logs(last_logs))
            if dry_run or (diagnostics_every and batch_index % diagnostics_every == 0):
                last_logs.update(regularizer_diagnostics(
                    prediction["xyz_local"], batch["highlight_masks"], batch["clean_images"],
                    eps=loss_function.config.eps))
            timing_events["loss_end"] = _record_cuda_event(timing_enabled)
            last_logs["stats/amp_fp32_retry"] = float(retried)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite direct distillation loss: {}".format(last_logs))
            if dry_run and attention_loss_function is not None:
                attention_gradient_tensors = _audit_attention_backward(
                    attention_loss,
                    model,
                    prediction["attention"],
                    attention_config.student_layers,
                    scaler,
                )
                last_logs["stats/attention_only_parameter_grad_tensors"] = float(
                    attention_gradient_tensors
                )
            window_start = (batch_index // accumulation) * accumulation
            window_size = 1 if dry_run else min(accumulation, len(loader) - window_start)
            timing_events["backward_start"] = _record_cuda_event(timing_enabled)
            scaler.scale(loss / window_size).backward()
            timing_events["backward_end"] = _record_cuda_event(timing_enabled)
            should_step = dry_run or ((batch_index + 1) % accumulation == 0) or (
                batch_index + 1 == len(loader)
            )
            if not should_step:
                finish_timing(
                    timing_events, epoch=epoch, batch_index=batch_index,
                    data_wait_seconds=data_wait_seconds, iteration_start=iteration_start,
                    optimizer_step=False, retried=retried,
                )
                previous_iteration_end = time.perf_counter()
                continue
            if dry_run:
                group_gradients = {
                    name: any(parameter.grad is not None for parameter in group)
                    for name, group in model.parameter_groups().items() if group
                }
                required_gradients = ("backbone", "depth_head", "camera_decoder")
                missing_gradients = [
                    name for name in required_gradients
                    if not group_gradients.get(name, False)
                ]
                if missing_gradients:
                    raise RuntimeError(
                        "Dry-run loss did not reach DA3 components {}".format(missing_gradients)
                    )
                if group_gradients.get("camera_encoder", False):
                    raise RuntimeError("Inactive camera encoder unexpectedly received gradients")
                if attention_config.enabled:
                    for layer in attention_config.student_layers:
                        feature = prediction["attention"][layer]
                        for name in ("q", "k"):
                            gradient = feature[name].grad
                            if gradient is None or not bool(torch.isfinite(gradient).all()):
                                raise RuntimeError(
                                    "Student attention {} at layer {} did not receive a finite gradient"
                                    .format(name.upper(), layer)
                                )
                            if not bool((gradient.abs() > 0).any()):
                                raise RuntimeError(
                                    "Student attention {} at layer {} received only zero gradient"
                                    .format(name.upper(), layer)
                                )
                shapes = {
                    key: (
                        {
                            layer: {
                                name: list(value.shape)
                                for name, value in feature.items()
                                if isinstance(value, torch.Tensor)
                            }
                            for layer, feature in value.items()
                        }
                        if key == "attention"
                        else list(value.shape)
                    )
                    for key, value in prediction.items()
                }
                print(
                    "VGGT-DA3 direct dry run passed: shapes={} teacher_depth_shape={} "
                    "gradients={} ray_forward_count={}".format(
                        shapes,
                        list(batch["teacher"]["depth"].shape),
                        group_gradients,
                        model._ray_forward_count,
                    )
                )
                finish_timing(
                    timing_events, epoch=epoch, batch_index=batch_index,
                    data_wait_seconds=data_wait_seconds, iteration_start=iteration_start,
                    optimizer_step=False, retried=retried,
                )
                if device.type == "cuda":
                    last_logs["memory/max_allocated_bytes"] = torch.cuda.max_memory_allocated(
                        device
                    )
                    last_logs["memory/max_reserved_bytes"] = torch.cuda.max_memory_reserved(
                        device
                    )
                    print(
                        "Dry-run CUDA peak: allocated={} reserved={} bytes".format(
                            last_logs["memory/max_allocated_bytes"],
                            last_logs["memory/max_reserved_bytes"],
                        )
                    )
                return {
                    "status": "passed", "output_shapes": shapes,
                    "gradient_components": group_gradients, **last_logs,
                }
            timing_events["optimizer_start"] = _record_cuda_event(timing_enabled)
            scaler.unscale_(optimizer)
            bad_gradients = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if bad_gradients:
                raise FloatingPointError("Non-finite gradients: {}".format(bad_gradients[:20]))
            clip = float(training_config.get("gradient_clip_norm", 1.0))
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(parameters, clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            timing_events["optimizer_end"] = _record_cuda_event(timing_enabled)
            global_step += 1
            finish_timing(
                timing_events, epoch=epoch, batch_index=batch_index,
                data_wait_seconds=data_wait_seconds, iteration_start=iteration_start,
                optimizer_step=True, retried=retried,
            )
            record = {
                "phase": "train", "epoch": epoch, "global_step": global_step,
                **{
                    "learning_rate/{}".format(group["name"]): group["lr"]
                    for group in optimizer.param_groups
                },
                **last_logs,
            }
            if global_step % int(training_config.get("log_every", 10)) == 0:
                print(" ".join("{}={}".format(key, value) for key, value in record.items()))
            _append_jsonl(output_dir / "metrics.jsonl", record)
            previous_iteration_end = time.perf_counter()
            if max_steps is not None and global_step >= max_steps:
                return {"status": "stopped", "global_step": global_step, **last_logs}

        state = {
            "objective_protocol": DIRECT_TEACHER_DISTILLATION_PROTOCOL,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "global_step": global_step, "config": config,
            "parameter_statistics": stats,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader_generator_state": loader.generator.get_state(),
        }
        atomic_torch_save(output_dir / "last.pt", state)
        save_every = int(training_config.get("save_every", 1))
        if save_every > 0 and (epoch + 1) % save_every == 0:
            atomic_torch_save(output_dir / "epoch_{:04d}.pt".format(epoch + 1), state)
    return {"status": "complete", "global_step": global_step, **last_logs}


__all__ = ["build_direct_distillation_optimizer", "train_direct_teacher_distillation"]
