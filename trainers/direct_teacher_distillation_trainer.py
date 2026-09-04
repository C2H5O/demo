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
from datasets.direct_teacher_distillation_dataset import (
    DirectTeacherDistillationDataset,
    build_direct_teacher_distillation_dataloader,
)
from losses.direct_teacher_distillation_loss import DirectTeacherDistillationLoss
from losses.attention_distillation_loss import (
    AttentionDistillationConfig,
    CrossFrameAttentionDistillationLoss,
)
from models.student.da3_small_student import DA3SmallStudent
from models.teacher.vggt_omega_wrapper import VGGTOmegaTeacher
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


def _build_dataset(
    config: Dict[str, Any], split: str
) -> DirectTeacherDistillationDataset:
    teacher = config["teacher"]
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
        expected_base_checkpoint=str(teacher["pretrained_checkpoint"]),
        online_teacher_attention=bool(
            config.get("attention_distill", {}).get("enabled", False)
        ),
    )
    print(
        "same-clip cache sampling: matched={} skipped_without_cache={} root={}".format(
            len(dataset), dataset.skipped_without_cache, cache_root
        )
    )
    return dataset


def _print_same_clip_examples(
    dataset: DirectTeacherDistillationDataset, limit: int = 3
) -> None:
    for index in range(min(limit, len(dataset))):
        metadata = dataset.metadata(index)
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
    if tuple(teacher_images.shape[1:]) != (16, 3, 1024, 1280):
        raise RuntimeError(
            "Online Teacher batch must have shape [B,16,3,1024,1280]; got {}"
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


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = dict(batch)
    for key in (
        "images", "clean_images", "highlight_masks", "absolute_frame_ids", "clip_start"
    ):
        moved[key] = batch[key].to(device, non_blocking=True)
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
    require_student_cache_protocol(checkpoint, CROSSCLIP_CACHE_PROTOCOL)
    checkpoint_config = checkpoint.get("config", {})
    if (
        checkpoint_config.get("experiment", {}).get("objective_protocol")
        != DIRECT_TEACHER_DISTILLATION_PROTOCOL
    ):
        raise ValueError("Resume checkpoint config has an incompatible objective protocol")
    if checkpoint_config.get("loss", {}).get("mode") != "direct_teacher_distillation":
        raise ValueError("Resume checkpoint does not use direct_teacher_distillation loss")
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
    objective = config.get("experiment", {}).get("objective_protocol")
    if objective != DIRECT_TEACHER_DISTILLATION_PROTOCOL:
        raise ValueError(
            "experiment.objective_protocol must be {}".format(
                DIRECT_TEACHER_DISTILLATION_PROTOCOL
            )
        )
    teacher = config.get("teacher", {})
    if str(teacher.get("cache_protocol")) != CROSSCLIP_CACHE_PROTOCOL:
        raise ValueError("teacher.cache_protocol must remain crossclip_local_v1")
    if str(teacher.get("variant")) != "base" or not bool(teacher.get("frozen", True)):
        raise ValueError("Direct distillation requires the frozen base teacher cache")
    attention_config = AttentionDistillationConfig.from_mapping(
        config.get("attention_distill", {})
    )
    if attention_config.enabled:
        if bool(teacher.get("save_attention", False)):
            raise ValueError(
                "Online attention distillation requires teacher.save_attention=false"
            )
        configured_teacher_layers = tuple(
            int(value) for value in teacher.get("attention_layers", ())
        )
        if configured_teacher_layers != attention_config.teacher_layers:
            raise ValueError(
                "teacher.attention_layers must match attention_distill.teacher_layers"
            )
    dataset_config = config.get("dataset", {})
    if (
        int(dataset_config.get("clip_length", -1)),
        int(dataset_config.get("sample_stride", -1)),
        int(dataset_config.get("window_stride", -1)),
    ) != (16, 1, 8):
        raise ValueError(
            "Dataset must use 16 consecutive frames and cache sampling stride 8"
        )
    seed_everything(int(config.get("seed", 42)))
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    dataset = _build_dataset(config, "train")
    _print_same_clip_examples(dataset)
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
    if attention_config.enabled:
        online_teacher_config = dict(teacher)
        online_teacher_config.update(
            {
                "save_attention": True,
                "attention_layers": list(attention_config.teacher_layers),
                "attention_cache_dtype": attention_config.teacher_output_dtype,
                "attention_output_device": "source",
                "attention_only": True,
            }
        )
        online_teacher = VGGTOmegaTeacher.from_config(
            online_teacher_config, device=device
        )
        online_teacher.eval()
        if any(parameter.requires_grad for parameter in online_teacher.parameters()):
            raise RuntimeError("Online VGGT-Omega Teacher is not fully frozen")
        teacher_amp_enabled, teacher_amp_dtype = _teacher_amp_settings(teacher, device)
    loss_function = DirectTeacherDistillationLoss(config["loss"]).to(device)
    attention_loss_function = (
        CrossFrameAttentionDistillationLoss(attention_config).to(device)
        if attention_config.enabled
        else None
    )
    model.retain_attention_gradients(dry_run and attention_config.enabled)
    training_config = config["training"]
    timing_config = dict(training_config.get("timing", {}))
    timing_enabled = bool(timing_config.get("enabled", False))
    timing_log_every = int(timing_config.get("log_every_micro_batches", 1))
    if timing_enabled and device.type != "cuda":
        raise ValueError("training.timing requires a CUDA device")
    if timing_log_every <= 0:
        raise ValueError("training.timing.log_every_micro_batches must be positive")
    model.enable_cuda_timing(timing_enabled)
    optimizer = build_direct_distillation_optimizer(model, training_config)
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
    print(
        "VGGT-DA3 direct setup: clips={} batch={} frames=16 input=448x560 "
        "cache_sampling_stride=8 trainable={:,} backbone_trainable={:,} "
        "backbone_lora_trainable={:,} lora_modules={} depth_trainable={:,} "
        "camera_encoder_trainable={:,} camera_decoder_trainable={:,} "
        "ray_trainable={:,} attention_distill={} attention_source={} "
        "online_teacher_batch={} attention_weight={}".format(
            len(dataset), config["dataloader"]["batch_size"], stats["trainable"],
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
                if dry_run and attention_loss_function is not None:
                    attention_parameters = [
                        (name, parameter)
                        for name, parameter in model.backbone.named_parameters()
                        if parameter.requires_grad
                    ]
                    attention_only_gradients = torch.autograd.grad(
                        attention_loss,
                        [parameter for _, parameter in attention_parameters],
                        retain_graph=True,
                        allow_unused=True,
                    )
                    attention_only_nonzero = [
                        name
                        for (name, _), gradient in zip(
                            attention_parameters, attention_only_gradients
                        )
                        if gradient is not None
                        and bool(torch.isfinite(gradient).all())
                        and bool((gradient.abs() > 0).any())
                    ]
                    if not attention_only_nonzero:
                        raise RuntimeError(
                            "L_attention did not produce a finite non-zero gradient on any "
                            "trainable DA3 backbone parameter"
                        )
                    last_logs["stats/attention_only_parameter_grad_tensors"] = float(
                        len(attention_only_nonzero)
                    )
                loss = baseline_loss + attention_config.weight * attention_loss
                last_logs["loss/baseline"] = float(baseline_loss.detach().cpu())
                last_logs["loss/attention"] = float(attention_loss.detach().cpu())
                last_logs["loss/attention_weighted"] = float(
                    (attention_config.weight * attention_loss).detach().cpu()
                )
                last_logs["loss/total"] = float(loss.detach().cpu())
            timing_events["loss_end"] = _record_cuda_event(timing_enabled)
            last_logs["stats/amp_fp32_retry"] = float(retried)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite direct distillation loss: {}".format(last_logs))
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
                    "VGGT-DA3 direct dry run passed: shapes={} gradients={} "
                    "ray_forward_count={}".format(
                        shapes, group_gradients, model._ray_forward_count
                    )
                )
                finish_timing(
                    timing_events, epoch=epoch, batch_index=batch_index,
                    data_wait_seconds=data_wait_seconds, iteration_start=iteration_start,
                    optimizer_step=False, retried=retried,
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
