"""Trainer for frozen-DUNE cross-clip teacher projection."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from datasets.crossclip_teacher_dataset import (
    CROSSCLIP_CACHE_PROTOCOL,
    ScaredCrossClipProjectionDataset,
    build_crossclip_projection_dataloader,
    make_crossclip_rgb_dataset,
)
from losses.crossclip_projection_loss import CrossClipProjectionLoss
from models.student.dune_fast3r_head import DuneFast3RHeadStudent
from utils.checkpoint import atomic_torch_save, require_student_cache_protocol
from utils.config import ensure_dir, load_config
from utils.seed import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _build_dataset(
    config: Dict[str, Any], split: str
) -> ScaredCrossClipProjectionDataset:
    rgb = make_crossclip_rgb_dataset(config["dataset"], split)
    teacher = config["teacher"]
    use_aligned = bool(teacher.get("use_aligned_cache", True))
    root_key = "aligned_cache_root" if use_aligned else "raw_cache_root"
    if not teacher.get(root_key):
        raise ValueError("teacher.{} must be configured".format(root_key))
    dataset = ScaredCrossClipProjectionDataset(
        rgb,
        Path(str(teacher[root_key])) / split,
        expected_base_checkpoint=str(teacher["pretrained_checkpoint"]),
        expected_stage="aligned" if use_aligned else "raw",
    )
    missing = dataset.missing_neighbor_cache_paths(limit=5)
    if missing:
        raise FileNotFoundError(
            "Cross-clip neighbor caches are incomplete: {}".format(
                ", ".join(str(path) for path in missing)
            )
        )
    return dataset


def build_crossclip_optimizer(
    model: DuneFast3RHeadStudent, training_config: Dict[str, Any]
) -> torch.optim.AdamW:
    model.assert_freeze_contract()
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Optimizer has no Fast3R DPT head parameters")
    if any(id(parameter) in encoder_ids for parameter in parameters):
        raise RuntimeError("Optimizer unexpectedly contains frozen DUNE parameters")
    return torch.optim.AdamW(
        parameters,
        lr=float(training_config["learning_rate"]),
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
        raise ValueError(
            "minimum_learning_rate must be between zero and initial_learning_rate"
        )
    minimum_ratio = minimum_learning_rate / initial_learning_rate

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(total_steps - warmup_steps, 1)
        )
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )
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
        raise RuntimeError(
            "training.amp_dtype=bfloat16 but this CUDA device does not support BF16"
        )
    dtype = (
        torch.bfloat16
        if enabled
        and (requested == "bfloat16" or (requested == "auto" and bf16_supported))
        else torch.float16
    )
    return enabled, dtype, enabled and dtype == torch.float16


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = dict(batch)
    for key in (
        "images",
        "clean_images",
        "highlight_masks",
        "frame_indices",
        "absolute_frame_ids",
        "clip_start",
    ):
        moved[key] = batch[key].to(device, non_blocking=True)
    for side_name in ("teacher_left", "teacher_right"):
        side = dict(batch[side_name])
        for key, value in side.items():
            if torch.is_tensor(value):
                # Teacher cache is immutable pseudo-label data and must never
                # build a gradient graph during student training.
                side[key] = value.detach().to(device, non_blocking=True)
        moved[side_name] = side
    return moved


def _prediction_is_finite(prediction: Dict[str, torch.Tensor]) -> bool:
    return bool(prediction) and all(
        bool(torch.isfinite(value).all()) for value in prediction.values()
    )


def _forward_with_fp32_retry(
    model: torch.nn.Module,
    images: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[Dict[str, torch.Tensor], bool, bool]:
    with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
        prediction = model(images)
    finite = _prediction_is_finite(prediction)
    if finite or not amp_enabled:
        return prediction, False, finite
    del prediction
    if images.device.type == "cuda":
        torch.cuda.empty_cache()
    with torch.cuda.amp.autocast(enabled=False):
        prediction = model(images.float())
    return prediction, True, _prediction_is_finite(prediction)


def _append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def train_crossclip_projection(
    config_path: Path,
    dry_run: bool = False,
    resume_override: Optional[Path] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    teacher = config.get("teacher", {})
    if str(teacher.get("cache_protocol")) != CROSSCLIP_CACHE_PROTOCOL:
        raise ValueError("teacher.cache_protocol must be crossclip_local_v1")
    if str(teacher.get("variant")) != "base":
        raise ValueError("This experiment requires the frozen base teacher")
    if not bool(teacher.get("frozen", True)):
        raise ValueError("Teacher must be frozen")
    dataset_config = config.get("dataset", {})
    if not bool(dataset_config.get("random_clip_sampling", True)):
        raise ValueError("Training must randomly sample legal clip-start IDs")
    if int(dataset_config.get("teacher_neighbor_offset", 1)) != 1:
        raise ValueError("teacher_neighbor_offset must be one clip start")
    seed_everything(int(config.get("seed", 42)))
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    dataset = _build_dataset(config, "train")
    loader = build_crossclip_projection_dataloader(
        dataset,
        config["dataloader"],
        int(config.get("seed", 42)),
        shuffle=True,
    )
    model = DuneFast3RHeadStudent(config["student"], device=device)
    model.train()
    loss_function = CrossClipProjectionLoss(config["loss"]).to(device)
    training_config = config["training"]
    optimizer = build_crossclip_optimizer(model, training_config)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    accumulation = int(training_config.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    epochs = int(training_config.get("epochs", 20))
    updates_per_epoch = max(math.ceil(len(loader) / accumulation), 1)
    total_updates = updates_per_epoch * epochs
    warmup_steps = int(training_config.get("warmup_steps", 0))
    if warmup_steps == 0:
        warmup_steps = int(
            round(float(training_config.get("warmup_fraction", 0.05)) * total_updates)
        )
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
        require_student_cache_protocol(checkpoint, CROSSCLIP_CACHE_PROTOCOL)
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
        "Cross-clip setup: clips={} batch={} frames=16 layers=[2,5,8,11] "
        "trainable={:,} dune_frozen={:,}".format(
            len(dataset), config["dataloader"]["batch_size"], stats["trainable"], stats["dune_frozen"]
        )
    )
    optimizer.zero_grad(set_to_none=True)
    last_logs: Dict[str, float] = {}
    consecutive_zero_projection_batches = 0
    max_zero_projection_batches = int(
        training_config.get("max_consecutive_zero_projection_batches", 5)
    )
    if max_zero_projection_batches <= 0:
        raise ValueError("max_consecutive_zero_projection_batches must be positive")
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch_index, cpu_batch in enumerate(loader):
            batch = _move_batch(cpu_batch, device)
            prediction, retried, finite = _forward_with_fp32_retry(
                model, batch["images"], amp_enabled, amp_dtype
            )
            if not finite:
                raise FloatingPointError(
                    "Student output remained non-finite after FP32 retry at epoch={} batch={}".format(
                        epoch, batch_index
                    )
                )
            with torch.cuda.amp.autocast(
                enabled=amp_enabled and not retried, dtype=amp_dtype
            ):
                loss, last_logs = loss_function(prediction, batch)
            last_logs["stats/amp_fp32_retry"] = float(retried)
            has_projection = (
                last_logs["stats/proj_left_valid_ratio"] > 0.0
                or last_logs["stats/proj_right_valid_ratio"] > 0.0
            )
            consecutive_zero_projection_batches = (
                0 if has_projection else consecutive_zero_projection_batches + 1
            )
            if consecutive_zero_projection_batches >= max_zero_projection_batches:
                raise RuntimeError(
                    "No valid student-to-teacher projection for {} consecutive "
                    "batches; positive_depth_ratio={} depth_range=[{},{}]. "
                    "Training was stopped to prevent a silent zero-loss run.".format(
                        consecutive_zero_projection_batches,
                        last_logs["stats/student_positive_depth_ratio"],
                        last_logs["stats/student_depth_min"],
                        last_logs["stats/student_depth_max"],
                    )
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite cross-clip loss: {}".format(last_logs))
            window_start = (batch_index // accumulation) * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            scaler.scale(loss / window_size).backward()
            should_step = ((batch_index + 1) % accumulation == 0) or (
                batch_index + 1 == len(loader)
            )
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            bad_gradients = [
                name
                for name, parameter in model.named_parameters()
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
            global_step += 1
            record = {
                "phase": "train",
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **last_logs,
            }
            if dry_run or global_step % int(training_config.get("log_every", 10)) == 0:
                print(" ".join("{}={}".format(key, value) for key, value in record.items()))
            if dry_run:
                shapes = {key: list(value.shape) for key, value in prediction.items()}
                print("Cross-clip dry run passed: {}".format(shapes))
                return {"status": "passed", "output_shapes": shapes, **last_logs}
            _append_jsonl(output_dir / "metrics.jsonl", record)
            if max_steps is not None and global_step >= max_steps:
                return {"status": "stopped", "global_step": global_step, **last_logs}

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
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


__all__ = ["build_crossclip_optimizer", "train_crossclip_projection"]
