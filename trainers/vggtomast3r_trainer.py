"""Trainer for the strict two-view DUNE-MASt3R V1 experiment."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np

from datasets.scared_pair_dataset import (
    ScaredPairDistillDataset,
    build_pair_dataloader,
    make_scared_pair_rgb_dataset,
)
from losses.vggtomast3r_loss import VggToMast3RLoss
from models.student.dune_mast3r_adapter import DuneMast3RStudent
from trainers.student_distillation_trainer import _amp_settings, _build_scheduler
from utils.checkpoint import atomic_torch_save
from utils.config import ensure_dir, load_config
from utils.seed import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _build_dataset(config: Dict[str, Any], split: str) -> ScaredPairDistillDataset:
    rgb = make_scared_pair_rgb_dataset(config["dataset"], split)
    cache_root = config.get("teacher", {}).get("cache_root")
    if not cache_root:
        raise ValueError("teacher.cache_root must be configured")
    dataset = ScaredPairDistillDataset(
        rgb,
        Path(cache_root) / split,
        config.get("dataset", {}).get("ground_truth"),
        expected_teacher_variant="lora",
        expected_lora_checkpoint=str(config.get("teacher", {}).get("lora_checkpoint", "")),
    )
    missing = dataset.missing_cache_paths(limit=5)
    if missing:
        raise FileNotFoundError(
            "Teacher pair caches are incomplete for {}: {}. Run generate_teacher_pair_cache.py first.".format(
                split, ", ".join(str(path) for path in missing)
            )
        )
    return dataset


def _move_target(target: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in target.items()}


def build_v1_optimizer(
    model: DuneMast3RStudent, training_config: Dict[str, Any]
) -> torch.optim.AdamW:
    model.assert_freeze_contract()
    dune_ids = {id(parameter) for parameter in model.dune_encoder.parameters()}
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Optimizer has no MASt3R decoder/head parameters")
    if any(id(parameter) in dune_ids for parameter in parameters):
        raise RuntimeError("Optimizer unexpectedly contains DUNE parameters")
    return torch.optim.AdamW(
        parameters,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.05)),
    )


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prediction_is_finite(prediction: Dict[str, torch.Tensor]) -> bool:
    if not prediction:
        return False
    checks = [torch.isfinite(value).all() for value in prediction.values()]
    return bool(torch.stack(checks).all())


def _tensor_numeric_summary(value: torch.Tensor) -> Dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite_fraction": float(finite.float().mean().cpu()),
        "finite_min": (
            float(finite_values.min().float().cpu())
            if finite_values.numel()
            else None
        ),
        "finite_max": (
            float(finite_values.max().float().cpu())
            if finite_values.numel()
            else None
        ),
    }


def _forward_with_fp32_retry(
    model: torch.nn.Module,
    images: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[Dict[str, torch.Tensor], bool, bool]:
    """Retry an AMP-only numeric failure without masking invalid outputs."""
    with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
        prediction = model(images)
    prediction_finite = _prediction_is_finite(prediction)
    if prediction_finite or not amp_enabled:
        return prediction, False, prediction_finite

    # The failed graph is never used for backward. Release it before the
    # higher-memory full-precision retry of the same batch.
    del prediction
    if images.device.type == "cuda":
        torch.cuda.empty_cache()
    with torch.cuda.amp.autocast(enabled=False):
        prediction = model(images.float())
    return prediction, True, _prediction_is_finite(prediction)


def train(
    config_path: Path,
    dry_run: bool = False,
    resume_override: Optional[Path] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    if not bool(config.get("teacher", {}).get("frozen", True)):
        raise ValueError("V1 requires a frozen teacher")
    seed_everything(int(config.get("seed", 42)))
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    dataset = _build_dataset(config, "train")
    loader = build_pair_dataloader(
        dataset, config["dataloader"], int(config.get("seed", 42)), shuffle=True
    )
    model = DuneMast3RStudent(config["student"], device=device)
    model.train()
    loss_function = VggToMast3RLoss(config["loss"]).to(device)
    training_config = config["training"]
    optimizer = build_v1_optimizer(model, training_config)
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
        checkpoint = torch.load(str(_project_path(resume_value)), map_location="cpu", weights_only=False)
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
        "V1 setup: pairs={} batch={} resolution=448x560 stride={} trainable={:,} frozen={:,} dune_frozen={:,}".format(
            len(dataset), config["dataloader"]["batch_size"], config["dataset"]["pair_stride"],
            stats["trainable"], stats["frozen"], stats["dune_frozen"],
        )
    )
    optimizer.zero_grad(set_to_none=True)
    last_logs: Dict[str, float] = {}
    for epoch in range(start_epoch, epochs):
        model.train()
        supervised_pixels_seen = False
        fp32_fallbacks = 0
        max_fp32_fallbacks = int(
            training_config.get("max_amp_fp32_fallbacks_per_epoch", 8)
        )
        if max_fp32_fallbacks < 0:
            raise ValueError(
                "training.max_amp_fp32_fallbacks_per_epoch cannot be negative"
            )
        for batch_index, batch in enumerate(loader):
            images = batch["images"].to(device, non_blocking=True)
            target = _move_target(batch["target"], device)
            gt = batch.get("ground_truth_depth_ref")
            gt_valid = batch.get("ground_truth_valid_mask_ref")
            if gt is not None:
                gt = gt.to(device, non_blocking=True)
                gt_valid = gt_valid.to(device, non_blocking=True)
            prediction, used_fp32_fallback, prediction_finite = _forward_with_fp32_retry(
                model, images, amp_enabled, amp_dtype
            )
            if used_fp32_fallback:
                fp32_fallbacks += 1
                event = {
                    "event": "amp_nonfinite_output_fp32_retry",
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "global_step": global_step,
                    "amp_dtype": str(amp_dtype),
                    "fp32_retry_finite": prediction_finite,
                    "sequence_id": batch.get("sequence_id"),
                    "frame_names": batch.get("frame_names"),
                }
                _append_jsonl(output_dir / "numeric_events.jsonl", event)
                print(
                    "Numeric warning: AMP output was non-finite; "
                    "retried batch in FP32 (epoch={} batch={} global_step={} "
                    "retry_finite={})".format(
                        epoch,
                        batch_index,
                        global_step,
                        event["fp32_retry_finite"],
                    ),
                    flush=True,
                )
            if not prediction_finite:
                diagnostic = {
                    "event": "nonfinite_output_after_fp32_retry",
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "global_step": global_step,
                    "amp_enabled": amp_enabled,
                    "amp_dtype": str(amp_dtype),
                    "sequence_id": batch.get("sequence_id"),
                    "frame_names": batch.get("frame_names"),
                    "images": _tensor_numeric_summary(images),
                    "prediction": {
                        name: _tensor_numeric_summary(value)
                        for name, value in prediction.items()
                    },
                    "nonfinite_parameters": [
                        name
                        for name, parameter in model.named_parameters()
                        if not torch.isfinite(parameter).all()
                    ][:20],
                }
                _append_jsonl(output_dir / "numeric_events.jsonl", diagnostic)
                raise FloatingPointError(
                    "Student output remained non-finite after FP32 retry at "
                    "epoch={} batch={} global_step={}; see {}".format(
                        epoch,
                        batch_index,
                        global_step,
                        output_dir / "numeric_events.jsonl",
                    )
                )
            if used_fp32_fallback and fp32_fallbacks > max_fp32_fallbacks:
                raise FloatingPointError(
                    "AMP required more than {} FP32 fallbacks in epoch {}; "
                    "stop and investigate numeric_events.jsonl".format(
                        max_fp32_fallbacks, epoch
                    )
                )
            with torch.cuda.amp.autocast(
                enabled=amp_enabled and not used_fp32_fallback,
                dtype=amp_dtype,
            ):
                loss, last_logs = loss_function(prediction, target, gt, gt_valid)
            last_logs["amp_fp32_fallback"] = float(used_fp32_fallback)
            supervised_pixels_seen = supervised_pixels_seen or (
                float(last_logs.get("supervised_depth_valid_fraction", 0.0)) > 0.0
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite V1 loss: {}".format(last_logs))
            window_start = (batch_index // accumulation) * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            scaler.scale(loss / window_size).backward()
            should_step = ((batch_index + 1) % accumulation == 0) or (batch_index + 1 == len(loader))
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            bad_gradients = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if bad_gradients:
                raise FloatingPointError("Non-finite gradients: {}".format(bad_gradients[:20]))
            clip = float(training_config.get("gradient_clip_norm", 1.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters, clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            record = {
                "phase": "train", "epoch": epoch, "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"], **last_logs,
            }
            if dry_run or global_step % int(training_config.get("log_every", 10)) == 0:
                print(" ".join("{}={}".format(key, value) for key, value in record.items()))
            if dry_run:
                if float(config["loss"].get("lambda_supervised_depth", 0.0)) > 0 and not supervised_pixels_seen:
                    raise RuntimeError("Dry run found no valid SCARED GT depth pixels")
                shapes = {key: list(value.shape) for key, value in prediction.items()}
                print("Dry run passed: pair/cache/model/forward/loss/backward/optimizer shapes={}".format(shapes))
                return {"status": "passed", "parameter_statistics": stats, "output_shapes": shapes, **last_logs}
            _append_jsonl(output_dir / "metrics.jsonl", record)
            if max_steps is not None and global_step >= max_steps:
                return {"status": "stopped", "global_step": global_step, **last_logs}

        if float(config["loss"].get("lambda_supervised_depth", 0.0)) > 0 and not supervised_pixels_seen:
            raise RuntimeError(
                "An entire training epoch had no valid SCARED GT depth pixels; "
                "check GT paths, millimetre-to-metre scale, and masks"
            )
        state = {
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
        if int(training_config.get("save_every", 1)) > 0 and (epoch + 1) % int(training_config.get("save_every", 1)) == 0:
            atomic_torch_save(output_dir / "epoch_{:04d}.pt".format(epoch + 1), state)
    return {"status": "complete", "global_step": global_step, **last_logs}


__all__ = ["build_v1_optimizer", "train"]
