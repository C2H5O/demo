"""Train DUNE ViT-Small from offline frozen VGGT-Omega SCARED caches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from datasets.scared_clip_dataset import (
    ScaredDistillDataset,
    build_distill_dataloader,
    make_scared_rgb_dataset,
)
from losses.distillation_loss import ScaredDistillationLoss
from models.student.dune_model import DUNEViTSmallPointMapStudent
from utils.config import ensure_dir, load_config
from utils.seed import seed_everything


def _move_target(target: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in target.items()}


def _build_dataset(config: Dict[str, Any], split: str) -> ScaredDistillDataset:
    rgb_dataset = make_scared_rgb_dataset(config["dataset"], split)
    cache_root_value = config.get("teacher", {}).get("cache_root")
    if not cache_root_value:
        raise ValueError("teacher.cache_root must be configured")
    cache_root = Path(cache_root_value) / split
    dataset = ScaredDistillDataset(
        rgb_dataset,
        cache_root,
        config.get("dataset", {}).get("ground_truth"),
    )
    missing = dataset.missing_cache_paths(limit=5)
    if missing:
        raise FileNotFoundError(
            "Teacher caches are incomplete for split '{}'. First missing paths: {}. "
            "Run: python generate_teacher_cache.py --config {} --split {}".format(
                split,
                [str(path) for path in missing],
                "configs/scared_distill_pipeline.yaml",
                split,
            )
        )
    return dataset


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
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _atomic_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(state, temporary)
    temporary.replace(path)


def _append_log(path: Path, values: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(values, ensure_ascii=False) + "\n")


def _amp_settings(
    training_config: Dict[str, Any],
    device: torch.device,
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
        if enabled and (requested == "bfloat16" or (requested == "auto" and bf16_supported))
        else torch.float16
    )
    return enabled, dtype, enabled and dtype == torch.float16


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loss_function: ScaredDistillationLoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    batches = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        target = _move_target(batch["target"], device)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        ground_truth = batch.get("ground_truth_depth")
        ground_truth_valid = batch.get("ground_truth_valid_mask")
        if ground_truth is not None:
            ground_truth = ground_truth.to(device, non_blocking=True)
            ground_truth_valid = ground_truth_valid.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(
            enabled=amp_enabled, dtype=amp_dtype
        ):
            prediction = model(images)
        _, logs = loss_function(
            prediction,
            target,
            images,
            valid,
            ground_truth,
            ground_truth_valid,
        )
        for name, value in logs.items():
            totals[name] = totals.get(name, 0.0) + value
        batches += 1
    averages = {
        name: value / max(batches, 1) for name, value in totals.items()
    }
    if (
        loss_function.config.lambda_supervised_depth > 0
        and averages.get("supervised_depth_valid_fraction", 0.0) <= 0
    ):
        raise RuntimeError(
            "The entire validation split has no valid supervised GT pixels. "
            "Check dataset.ground_truth.scale and supervised depth bounds."
        )
    return averages


def train(
    config_path: Path,
    dry_run: bool = False,
    max_steps: Optional[int] = None,
    resume_override: Optional[Path] = None,
) -> None:
    config = load_config(config_path)
    if not bool(config.get("teacher", {}).get("frozen", True)):
        raise ValueError("Student distillation requires teacher.frozen=true")
    seed_everything(int(config.get("seed", 42)))
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device(requested_device)

    train_dataset = _build_dataset(config, "train")
    train_loader = build_distill_dataloader(
        train_dataset,
        config["dataloader"],
        seed=int(config.get("seed", 42)),
        shuffle=True,
    )
    evaluation_loader = None
    if bool(config["training"].get("use_test_for_evaluation", True)):
        evaluation_dataset = _build_dataset(config, "test")
        evaluation_loader = build_distill_dataloader(
            evaluation_dataset,
            config["dataloader"],
            seed=int(config.get("seed", 42)),
            shuffle=False,
        )

    model = DUNEViTSmallPointMapStudent(config["student"]).to(device)
    loss_function = ScaredDistillationLoss(config["loss"]).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial_learning_rate = float(config["training"]["learning_rate"])
    minimum_learning_rate = float(
        config["training"].get("min_learning_rate", 0.0)
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=initial_learning_rate,
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    epochs = int(config["training"]["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = _build_scheduler(
        optimizer,
        total_steps=max(updates_per_epoch * epochs, 1),
        warmup_steps=int(config["training"].get("warmup_steps", 0)),
        initial_learning_rate=initial_learning_rate,
        minimum_learning_rate=minimum_learning_rate,
    )
    amp_enabled, amp_dtype, scaler_enabled = _amp_settings(
        config["training"], device
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=scaler_enabled,
        init_scale=float(config["training"].get("amp_initial_scale", 128.0)),
        growth_interval=int(
            config["training"].get("amp_growth_interval", 2000)
        ),
    )
    output_dir = ensure_dir(config["training"]["output_dir"])
    log_path = output_dir / "metrics.jsonl"

    start_epoch = 0
    global_step = 0
    resume_value = resume_override or config["training"].get("resume")
    if resume_value:
        resume_path = Path(resume_value)
        checkpoint = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        print("Resumed from {} at epoch={} global_step={}".format(resume_path, start_epoch, global_step))

    print(
        "Training setup: train_clips={} test_clips={} batch_size={} frames={} "
        "trainable_params={:,} device={} learning_rate={:.2e} "
        "min_learning_rate={:.2e} scheduler=cosine total_updates={} "
        "amp={} amp_dtype={} grad_scaler={} scale={:.1f}".format(
            len(train_dataset),
            len(evaluation_loader.dataset) if evaluation_loader is not None else 0,
            config["dataloader"]["batch_size"],
            config["dataset"]["clip_length"],
            model.trainable_parameter_count(),
            device,
            initial_learning_rate,
            minimum_learning_rate,
            updates_per_epoch * epochs,
            amp_enabled,
            str(amp_dtype).replace("torch.", ""),
            scaler_enabled,
            scaler.get_scale(),
        )
    )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            target = _move_target(batch["target"], device)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            ground_truth = batch.get("ground_truth_depth")
            ground_truth_valid = batch.get("ground_truth_valid_mask")
            if ground_truth is not None:
                ground_truth = ground_truth.to(device, non_blocking=True)
                ground_truth_valid = ground_truth_valid.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(
                enabled=amp_enabled, dtype=amp_dtype
            ):
                prediction = model(images)
            loss, logs = loss_function(
                prediction,
                target,
                images,
                valid,
                ground_truth,
                ground_truth_valid,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss at epoch={} batch={}: {}".format(epoch, batch_index, logs))
            scaler.scale(loss / accumulation).backward()
            should_update = ((batch_index + 1) % accumulation == 0) or (batch_index + 1 == len(train_loader))
            if should_update:
                scaler.unscale_(optimizer)
                if dry_run:
                    non_finite_gradients = [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all()
                    ]
                    if non_finite_gradients:
                        raise FloatingPointError(
                            "Dry run found non-finite gradients: {}".format(
                                non_finite_gradients[:20]
                            )
                        )
                clip_norm = float(config["training"].get("gradient_clip_norm", 0.0))
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
                step_succeeded = True
                if not dry_run:
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    step_succeeded = scaler.get_scale() >= previous_scale
                    if step_succeeded:
                        scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if not step_succeeded:
                    print(
                        "Skipped optimizer/scheduler update after AMP "
                        "gradient overflow at epoch={} batch={} "
                        "scale={:.1f}->{:.1f}".format(
                            epoch,
                            batch_index,
                            previous_scale,
                            scaler.get_scale(),
                        )
                    )
                    continue
                global_step += 1

                record = {
                    "phase": "train",
                    "epoch": epoch,
                    "global_step": global_step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **logs,
                }
                if global_step % int(config["training"].get("log_every", 10)) == 0 or dry_run:
                    print(" ".join("{}={}".format(key, "{:.6f}".format(value) if isinstance(value, float) else value) for key, value in record.items()))
                if not dry_run:
                    _append_log(log_path, record)

                if dry_run:
                    print("Dry run passed: forward, all losses, backward, and gradients are finite. No optimizer step was written.")
                    return
                if max_steps is not None and global_step >= max_steps:
                    print("Stopped at requested max_steps={}".format(max_steps))
                    return

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
        }
        _atomic_checkpoint(output_dir / "last.pt", state)
        save_every = int(config["training"].get("save_every", 1))
        if save_every > 0 and (epoch + 1) % save_every == 0:
            epoch_path = output_dir / "epoch_{:04d}.pt".format(epoch + 1)
            _atomic_checkpoint(epoch_path, state)
            print(
                "Saved student checkpoints: {} and {}".format(
                    output_dir / "last.pt", epoch_path
                )
            )

        if evaluation_loader is not None and (epoch + 1) % int(config["training"].get("validate_every", 1)) == 0:
            validation_logs = validate(
                model,
                loss_function,
                evaluation_loader,
                device,
                amp_enabled,
                amp_dtype,
            )
            validation_record = {"phase": "test", "epoch": epoch, "global_step": global_step, **validation_logs}
            print(" ".join("{}={}".format(key, "{:.6f}".format(value) if isinstance(value, float) else value) for key, value in validation_record.items()))
            _append_log(log_path, validation_record)
    print("Training complete. Last checkpoint: {}".format(output_dir / "last.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/student_distillation.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run one forward/loss/backward without updating parameters")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer updates")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    train(Path(args.config), args.dry_run, args.max_steps, args.resume)


if __name__ == "__main__":
    main()
