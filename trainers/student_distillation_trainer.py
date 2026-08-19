"""Train the official Distill3R student from frozen VGGT-Omega caches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from datasets.scared_clip_dataset import (
    ScaredDistillDataset,
    build_distill_dataloader,
    make_scared_rgb_dataset,
)
from losses.distillation_loss import ScaredDistillationLoss
from models.student.distill3r_wrapper import Distill3RStudent
from utils.config import ensure_dir, load_config
from utils.seed import seed_everything


BRANCH0_DECONV_STATE_KEYS = {
    "student.downstream_head.dpt.act_postprocess.0.1.weight",
    "student.downstream_head.dpt.act_postprocess.0.1.bias",
    "student.downstream_head_local.dpt.act_postprocess.0.1.weight",
    "student.downstream_head_local.dpt.act_postprocess.0.1.bias",
}


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def load_bilinear_head_initialization(
    model: Distill3RStudent, checkpoint_path: str | Path
) -> None:
    """Load a baseline student while rejecting every unplanned incompatibility."""

    path = _resolve_project_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError("Initial student checkpoint does not exist: {}".format(path))
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, Mapping):
        raise TypeError("Initial student checkpoint does not contain a model state dictionary")
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    print("Initial checkpoint missing keys: {}".format(sorted(missing)))
    print("Initial checkpoint unexpected keys: {}".format(sorted(unexpected)))
    if missing:
        raise RuntimeError("Initial checkpoint has unplanned missing keys: {}".format(sorted(missing)))
    expected_unexpected = (
        BRANCH0_DECONV_STATE_KEYS
        if model.config.dpt_branch0_resize == "bilinear"
        else set()
    )
    if unexpected != expected_unexpected:
        raise RuntimeError(
            "Initial checkpoint has unplanned unexpected keys; expected={}, actual={}".format(
                sorted(expected_unexpected), sorted(unexpected)
            )
        )
    print("Loaded model weights from {}. Optimizer and scheduler state were not loaded.".format(path))


def validate_head_only_training(model: Distill3RStudent) -> Dict[str, int]:
    """Fail fast unless the complete Global/Local heads are the only trainable modules."""

    encoder = model.student.encoder
    decoder = model.student.decoder
    heads = model.dpt_heads()
    encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    head_trainable = {
        name: sum(p.numel() for p in head.parameters() if p.requires_grad)
        for name, head in heads
    }
    allowed_ids = {id(p) for _, head in heads for p in head.parameters()}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if encoder_trainable or decoder_trainable:
        raise RuntimeError("Encoder and decoder must have zero trainable parameters")
    if any(count <= 0 for count in head_trainable.values()):
        raise RuntimeError("Both complete DPT heads must contain trainable parameters")
    if trainable_ids != allowed_ids:
        raise RuntimeError(
            "Trainable parameter scope is not exactly the Global and Local DPT heads"
        )
    if encoder.training or decoder.training:
        raise RuntimeError("Frozen encoder and decoder must remain in eval mode")
    if any(not head.training for _, head in heads):
        raise RuntimeError("Global and Local DPT heads must remain in train mode")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts = {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "encoder_trainable": encoder_trainable,
        "decoder_trainable": decoder_trainable,
        "global_head_trainable": head_trainable["Global"],
        "local_head_trainable": head_trainable["Local"],
    }
    print("Parameter scope: total={:,} trainable={:,} frozen={:,}".format(total, trainable, total - trainable))
    print("DUNE Encoder: frozen, eval, trainable params = {}".format(encoder_trainable))
    print("Fast3R Decoder: frozen, eval, trainable params = {}".format(decoder_trainable))
    print("Global DPT Head: trainable params = {:,}".format(head_trainable["Global"]))
    print("Local DPT Head: trainable params = {:,}".format(head_trainable["Local"]))
    return counts


class BilinearHeadShapeCheck:
    """Capture and validate the first real training batch without another forward."""

    EXPECTED = {
        "branch0_projected": (96, 32, 40),
        "branch0_bilinear": (96, 128, 160),
        "branch1": (192, 64, 80),
        "branch2": (384, 32, 40),
        "branch3": (768, 16, 20),
        "scratch0": (256, 128, 160),
        "path1": (256, 256, 320),
        "raw_output": (4, 448, 560),
    }

    def __init__(self, model: Distill3RStudent) -> None:
        self.model = model
        self.handles: List[Any] = []
        self.shapes: Dict[str, Dict[str, List[Tuple[int, ...]]]] = {}

    def _hook(self, head_name: str, stage: str):
        def callback(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
            if not torch.is_tensor(output):
                raise TypeError("{} {} shape hook expected a tensor".format(head_name, stage))
            self.shapes.setdefault(head_name, {}).setdefault(stage, []).append(tuple(output.shape))

        return callback

    def register(self) -> None:
        for head_name, dpt in self.model.dpt_heads():
            modules = {
                "branch0_projected": dpt.act_postprocess[0][0],
                "branch0_bilinear": dpt.act_postprocess[0][1],
                "branch1": dpt.act_postprocess[1],
                "branch2": dpt.act_postprocess[2],
                "branch3": dpt.act_postprocess[3],
                "scratch0": dpt.scratch.layer_rn[0],
                "path1": dpt.scratch.refinenet1,
                "raw_output": dpt.head[4],
            }
            for stage, module in modules.items():
                self.handles.append(module.register_forward_hook(self._hook(head_name, stage)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def validate(self, prediction: Mapping[str, torch.Tensor], batch: int, views: int) -> None:
        try:
            for head_name, _ in self.model.dpt_heads():
                actual_stages = self.shapes.get(head_name, {})
                for stage, expected in self.EXPECTED.items():
                    calls = actual_stages.get(stage, [])
                    if not calls:
                        raise RuntimeError("{} {} did not run in the first batch".format(head_name, stage))
                    wrong = [shape for shape in calls if shape[-3:] != expected]
                    if wrong:
                        raise RuntimeError(
                            "{} {} expected [N,{}], got {}".format(
                                head_name, stage, ",".join(map(str, expected)), wrong
                            )
                        )
                    print("{} Head {}: {}".format(head_name, stage, list(calls[0])))
            expected_xyz = (batch, views, 448, 560, 3)
            for name in ("xyz_local", "xyz_global"):
                if tuple(prediction[name].shape) != expected_xyz:
                    raise RuntimeError("{} expected {}, got {}".format(name, expected_xyz, tuple(prediction[name].shape)))
                print("{}: {}".format(name, list(prediction[name].shape)))
            print("First-batch bilinear-head shape check passed")
        finally:
            self.remove()


class FrozenUpdateCheck:
    """Verify the frozen backbones and one gradient-bearing head tensor after step one."""

    def __init__(self, model: Distill3RStudent) -> None:
        self.model = model
        self.before: Dict[str, torch.Tensor] = {}
        self.head_parameter: Optional[torch.nn.Parameter] = None
        self.head_name: Optional[str] = None

    @staticmethod
    def _first_parameter(module: torch.nn.Module, label: str) -> torch.nn.Parameter:
        parameter = next(module.parameters(), None)
        if parameter is None:
            raise RuntimeError("{} has no parameter for update validation".format(label))
        return parameter

    def begin(self) -> None:
        encoder_parameter = self._first_parameter(self.model.student.encoder, "encoder")
        decoder_parameter = self._first_parameter(self.model.student.decoder, "decoder")
        candidates = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.detach().abs().max().item() > 0
        ]
        if not candidates:
            raise RuntimeError("No finite non-zero DPT-head gradient exists before the first optimizer step")
        self.head_name, self.head_parameter = max(
            candidates, key=lambda item: item[1].grad.detach().abs().max().item()
        )
        self.before = {
            "encoder": encoder_parameter.detach().clone(),
            "decoder": decoder_parameter.detach().clone(),
            "head": self.head_parameter.detach().clone(),
        }

    def finish(self) -> None:
        if self.head_parameter is None:
            raise RuntimeError("Frozen update check was not initialized")
        current = {
            "encoder": self._first_parameter(self.model.student.encoder, "encoder"),
            "decoder": self._first_parameter(self.model.student.decoder, "decoder"),
            "head": self.head_parameter,
        }
        changes = {
            name: float((current[name].detach() - previous).abs().max().item())
            for name, previous in self.before.items()
        }
        print(
            "First-step update check: encoder_parameter_max_abs_change={} "
            "decoder_parameter_max_abs_change={} head_parameter={} "
            "head_parameter_max_abs_change={}".format(
                changes["encoder"], changes["decoder"], self.head_name, changes["head"]
            )
        )
        if changes["encoder"] != 0.0 or changes["decoder"] != 0.0:
            raise RuntimeError("A frozen encoder/decoder parameter changed during optimizer step")
        if changes["head"] <= 0.0:
            raise RuntimeError("The selected gradient-bearing DPT-head parameter did not update")


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

    model = Distill3RStudent(config["student"]).to(device)
    training_config = config["training"]
    resume_value = resume_override or training_config.get("resume")
    initial_checkpoint = training_config.get("initial_checkpoint")
    if resume_value and initial_checkpoint:
        print(
            "Resume checkpoint {} takes precedence over model-only initialization {}".format(
                resume_value, initial_checkpoint
            )
        )
    elif initial_checkpoint:
        load_bilinear_head_initialization(model, initial_checkpoint)
    head_only_experiment = bool(
        config["student"].get("freeze_encoder", False)
        and config["student"].get("freeze_decoder", False)
        and config["student"].get("dpt_branch0_resize") == "bilinear"
    )
    if head_only_experiment and not resume_value and not initial_checkpoint:
        raise ValueError(
            "The bilinear-head experiment requires training.initial_checkpoint; "
            "random initialization is not allowed"
        )
    model.train()
    if head_only_experiment:
        for head_name, dpt in model.dpt_heads():
            print("{} Head branch0 resize: {}".format(head_name, dpt.act_postprocess[0][1]))
        validate_head_only_training(model)
    loss_function = ScaredDistillationLoss(config["loss"]).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("The optimizer has no trainable parameters")
    initial_learning_rate = float(training_config["learning_rate"])
    minimum_learning_rate = float(
        training_config.get("min_learning_rate", 0.0)
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=initial_learning_rate,
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    accumulation = int(training_config.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    epochs = int(training_config["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = _build_scheduler(
        optimizer,
        total_steps=max(updates_per_epoch * epochs, 1),
        warmup_steps=int(training_config.get("warmup_steps", 0)),
        initial_learning_rate=initial_learning_rate,
        minimum_learning_rate=minimum_learning_rate,
    )
    amp_enabled, amp_dtype, scaler_enabled = _amp_settings(
        training_config, device
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=scaler_enabled,
        init_scale=float(training_config.get("amp_initial_scale", 128.0)),
        growth_interval=int(
            training_config.get("amp_growth_interval", 2000)
        ),
    )
    output_dir = ensure_dir(training_config["output_dir"])
    log_path = output_dir / "metrics.jsonl"

    start_epoch = 0
    global_step = 0
    if resume_value:
        resume_path = _resolve_project_path(resume_value)
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
    shape_check: Optional[BilinearHeadShapeCheck] = None
    if head_only_experiment:
        shape_check = BilinearHeadShapeCheck(model)
        shape_check.register()
    update_check = FrozenUpdateCheck(model) if head_only_experiment else None
    update_check_complete = False
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
            if shape_check is not None:
                shape_check.validate(
                    prediction,
                    batch=int(images.shape[0]),
                    views=int(images.shape[1]),
                )
                shape_check = None
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
                    if update_check is not None and not update_check_complete:
                        update_check.begin()
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    step_succeeded = scaler.get_scale() >= previous_scale
                    if step_succeeded:
                        if update_check is not None and not update_check_complete:
                            update_check.finish()
                            update_check_complete = True
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
