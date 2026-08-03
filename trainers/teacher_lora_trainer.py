"""Train VGGT-Omega MLP LoRA with endoscopic temporal self-supervision."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.nn.parallel import DistributedDataParallel

from datasets.scared_clip_dataset import make_scared_rgb_dataset
from datasets.scared_dataset import build_scared_dataloader
from datasets.transforms import unnormalize_image
from losses.teacher_self_supervised import TeacherSelfSupervisedLoss
from models.teacher.lora_injection import (
    assert_only_lora_trainable,
    print_trainable_parameters,
    set_lora_training_mode,
)
from models.teacher.output_adapter import adapt_teacher_outputs
from models.teacher.vggt_omega_wrapper import VGGTOmegaTeacher
from utils.checkpoint import load_lora_checkpoint, save_lora_checkpoint
from utils.config import ensure_dir, load_config
from utils.distributed import DistributedContext, initialize_distributed
from utils.logger import JsonlLogger
from utils.seed import seed_everything


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(total_steps - warmup_steps, 1)
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _zero_one_video(images: torch.Tensor, normalize_mode: str) -> torch.Tensor:
    batch, frames = images.shape[:2]
    flattened = images.reshape(batch * frames, *images.shape[2:])
    restored = torch.stack(
        [unnormalize_image(image, normalize_mode) for image in flattened], dim=0
    )
    return restored.reshape(batch, frames, *restored.shape[1:])


def _float_prediction_tensors(
    predictions: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Promote floating predictions to FP32 without detaching their graph."""
    return {
        name: value.float() if torch.is_floating_point(value) else value
        for name, value in predictions.items()
    }


def _gradient_issues(
    model: torch.nn.Module,
) -> tuple[list[str], list[str]]:
    missing, non_finite = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            non_finite.append(name)
    return missing, non_finite


def _parameter_norm(
    parameters: list[torch.nn.Parameter],
) -> float:
    squares = [
        parameter.detach().float().square().sum()
        for parameter in parameters
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu())


def _lora_update_norm(model: torch.nn.Module) -> float:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith(".lora_B")
    ]
    return _parameter_norm(parameters)


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu())


class TeacherLoRATrainer:
    def __init__(
        self,
        config_path: Path,
        resume_override: Optional[Path] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        runtime = dict(self.config.get("runtime", {}))
        if bool(runtime.get("distributed", False)):
            self.distributed = initialize_distributed(
                str(runtime.get("backend", "nccl"))
            )
        else:
            self.distributed = DistributedContext(False, 0, 0, 1)
        seed = int(self.config.get("experiment", {}).get("seed", 42))
        seed_everything(seed + self.distributed.rank)

        requested_device = str(self.config.get("device", "cuda"))
        if self.distributed.enabled:
            requested_device = "cuda:{}".format(self.distributed.local_rank)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        self.device = torch.device(requested_device)

        dataset = make_scared_rgb_dataset(self.config["dataset"], "train")
        loader_config = dict(self.config.get("dataloader", {}))
        loader_config.setdefault(
            "batch_size", int(self.config["training"].get("batch_size", 1))
        )
        self.loader = build_scared_dataloader(
            dataset,
            batch_size=int(loader_config.get("batch_size", 1)),
            shuffle=True,
            num_workers=int(loader_config.get("num_workers", 0)),
            pin_memory=bool(loader_config.get("pin_memory", False)),
            persistent_workers=bool(loader_config.get("persistent_workers", False)),
            prefetch_factor=int(loader_config.get("prefetch_factor", 2)),
            drop_last=bool(loader_config.get("drop_last", False)),
            seed=seed,
            distributed=self.distributed.enabled,
            rank=self.distributed.rank,
            world_size=self.distributed.world_size,
        )

        teacher = VGGTOmegaTeacher.from_config(
            self.config["teacher"], device=self.device, load_lora=False
        )
        assert_only_lora_trainable(teacher)
        if self.distributed.enabled:
            self.teacher: torch.nn.Module = DistributedDataParallel(
                teacher,
                device_ids=[self.distributed.local_rank],
                output_device=self.distributed.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.teacher = teacher
        self.parameters = [
            parameter for parameter in self.teacher.parameters() if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"].get("weight_decay", 0.0)),
        )
        accumulation = int(
            self.config["training"].get("gradient_accumulation_steps", 1)
        )
        if accumulation <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.accumulation = accumulation
        epochs = int(self.config["training"]["epochs"])
        updates_per_epoch = math.ceil(len(self.loader) / accumulation)
        self.scheduler = _build_scheduler(
            self.optimizer,
            max(epochs * updates_per_epoch, 1),
            int(self.config["training"].get("warmup_steps", 0)),
        )
        self.amp_enabled = (
            bool(self.config["training"].get("mixed_precision", True))
            and self.device.type == "cuda"
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.loss_function = TeacherSelfSupervisedLoss(
            self.config.get("self_supervised_loss", {})
        ).to(self.device)
        self.start_epoch = 0
        self.global_step = 0

        resume = resume_override or self.config["training"].get("resume")
        if resume:
            state = load_lora_checkpoint(
                Path(resume),
                self.teacher_model,
                self.optimizer,
                self.scheduler,
                self.scaler,
            )
            self.start_epoch = int(state.get("epoch", -1)) + 1
            self.global_step = int(state.get("global_step", 0))

        output_dir = Path(self.config["experiment"]["output_dir"])
        self.output_dir = ensure_dir(output_dir) if self.distributed.is_main_process else output_dir
        self.logger = (
            JsonlLogger(self.output_dir / "metrics.jsonl")
            if self.distributed.is_main_process
            else None
        )
        if self.distributed.is_main_process:
            print_trainable_parameters(self.teacher_model)

    @property
    def teacher_model(self) -> VGGTOmegaTeacher:
        if isinstance(self.teacher, DistributedDataParallel):
            return self.teacher.module
        return self.teacher

    def _adapt(
        self,
        raw: Dict[str, torch.Tensor],
        image_shape: tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        teacher_config = self.config["teacher"]
        return adapt_teacher_outputs(
            raw,
            image_shape,
            min_depth=float(teacher_config.get("min_depth", 0.1)),
            max_depth=float(teacher_config.get("max_depth", 150.0)),
        )

    def _forward_loss(
        self, batch: Dict[str, Any]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        images = batch["images"].to(self.device, non_blocking=True)
        images = _zero_one_video(
            images, str(self.config["dataset"].get("normalize_mode", "imagenet"))
        )
        highlight_masks = batch.get("highlight_masks")
        inpainted_images = batch.get("inpainted_images")
        if highlight_masks is None or inpainted_images is None:
            raise RuntimeError(
                "Teacher training requires dataset.highlight.enabled=true"
            )
        highlight_masks = highlight_masks.to(self.device, non_blocking=True)
        inpainted_images = inpainted_images.to(self.device, non_blocking=True)
        # Keep the 1B-parameter adapted forward under AMP, but run camera
        # decoding, matrix inversion, temporal warping, surface normals, and
        # the final losses in FP32. Those operations have a much narrower
        # stable range than Transformer matmuls.
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            raw = self.teacher(images)
        with torch.cuda.amp.autocast(enabled=False):
            outputs = self._adapt(
                _float_prediction_tensors(raw), tuple(images.shape[-2:])
            )
        inpainted_outputs = None
        if self.loss_function.config.inpaint_consistency_weight > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    inpainted_raw = self.teacher(inpainted_images)
                with torch.cuda.amp.autocast(enabled=False):
                    inpainted_outputs = self._adapt(
                        _float_prediction_tensors(inpainted_raw),
                        tuple(inpainted_images.shape[-2:]),
                    )
        with torch.cuda.amp.autocast(enabled=False):
            loss, logs = self.loss_function(
                outputs,
                images.float(),
                outputs["intrinsics"],
                highlight_masks.float(),
                inpainted_images.float(),
                inpainted_outputs,
            )
        return loss, logs

    def save(self, path: Path, epoch: int) -> None:
        if not self.distributed.is_main_process:
            return
        save_lora_checkpoint(
            path,
            self.teacher_model,
            epoch,
            self.global_step,
            self.optimizer,
            self.scheduler,
            self.config,
            self.scaler,
        )

    def train(
        self,
        dry_run: bool = False,
        max_steps: Optional[int] = None,
    ) -> None:
        epochs = int(self.config["training"]["epochs"])
        log_every = int(self.config["training"].get("log_every", 10))
        clip_norm = float(
            self.config["training"].get("gradient_clip_norm", 0.0)
        )
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.start_epoch, epochs):
            sampler = getattr(self.loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            set_lora_training_mode(self.teacher_model)
            for batch_index, batch in enumerate(self.loader):
                loss, logs = self._forward_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite teacher loss at epoch={} batch={}: {}".format(
                            epoch, batch_index, logs
                        )
                    )
                normalized_loss = loss / self.accumulation
                if dry_run:
                    # A dry run is a numerical/graph audit. Scaling a single
                    # first step can itself overflow and obscure whether the
                    # underlying gradients are valid.
                    normalized_loss.backward()
                else:
                    self.scaler.scale(normalized_loss).backward()
                should_update = (
                    (batch_index + 1) % self.accumulation == 0
                    or batch_index + 1 == len(self.loader)
                )
                if not should_update:
                    continue
                if not dry_run:
                    self.scaler.unscale_(self.optimizer)
                missing_gradients, non_finite_gradients = _gradient_issues(
                    self.teacher_model
                )
                if missing_gradients:
                    raise FloatingPointError(
                        "Missing LoRA gradients ({} total; first 20 shown): {}. "
                        "The loss graph is disconnected from these adapters.".format(
                            len(missing_gradients), missing_gradients[:20]
                        )
                    )
                if non_finite_gradients:
                    raise FloatingPointError(
                        "Non-finite LoRA gradients ({} total; first 20 shown): {}. "
                        "Loss diagnostics: {}".format(
                            len(non_finite_gradients),
                            non_finite_gradients[:20],
                            logs,
                        )
                    )
                if clip_norm > 0:
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters, clip_norm
                        ).detach().cpu()
                    )
                else:
                    gradient_norm = _gradient_norm(self.parameters)
                if not dry_run:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                record = {
                    "phase": "train",
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "gradient_norm": gradient_norm,
                    "lora_parameter_norm": _parameter_norm(self.parameters),
                    "lora_update_norm": _lora_update_norm(self.teacher_model),
                    **logs,
                }
                if self.distributed.is_main_process and (
                    self.global_step % log_every == 0 or dry_run
                ):
                    print(
                        " ".join(
                            "{}={}".format(
                                key,
                                "{:.6f}".format(value)
                                if isinstance(value, float)
                                else value,
                            )
                            for key, value in record.items()
                        )
                    )
                if self.logger is not None and not dry_run:
                    self.logger.log(record)
                if dry_run:
                    print(
                        "Teacher dry run passed: forward, all losses, backward, "
                        "and LoRA gradients are finite. No optimizer update was written."
                    )
                    return
                if max_steps is not None and self.global_step >= max_steps:
                    self.save(self.output_dir / "last.pt", epoch)
                    return
            self.save(self.output_dir / "last.pt", epoch)
            save_every = int(self.config["training"].get("save_every", 1))
            if save_every > 0 and (epoch + 1) % save_every == 0:
                self.save(
                    self.output_dir / "epoch_{:04d}.pt".format(epoch + 1), epoch
                )


def train(
    config_path: Path,
    dry_run: bool = False,
    max_steps: Optional[int] = None,
    resume_override: Optional[Path] = None,
) -> None:
    TeacherLoRATrainer(config_path, resume_override).train(dry_run, max_steps)
