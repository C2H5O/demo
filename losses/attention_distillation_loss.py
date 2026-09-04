"""Cross-frame relation distillation with patch-overlap spatial alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class AttentionDistillationConfig:
    enabled: bool
    teacher_layers: Tuple[int, ...]
    student_layers: Tuple[int, ...]
    attention_type: str
    spatial_alignment: str
    common_grid: str
    head_aggregation: str
    divergence: str
    temperature_teacher: float
    temperature_student: float
    weight: float
    frame_offsets: Tuple[int, ...]
    query_chunk_size: int
    eps: float

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "AttentionDistillationConfig":
        required = {
            "enabled",
            "teacher_layers",
            "student_layers",
            "attention_type",
            "spatial_alignment",
            "common_grid",
            "head_aggregation",
            "divergence",
            "temperature_teacher",
            "temperature_student",
            "weight",
            "frame_offsets",
            "query_chunk_size",
            "eps",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(
                "attention_distill config is missing explicit fields {}".format(missing)
            )
        result = cls(
            enabled=bool(config["enabled"]),
            teacher_layers=tuple(int(value) for value in config["teacher_layers"]),
            student_layers=tuple(int(value) for value in config["student_layers"]),
            attention_type=str(config["attention_type"]),
            spatial_alignment=str(config["spatial_alignment"]),
            common_grid=str(config["common_grid"]),
            head_aggregation=str(config["head_aggregation"]),
            divergence=str(config["divergence"]),
            temperature_teacher=float(config["temperature_teacher"]),
            temperature_student=float(config["temperature_student"]),
            weight=float(config["weight"]),
            frame_offsets=tuple(int(value) for value in config["frame_offsets"]),
            query_chunk_size=int(config["query_chunk_size"]),
            eps=float(config["eps"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.teacher_layers or len(self.teacher_layers) != len(self.student_layers):
            raise ValueError("attention_distill layer lists must be non-empty and equal length")
        if len(set(self.teacher_layers)) != len(self.teacher_layers):
            raise ValueError("attention_distill.teacher_layers contains duplicates")
        if len(set(self.student_layers)) != len(self.student_layers):
            raise ValueError("attention_distill.student_layers contains duplicates")
        if self.attention_type != "cross_frame_global":
            raise ValueError("Only cross_frame_global attention distillation is supported")
        if self.spatial_alignment != "patch_overlap" or self.common_grid != "student":
            raise ValueError("Attention alignment must be patch_overlap onto the student grid")
        if self.head_aggregation != "mean":
            raise ValueError("Only mean attention-head aggregation is supported")
        if self.divergence not in {"js", "kl"}:
            raise ValueError("attention_distill.divergence must be js or kl")
        if self.temperature_teacher <= 0.0 or self.temperature_student <= 0.0:
            raise ValueError("Attention temperatures must be positive")
        if self.weight < 0.0:
            raise ValueError("attention_distill.weight cannot be negative")
        if not self.frame_offsets or 0 in self.frame_offsets:
            raise ValueError("frame_offsets must contain non-zero temporal offsets")
        if self.query_chunk_size <= 0:
            raise ValueError("attention_distill.query_chunk_size must be positive")
        if self.eps <= 0.0:
            raise ValueError("attention_distill.eps must be positive")
        if not self.enabled and self.weight != 0.0:
            raise ValueError("Disabled attention distillation must have weight=0")
        if self.enabled and self.weight <= 0.0:
            raise ValueError("Enabled attention distillation requires a positive weight")


def _axis_overlap_matrix(
    source_count: int,
    target_count: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return target-by-source normalized 1-D patch-overlap weights."""
    if source_count <= 0 or target_count <= 0:
        raise ValueError("Patch-grid dimensions must be positive")
    source_left = torch.arange(source_count, device=device, dtype=torch.float64) / source_count
    source_right = (torch.arange(source_count, device=device, dtype=torch.float64) + 1) / source_count
    target_left = torch.arange(target_count, device=device, dtype=torch.float64) / target_count
    target_right = (torch.arange(target_count, device=device, dtype=torch.float64) + 1) / target_count
    overlap = torch.minimum(target_right[:, None], source_right[None, :]) - torch.maximum(
        target_left[:, None], source_left[None, :]
    )
    overlap = overlap.clamp_min(0.0)
    denominator = overlap.sum(dim=1, keepdim=True)
    if bool((denominator <= 0.0).any()):
        raise RuntimeError("Patch-overlap alignment produced an uncovered target token")
    return (overlap / denominator).to(dtype=dtype)


def patch_overlap_matrix(
    source_grid: Tuple[int, int],
    target_grid: Tuple[int, int],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the dense reference matrix used only by tests and diagnostics.

    Runtime alignment is separable and never materializes this potentially very
    large target-token by source-token matrix.
    """
    source_h, source_w = (int(value) for value in source_grid)
    target_h, target_w = (int(value) for value in target_grid)
    overlap_h = _axis_overlap_matrix(
        source_h, target_h, device=device, dtype=torch.float64
    )
    overlap_w = _axis_overlap_matrix(
        source_w, target_w, device=device, dtype=torch.float64
    )
    matrix = torch.einsum("ai,bj->abij", overlap_h, overlap_w).reshape(
        target_h * target_w, source_h * source_w
    )
    return matrix.to(dtype=dtype)


class SpatialTokenAligner(nn.Module):
    """Align only the spatial token axis; never resize an attention matrix."""

    def __init__(self, source_grid: Tuple[int, int], target_grid: Tuple[int, int]) -> None:
        super().__init__()
        self.source_grid = tuple(int(value) for value in source_grid)
        self.target_grid = tuple(int(value) for value in target_grid)
        source_h, source_w = self.source_grid
        target_h, target_w = self.target_grid
        if min(source_h, source_w, target_h, target_w) <= 0:
            raise ValueError("Patch grids must be positive")
        self._pool_factors: Tuple[int, int] | None = None
        if source_h % target_h == 0 and source_w % target_w == 0:
            self._pool_factors = (source_h // target_h, source_w // target_w)
            self.register_buffer("height_projection", None, persistent=False)
            self.register_buffer("width_projection", None, persistent=False)
        else:
            self.register_buffer(
                "height_projection",
                _axis_overlap_matrix(source_h, target_h),
                persistent=False,
            )
            self.register_buffer(
                "width_projection",
                _axis_overlap_matrix(source_w, target_w),
                persistent=False,
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError("Spatial Q/K tensor must have shape [B,F,H,N,D]")
        expected = self.source_grid[0] * self.source_grid[1]
        if int(value.shape[-2]) != expected:
            raise ValueError(
                "Spatial token count {} does not match source grid {}".format(
                    value.shape[-2], self.source_grid
                )
            )
        batch, frames, heads, _, head_dim = value.shape
        source_h, source_w = self.source_grid
        target_h, target_w = self.target_grid
        spatial = value.reshape(batch, frames, heads, source_h, source_w, head_dim)

        if self._pool_factors is not None:
            pool_h, pool_w = self._pool_factors
            channel_first = spatial.permute(0, 1, 2, 5, 3, 4).reshape(
                batch * frames * heads, head_dim, source_h, source_w
            )
            aligned = F.avg_pool2d(
                channel_first,
                kernel_size=(pool_h, pool_w),
                stride=(pool_h, pool_w),
            )
            return aligned.reshape(
                batch, frames, heads, head_dim, target_h, target_w
            ).permute(0, 1, 2, 4, 5, 3).reshape(
                batch, frames, heads, target_h * target_w, head_dim
            )

        height_projection = self.height_projection.to(
            device=value.device, dtype=value.dtype
        )
        width_projection = self.width_projection.to(
            device=value.device, dtype=value.dtype
        )
        width_aligned = torch.einsum(
            "vw,bfhuwd->bfhuvd", width_projection, spatial
        )
        aligned = torch.einsum(
            "su,bfhuvd->bfhsvd", height_projection, width_aligned
        )
        return aligned.reshape(batch, frames, heads, target_h * target_w, head_dim)


def _head_mean_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("Frame Q/K must have shape [B,H,N,D]")
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Frame Q/K heads or dimensions do not match")
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1))
    logits = logits / (math.sqrt(float(q.shape[-1])) * float(temperature))
    return torch.softmax(logits, dim=-1).mean(dim=1)


def _probability_divergence(
    teacher: torch.Tensor,
    student: torch.Tensor,
    kind: str,
    eps: float,
) -> torch.Tensor:
    teacher = teacher.detach().clamp_min(eps)
    student = student.clamp_min(eps)
    if kind == "kl":
        return (teacher * (teacher.log() - student.log())).sum(dim=-1)
    midpoint = (0.5 * (teacher + student)).clamp_min(eps)
    return 0.5 * (
        (teacher * (teacher.log() - midpoint.log())).sum(dim=-1)
        + (student * (student.log() - midpoint.log())).sum(dim=-1)
    )


def _metadata_grid(feature: Mapping[str, Any]) -> Tuple[int, int]:
    metadata = feature.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Attention feature is missing metadata")
    return int(metadata["patch_grid_h"]), int(metadata["patch_grid_w"])


def _metadata_image_extent(feature: Mapping[str, Any]) -> Tuple[int, int]:
    metadata = feature["metadata"]
    grid = _metadata_grid(feature)
    patch_size = int(metadata["patch_size"])
    return (
        int(metadata.get("image_height", grid[0] * patch_size)),
        int(metadata.get("image_width", grid[1] * patch_size)),
    )


class CrossFrameAttentionDistillationLoss(nn.Module):
    def __init__(self, config: AttentionDistillationConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = (
            AttentionDistillationConfig.from_mapping(config)
            if isinstance(config, Mapping)
            else config
        )
        self.config.validate()

    def _layer_loss(
        self,
        teacher: Mapping[str, Any],
        student: Mapping[str, Any],
    ) -> torch.Tensor:
        student_q, student_k = student["q"], student["k"]
        teacher_q = teacher["q"].detach().to(
            device=student_q.device, non_blocking=True
        )
        teacher_k = teacher["k"].detach().to(
            device=student_k.device, non_blocking=True
        )
        for name, value in (
            ("teacher Q", teacher_q),
            ("teacher K", teacher_k),
            ("student Q", student_q),
            ("student K", student_k),
        ):
            if value.ndim != 5:
                raise ValueError("{} must have shape [B,F,H,N,D]".format(name))
        if teacher_q.shape != teacher_k.shape or student_q.shape != student_k.shape:
            raise ValueError("Teacher or student Q/K shapes differ")
        if teacher_q.shape[:2] != student_q.shape[:2]:
            raise ValueError("Teacher/student batch or frame counts differ")
        teacher_grid, student_grid = _metadata_grid(teacher), _metadata_grid(student)
        teacher_extent = _metadata_image_extent(teacher)
        student_extent = _metadata_image_extent(student)
        if teacher_extent[0] * student_extent[1] != teacher_extent[1] * student_extent[0]:
            raise ValueError(
                "Teacher/student patch grids do not represent the same full-frame aspect ratio"
            )
        aligner = SpatialTokenAligner(teacher_grid, student_grid).to(teacher_q.device)
        teacher_q = aligner(teacher_q)
        teacher_k = aligner(teacher_k)
        expected_tokens = student_grid[0] * student_grid[1]
        if teacher_q.shape[-2] != expected_tokens or student_q.shape[-2] != expected_tokens:
            raise RuntimeError("Aligned Teacher and Student spatial token counts differ")

        total = student_q.new_zeros((), dtype=torch.float32)
        count = 0
        frames = int(student_q.shape[1])
        tokens = int(student_q.shape[-2])
        for source_frame in range(frames):
            for offset in self.config.frame_offsets:
                target_frame = source_frame + offset
                if not 0 <= target_frame < frames:
                    continue
                for start in range(0, tokens, self.config.query_chunk_size):
                    stop = min(tokens, start + self.config.query_chunk_size)
                    teacher_q_chunk = teacher_q[:, source_frame, :, start:stop]
                    teacher_k_frame = teacher_k[:, target_frame]
                    student_q_chunk = student_q[:, source_frame, :, start:stop]
                    student_k_frame = student_k[:, target_frame]

                    def chunk_sum(
                        tq: torch.Tensor,
                        tk: torch.Tensor,
                        sq: torch.Tensor,
                        sk: torch.Tensor,
                    ) -> torch.Tensor:
                        teacher_probability = _head_mean_attention(
                            tq, tk, self.config.temperature_teacher
                        )
                        student_probability = _head_mean_attention(
                            sq, sk, self.config.temperature_student
                        )
                        return _probability_divergence(
                            teacher_probability,
                            student_probability,
                            self.config.divergence,
                            self.config.eps,
                        ).sum()

                    if torch.is_grad_enabled() and (
                        student_q_chunk.requires_grad or student_k_frame.requires_grad
                    ):
                        value = checkpoint(
                            chunk_sum,
                            teacher_q_chunk,
                            teacher_k_frame,
                            student_q_chunk,
                            student_k_frame,
                            use_reentrant=False,
                        )
                    else:
                        value = chunk_sum(
                            teacher_q_chunk,
                            teacher_k_frame,
                            student_q_chunk,
                            student_k_frame,
                        )
                    total = total + value
                    count += int(student_q.shape[0]) * (stop - start)
        if count == 0:
            raise RuntimeError("No valid cross-frame attention pairs were produced")
        return total / float(count)

    def forward(
        self,
        teacher_features: Mapping[int, Mapping[str, Any]],
        student_features: Mapping[int, Mapping[str, Any]],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if not self.config.enabled:
            anchor = next(iter(student_features.values()))["q"]
            zero = anchor.reshape(-1)[0] * 0.0
            return zero, {"loss/attention": 0.0}
        layer_losses = []
        logs: Dict[str, float] = {}
        for teacher_layer, student_layer in zip(
            self.config.teacher_layers, self.config.student_layers
        ):
            if teacher_layer not in teacher_features:
                raise KeyError("Teacher attention cache lacks layer {}".format(teacher_layer))
            if student_layer not in student_features:
                raise KeyError("Student forward lacks layer {}".format(student_layer))
            layer_loss = self._layer_loss(
                teacher_features[teacher_layer], student_features[student_layer]
            )
            if not torch.isfinite(layer_loss):
                raise FloatingPointError(
                    "Non-finite attention loss for teacher {} -> student {}".format(
                        teacher_layer, student_layer
                    )
                )
            layer_losses.append(layer_loss)
            logs["loss/attn_t{}_s{}".format(teacher_layer, student_layer)] = float(
                layer_loss.detach().cpu()
            )
        total = torch.stack(layer_losses).mean()
        logs["loss/attention"] = float(total.detach().cpu())
        return total, logs


__all__ = [
    "AttentionDistillationConfig",
    "CrossFrameAttentionDistillationLoss",
    "SpatialTokenAligner",
    "patch_overlap_matrix",
]
