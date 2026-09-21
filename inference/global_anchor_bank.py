"""Deterministic sequence-wide DINO patch descriptors and diverse anchor selection."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def farthest_point_indices(descriptors: np.ndarray, count: int) -> list[int]:
    """Cosine FPS, seeded at input row zero; np.argmax breaks ties by row order."""
    features = np.asarray(descriptors, dtype=np.float64)
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError("Descriptors must be a nonempty finite matrix")
    if not 0 <= count <= len(features):
        raise ValueError("Invalid FPS count")
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Every descriptor must have nonzero norm")
    features = features / norms[:, None]
    chosen: list[int] = []
    nearest = np.full(len(features), np.inf)
    used = np.zeros(len(features), dtype=bool)
    candidate = 0
    for _ in range(count):
        chosen.append(candidate)
        used[candidate] = True
        nearest = np.minimum(nearest, 1.0 - features @ features[candidate])
        nearest[used] = -np.inf
        if len(chosen) < count:
            candidate = int(np.argmax(nearest))
    return chosen


@torch.inference_mode()
def extract_frame_descriptors(model, frames, *, device, batch_size: int = 8) -> np.ndarray:
    """Pool DA3 DINO patch features before cross-frame mixing.

    No depth head, teacher, labels or highlight processor is called. The same
    ImageNet normalization, patch embedding and four local blocks used by the
    formal forward apply. Each descriptor is independent of other video frames.
    """
    encoder = model.backbone.pretrained
    if batch_size < 1 or not len(frames):
        raise ValueError("Descriptor batch and sequence must be nonempty")
    result = []
    special = 1 + int(encoder.num_register_tokens)
    if int(encoder.alt_start) != 4 or int(encoder.rope_start) != 4:
        raise RuntimeError("DA3 descriptor expects four pre-global local DINO blocks before RoPE")
    for start in range(0, len(frames), batch_size):
        rgb = torch.stack([frames[i] for i in range(start, min(start + batch_size, len(frames)))])
        rgb = rgb.to(device)
        normalized = (rgb.unsqueeze(0) - model.imagenet_mean) / model.imagenet_std
        tokens = encoder.prepare_tokens_with_masks(normalized)
        for block in encoder.blocks[:4]:
            tokens = encoder.process_attention(tokens, block, "local")
        pooled = tokens[0, :, special:].float().mean(dim=1)
        result.append(F.normalize(pooled, dim=-1).cpu().numpy())
    return np.concatenate(result, axis=0)


@dataclass(frozen=True)
class GlobalAnchorBank:
    anchor_positions: tuple[int, ...]
    selection_seconds: float


def select_diverse_anchors(descriptors: np.ndarray, *, anchor_count: int = 50) -> GlobalAnchorBank:
    started = time.perf_counter()
    anchors = farthest_point_indices(descriptors, min(anchor_count, len(descriptors)))
    return GlobalAnchorBank(tuple(anchors),
                            time.perf_counter() - started)
