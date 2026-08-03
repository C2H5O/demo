"""Shape-safe adapters between temporal and flattened image batches."""

from __future__ import annotations

import torch


def flatten_video_batch(images: torch.Tensor) -> torch.Tensor:
    """Flatten [B,T,C,H,W] temporal RGB batches to [B*T,C,H,W]."""
    if images.ndim != 5:
        raise ValueError("Expected [B,T,C,H,W], got {}".format(tuple(images.shape)))
    batch, time, channels, height, width = images.shape
    return images.reshape(batch * time, channels, height, width)


def restore_video_batch(flattened: torch.Tensor, batch_size: int, time_steps: int) -> torch.Tensor:
    """Restore [B*T,...] tensors to [B,T,...] using explicit dimensions."""
    if flattened.ndim < 1:
        raise ValueError("flattened tensor must have at least one dimension")
    if batch_size <= 0 or time_steps <= 0:
        raise ValueError("batch_size and time_steps must be positive")
    if flattened.shape[0] != batch_size * time_steps:
        raise ValueError("Cannot restore first dimension {} to [B={},T={}].".format(flattened.shape[0], batch_size, time_steps))
    return flattened.reshape((batch_size, time_steps) + tuple(flattened.shape[1:]))
