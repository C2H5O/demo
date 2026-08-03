"""Deterministic seeding helper for data-only scripts."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without selecting a device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
