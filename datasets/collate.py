"""Custom collate function that preserves temporal clip metadata."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


_TENSOR_KEYS = ("images", "frame_indices", "dataset_id", "sequence_length", "clip_start", "clip_length", "sample_stride")
_OPTIONAL_TENSOR_KEYS = ("highlight_masks", "inpainted_images")


def scared_collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensors while retaining strings and paths as lists per batch element."""
    if not samples:
        raise ValueError("Cannot collate an empty SCARED batch")
    batch: Dict[str, Any] = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in _TENSOR_KEYS}
    for key in _OPTIONAL_TENSOR_KEYS:
        present = [key in sample for sample in samples]
        if any(present) and not all(present):
            raise ValueError("Optional tensor key {!r} is missing from part of the batch".format(key))
        if all(present):
            batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
    for key in samples[0]:
        if key not in _TENSOR_KEYS and key not in _OPTIONAL_TENSOR_KEYS:
            batch[key] = [sample[key] for sample in samples]
    return batch
