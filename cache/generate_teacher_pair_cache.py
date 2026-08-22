"""Compatibility entry point for the per-frame base-teacher cache protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cache.generate_teacher_frame_cache import generate_teacher_frame_cache


def generate_teacher_pair_cache(
    config_path: Path,
    split: str,
    limit: Optional[int] = None,
    overwrite: bool = False,
    base_teacher: bool = True,
    cache_root_override: Optional[Path] = None,
) -> None:
    """Generate reusable frame caches; never run a pair-conditioned teacher."""
    if not base_teacher:
        raise ValueError(
            "LoRA/pair-conditioned caches are disabled; use the frozen base teacher"
        )
    if limit is not None:
        raise ValueError(
            "The compatibility pair entry point does not accept --limit because "
            "frame and pair counts differ; use generate_teacher_frame_cache.py --limit"
        )
    generate_teacher_frame_cache(
        config_path, split, None, overwrite, cache_root_override
    )


__all__ = ["generate_teacher_pair_cache"]
