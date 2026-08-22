from __future__ import annotations

import argparse
from pathlib import Path

from cache.generate_teacher_frame_cache import generate_teacher_frame_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate independent frozen base-teacher caches per SCARED frame"
    )
    parser.add_argument("--config", default="configs/vggtomast3r_v1.yaml")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args()
    generate_teacher_frame_cache(
        Path(args.config), args.split, args.limit, args.overwrite, args.cache_root
    )


if __name__ == "__main__":
    main()
