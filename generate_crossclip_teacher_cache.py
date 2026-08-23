from __future__ import annotations

import argparse
from pathlib import Path

from cache.generate_crossclip_teacher_cache import generate_crossclip_teacher_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate raw frozen-base teacher caches for stride-one 16-frame clips"
    )
    parser.add_argument("--config", default="configs/crossclip_teacher_projection.yaml")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of 16-frame clips")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args()
    generate_crossclip_teacher_cache(
        Path(args.config), args.split, args.limit, args.overwrite, args.cache_root
    )


if __name__ == "__main__":
    main()
