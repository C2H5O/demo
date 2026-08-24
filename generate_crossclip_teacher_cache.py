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
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start at this global clip index before scanning existing caches",
    )
    parser.add_argument("--start-dataset-id", type=int, default=None)
    parser.add_argument("--start-keyframe-id", default=None, help="For example: keyframe_3")
    parser.add_argument("--start-clip-start", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args()
    generate_crossclip_teacher_cache(
        Path(args.config),
        args.split,
        args.limit,
        args.overwrite,
        args.cache_root,
        args.start_index,
        args.start_dataset_id,
        args.start_keyframe_id,
        args.start_clip_start,
    )


if __name__ == "__main__":
    main()
