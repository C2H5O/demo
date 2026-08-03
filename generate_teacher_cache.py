from __future__ import annotations

import argparse
from pathlib import Path

from cache.generate_teacher_cache import generate_teacher_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/student_distillation.yaml")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--base-teacher",
        action="store_true",
        help="Use only the pretrained VGGT-Omega weights; do not inject or load LoRA",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Override teacher.cache_root (required with --base-teacher)",
    )
    args = parser.parse_args()
    if args.base_teacher and args.cache_root is None:
        parser.error(
            "--base-teacher requires --cache-root so base and LoRA caches "
            "cannot be mixed"
        )
    generate_teacher_cache(
        Path(args.config),
        args.split,
        args.limit,
        args.overwrite,
        args.base_teacher,
        args.cache_root,
    )


if __name__ == "__main__":
    main()
