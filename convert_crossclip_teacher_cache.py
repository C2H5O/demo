"""Convert complete cross-clip teacher caches from compressed to uncompressed NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

from cache.convert_crossclip_teacher_cache import (
    convert_crossclip_teacher_cache_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially rewrite complete cross-clip teacher NPZ files with "
            "ZIP_STORED, verify exact equality, then atomically replace each source."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Exact raw or aligned teacher-cache root to convert recursively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List compressed caches without writing or taking the conversion lock.",
    )
    parser.add_argument(
        "--confirm-no-readers",
        action="store_true",
        help=(
            "Required for writes: confirm all training/evaluation/visualization/"
            "generation jobs using this cache root have stopped."
        ),
    )
    parser.add_argument(
        "--start-at",
        help=(
            "Resume at this exact cache path relative to --root. Already converted "
            "files are skipped automatically, so this is normally unnecessary."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many selected files (useful for a trial run).",
    )
    args = parser.parse_args()
    report = convert_crossclip_teacher_cache_root(
        args.root,
        dry_run=args.dry_run,
        confirm_no_readers=args.confirm_no_readers,
        start_at=args.start_at,
        limit=args.limit,
    )
    print(
        "Conversion summary: discovered={} selected={} converted={} "
        "already_uncompressed={} source_gib={:.3f} output_gib={:.3f} dry_run={}".format(
            report.discovered,
            report.selected,
            report.converted,
            report.already_uncompressed,
            report.source_bytes / (1024.0 ** 3),
            report.output_bytes / (1024.0 ** 3),
            report.dry_run,
        )
    )


if __name__ == "__main__":
    main()
