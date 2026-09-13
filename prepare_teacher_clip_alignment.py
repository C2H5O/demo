from __future__ import annotations

import argparse
from pathlib import Path

from cache.teacher_clip_alignment import (
    audit_teacher_clip_alignment,
    build_teacher_clip_alignment,
    print_alignment_audit,
    write_teacher_clip_alignment,
)


def _add_cache_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Raw cache split directory, for example .../teacher_cache.../train",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or audit offline Teacher cross-clip scale alignment metadata"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Compute and write alignment metadata")
    _add_cache_root(prepare)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--expected-overlap-frames", type=int, default=8)
    prepare.add_argument("--eps", type=float, default=1.0e-6)
    prepare.add_argument("--minimum-valid-pixels-per-frame", type=int, default=256)

    audit = commands.add_parser(
        "audit", help="Recompute from raw caches and verify existing metadata"
    )
    _add_cache_root(audit)
    audit.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        metadata = build_teacher_clip_alignment(
            args.cache_root,
            expected_overlap_frames=args.expected_overlap_frames,
            eps=args.eps,
            minimum_valid_pixels_per_frame=args.minimum_valid_pixels_per_frame,
        )
        output = write_teacher_clip_alignment(metadata, args.output)
        print("Teacher clip alignment metadata written: {}".format(output))
    else:
        metadata = audit_teacher_clip_alignment(args.cache_root, args.metadata)
        print("Teacher clip alignment audit passed: {}".format(args.metadata.resolve()))
    print_alignment_audit(metadata)


if __name__ == "__main__":
    main()
