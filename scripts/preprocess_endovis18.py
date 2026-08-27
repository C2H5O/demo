"""Preprocess EndoVis18 train/test releases' explicit left-eye directories.

The release-1 archive has an extra same-named directory level.  Discovery is
recursive so both that form and releases 2--4 are accepted; output identities
retain the complete relative path to prevent train/test or release collisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.preprocessing.common import contiguous_runs, image_files, process_image_run


def left_frame_directories(root: Path) -> list[Path]:
    """Find only sequence-level left frame directories across train and test."""
    return [
        directory
        for directory in [root, *sorted(root.rglob("*"))]
        if directory.is_dir()
        and directory.name.lower() in {"left_frames", "left"}
        and image_files(directory)
    ]


def sequence_id_for_left_frames(directory: Path, root: Path) -> str:
    """Preserve split/release/sequence identity, including release-1's nesting."""
    return directory.parent.relative_to(root).as_posix().replace("/", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--accept-published-left-frames", action="store_true", help="required acknowledgement: challenge's separate left_frames are used unchanged; no undocumented rectification is invented")
    args = parser.parse_args()
    if not args.accept_published_left_frames:
        parser.error("pass --accept-published-left-frames after confirming this release's left_frames geometry")
    candidates = left_frame_directories(args.input_root)
    if not candidates:
        raise SystemExit("no EndoVis18 left_frames/ directories found; packed stereo layouts are intentionally not guessed")
    results = []
    for directory in candidates:
        all_files = image_files(directory)
        for run_number, files in enumerate(contiguous_runs(all_files)):
            sequence_id = sequence_id_for_left_frames(directory, args.input_root)
            if len(files) < len(all_files):
                sequence_id += f"_run{run_number:02d}"
            destination = args.output_root / "EndoVis18" / sequence_id
            if not args.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
            results.append(process_image_run(destination, "EndoVis18", sequence_id, files, overwrite=args.overwrite, dry_run=args.dry_run,
                extra_metadata={"eye": "left", "stereo_layout": "separate left_frames/right_frames", "split": next((part for part in directory.relative_to(args.input_root).parts if part in {"train", "test"}), "UNVERIFIED"), "calibration_file_present": (directory.parent / "camera_calibration.txt").exists(), "rectification": "UNVERIFIED: published left image geometry retained without undocumented transform", "labels_used": False, "temporal_order": "natural numeric filename order"}))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
