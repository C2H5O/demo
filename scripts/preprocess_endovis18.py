"""Preprocess EndoVis18's explicit left-eye image directories only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.preprocessing.common import contiguous_runs, image_files, process_image_run


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
    candidates = [p for p in args.input_root.rglob("*") if p.is_dir() and p.name.lower() in {"left_frames", "left"} and image_files(p)]
    if not candidates:
        raise SystemExit("no EndoVis18 left_frames/ directories found; packed stereo layouts are intentionally not guessed")
    results = []
    for directory in candidates:
        all_files = image_files(directory)
        for run_number, files in enumerate(contiguous_runs(all_files)):
            sequence_id = directory.parent.relative_to(args.input_root).as_posix().replace("/", "_")
            if len(files) < len(all_files):
                sequence_id += f"_run{run_number:02d}"
            destination = args.output_root / "EndoVis18" / sequence_id
            if not args.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
            results.append(process_image_run(destination, "EndoVis18", sequence_id, files, overwrite=args.overwrite, dry_run=args.dry_run,
                extra_metadata={"eye": "left", "stereo_layout": "separate left_frames/right_frames", "calibration_file_present": (directory.parent / "camera_calibration.txt").exists(), "rectification": "UNVERIFIED: published left image geometry retained without undocumented transform", "labels_used": False, "temporal_order": "natural numeric filename order"}))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
