"""Preprocess only an already-published canonical C3VD RGB view.

C3VD's official registration code describes an omnidirectional camera model.
This entrypoint deliberately refuses to reinterpret raw omnidirectional images
as pinhole images: put an official registered/reprojected RGB directory under
the input root and attest it with --official-canonical-rgb.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.preprocessing.common import contiguous_runs, image_files, process_image_run


def candidate_rgb_dirs(root: Path) -> list[Path]:
    names = {"rgb", "registered_rgb", "reprojected_rgb", "perspective_rgb"}
    return [p for p in root.rglob("*") if p.is_dir() and p.name.lower() in names and image_files(p)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--official-canonical-rgb", action="store_true", help="attest selected RGB is official registered/reprojected perspective output")
    args = parser.parse_args()
    if not args.official_canonical_rgb:
        parser.error("C3VD raw RGB cannot be treated as pinhole; pass --official-canonical-rgb only after confirming official reprojection/registered RGB")
    directories = candidate_rgb_dirs(args.input_root)
    if not directories:
        raise SystemExit("no C3VD RGB directories found (expected rgb/ or registered_rgb/; raw camera files are intentionally unsupported)")
    results = []
    for directory in directories:
        relative = directory.relative_to(args.input_root).parent
        for run_index, files in enumerate(contiguous_runs(image_files(directory))):
            sequence_id = relative.as_posix().replace("/", "_")
            if sequence_id in {"", "."}:
                sequence_id = directory.parent.name or "c3vd"
            if len(files) != len(image_files(directory)):
                sequence_id += f"_run{run_index:02d}"
            destination = args.output_root / "C3VD" / sequence_id
            if not args.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
            results.append(process_image_run(destination, "C3VD", sequence_id, files, overwrite=args.overwrite, dry_run=args.dry_run,
                extra_metadata={"source_camera_model": "omnidirectional (official C3VD config)", "canonical_source": "official registered/reprojected RGB attested by operator", "training_gt_written": False, "source_frame_gap_split": len(files) != len(image_files(directory))}))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
