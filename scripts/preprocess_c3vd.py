"""Preprocess C3VD ``*_color.png`` frames from every C3VD sequence folder.

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


def color_frames(directory: Path) -> list[Path]:
    """Return only C3VD colour frames, never depth/flow/occlusion/normal maps.

    The local release has one C3VD sequence per directory and stores all frame
    modalities together.  Its RGB records are named ``color.png`` or
    ``*_color.png``.  Selecting generic image files here would accidentally
    feed the GT modalities to the teacher.
    """
    return [
        path
        for path in image_files(directory)
        if path.name.lower() == "color.png" or path.stem.lower().endswith("_color")
    ]


def candidate_rgb_dirs(root: Path) -> list[tuple[Path, list[Path]]]:
    names = {"rgb", "registered_rgb", "reprojected_rgb", "perspective_rgb"}
    candidates: list[tuple[Path, list[Path]]] = []
    for directory in [root, *sorted(root.rglob("*"))]:
        if not directory.is_dir():
            continue
        files = color_frames(directory)
        # Keep compatibility with an explicitly named official perspective RGB
        # directory, while preferring the actual C3VD colour-frame layout.
        if not files and directory.name.lower() in names:
            files = image_files(directory)
        if files:
            candidates.append((directory, files))
    return candidates


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
        raise SystemExit("no C3VD colour frames found (expected per-sequence *_color.png/color.png or an explicitly named official RGB directory)")
    results = []
    for directory, all_files in directories:
        relative = directory.relative_to(args.input_root)
        for run_index, files in enumerate(contiguous_runs(all_files)):
            sequence_id = relative.as_posix().replace("/", "_")
            if sequence_id in {"", "."}:
                sequence_id = directory.name or "c3vd"
            if len(files) != len(all_files):
                sequence_id += f"_run{run_index:02d}"
            destination = args.output_root / "C3VD" / sequence_id
            if not args.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
            results.append(process_image_run(destination, "C3VD", sequence_id, files, overwrite=args.overwrite, dry_run=args.dry_run,
                extra_metadata={"source_camera_model": "omnidirectional (official C3VD config)", "canonical_source": "official registered/reprojected RGB attested by operator", "source_rgb_pattern": "color.png or *_color.png", "training_gt_written": False, "source_frame_gap_split": len(files) != len(all_files)}))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
