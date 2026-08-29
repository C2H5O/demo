"""Materialize canonical teacher/student RGB pairs from extracted SCARED left frames.

The raw SCARED tree is read-only input. Every discovered ``data/left``
sequence is written transactionally under ``processed/SCARED`` with a complete
marker and a frame-level source mapping. Depth, disparity, masks, poses, and
other annotations are deliberately not copied into this RGB-only output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.preprocessing.common import contiguous_runs, process_image_run
from datasets.scared_discovery import SequenceRecord, discover_scared_sequences


def split_name(dataset_id: int) -> str:
    if 1 <= dataset_id <= 7:
        return "train"
    if dataset_id in {8, 9}:
        return "test"
    raise ValueError(f"unsupported SCARED dataset ID: {dataset_id}")


def output_location(output_root: Path, record: SequenceRecord, run_index: int, run_count: int) -> tuple[str, Path]:
    """Return a collision-safe canonical identity and directory for one run."""
    dataset_name = f"dataset_{record.dataset_id:02d}"
    keyframe_name = record.keyframe_id
    if run_count > 1:
        keyframe_name += f"_run{run_index:02d}"
    sequence_id = f"{dataset_name}/{keyframe_name}"
    return sequence_id, output_root / "SCARED" / dataset_name / keyframe_name


def preprocess_records(
    records: list[SequenceRecord], output_root: Path, *, overwrite: bool, dry_run: bool
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for record in records:
        runs = contiguous_runs(record.frame_paths)
        for run_index, files in enumerate(runs):
            sequence_id, destination = output_location(output_root, record, run_index, len(runs))
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
            results.append(process_image_run(
                destination, "SCARED", sequence_id, files, overwrite=overwrite, dry_run=dry_run,
                extra_metadata={
                    "source_split": split_name(record.dataset_id),
                    "source_dataset_id": record.dataset_id,
                    "source_keyframe_id": record.keyframe_id,
                    "source_frame_directory": str(record.frame_directory),
                    "source_frame_source": "left",
                    "source_calibration_path": str(record.calibration_path) if record.calibration_path else None,
                    "source_frame_gap_split": len(runs) > 1,
                    "teacher_input_contract": {"height": 1024, "width": 1280},
                    "student_input_contract": {"height": 448, "width": 560},
                    "training_gt_written": False,
                },
            ))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="SCARED root containing dataset_1 ... dataset_9")
    parser.add_argument("--output-root", type=Path, required=True, help="processed root; output is placed in SCARED/")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    records, malformed = discover_scared_sequences(args.input_root, split=args.split, frame_source="left", strict=True)
    results = preprocess_records(records, args.output_root, overwrite=args.overwrite, dry_run=args.dry_run)
    print(json.dumps({"processed_sequences": results, "skipped_sequences": malformed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
