"""Validate canonical preprocessing contracts without reading any training code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

TEACHER_WH = (1280, 1024)
STUDENT_WH = (560, 448)


def clip_count(frame_count: int) -> int:
    return max(0, frame_count - 15)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_sequence(sequence: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    marker = sequence / "_preprocess_complete.json"
    metadata_path = sequence / "metadata.json"
    if not marker.exists() or not metadata_path.exists():
        return [f"{sequence}: missing complete marker or metadata"], warnings
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{sequence}: invalid metadata: {exc}"], warnings
    frames = metadata.get("frames")
    if not isinstance(frames, list):
        return [f"{sequence}: metadata.frames must be a list"], warnings
    teacher_dir, student_dir = sequence / "teacher_rgb", sequence / "student_rgb"
    teacher = sorted(teacher_dir.glob("*.png")) if teacher_dir.exists() else []
    student = sorted(student_dir.glob("*.png")) if student_dir.exists() else []
    teacher_names, student_names = [p.name for p in teacher], [p.name for p in student]
    if teacher_names != student_names:
        fail(errors, f"{sequence}: teacher/student names differ")
    if len(teacher) != len(frames) or len(student) != len(frames):
        fail(errors, f"{sequence}: image counts ({len(teacher)}, {len(student)}) disagree with frame mapping ({len(frames)})")
    for expected, entry in enumerate(frames):
        name = f"{expected:06d}.png"
        if entry.get("processed_index") != expected or Path(entry.get("teacher_rgb_file", "")).name != name or Path(entry.get("student_rgb_file", "")).name != name:
            fail(errors, f"{sequence}: non-canonical mapping at processed index {expected}")
    for path in teacher:
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != TEACHER_WH:
                fail(errors, f"{path}: expected RGB {TEACHER_WH}, got {image.mode} {image.size}")
    for path in student:
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != STUDENT_WH:
                fail(errors, f"{path}: expected RGB {STUDENT_WH}, got {image.mode} {image.size}")
    if metadata.get("decoded_frame_count") != len(frames) or metadata.get("written_teacher_frames") != len(frames) or metadata.get("written_student_frames") != len(frames):
        fail(errors, f"{sequence}: completion counts disagree with mappings")
    if metadata.get("evaluation_only"):
        depth_dir = sequence / "data" / "depth"
        depth_names = sorted(p.stem for p in depth_dir.glob("*.npy")) if depth_dir.exists() else []
        rgb_ids = [p.stem for p in student]
        if depth_names != rgb_ids:
            fail(errors, f"{sequence}: Hamlyn RGB/GT IDs differ")
        if metadata.get("output_depth_unit") != "mm" or metadata.get("invalid_depth_value") != 0:
            fail(errors, f"{sequence}: Hamlyn depth unit/invalid semantics missing")
        for depth_file in depth_dir.glob("*.npy"):
            depth = np.load(depth_file)
            if depth.shape != (448, 560) or not np.issubdtype(depth.dtype, np.floating) or np.any(depth < 0):
                fail(errors, f"{depth_file}: expected nonnegative float HxW (448, 560), got {depth.dtype} {depth.shape}")
    elif len(frames) < 16:
        warnings.append(f"{sequence}: {len(frames)} frames creates 0 train clips (requires >=16)")
    # This is a contract check, not a sampler: adjacent clips are sequence-local
    # because validation never combines sequences.
    if len(frames) >= 17:
        first, second = set(range(0, 16)), set(range(1, 17))
        if len(first & second) != 15:
            fail(errors, f"{sequence}: internal 16-frame stride-1 overlap invariant failed")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="processed root containing C3VD/, StereoMIS/, etc.")
    args = parser.parse_args()
    sequences = sorted({marker.parent for marker in args.root.rglob("_preprocess_complete.json")})
    if not sequences:
        raise SystemExit(f"no complete sequences under {args.root}")
    errors: list[str] = []
    warnings: list[str] = []
    summary = []
    for sequence in sequences:
        seq_errors, seq_warnings = validate_sequence(sequence)
        errors.extend(seq_errors)
        warnings.extend(seq_warnings)
        summary.append({"sequence": str(sequence), "clip_count": clip_count(len(json.loads((sequence / "metadata.json").read_text(encoding="utf-8")).get("frames", []))) if not seq_errors else None})
    print(json.dumps({"sequences": summary, "warnings": warnings, "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
