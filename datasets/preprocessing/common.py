"""Transactional output, frame identity, and deterministic sequence utilities."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image

from .geometry import STUDENT_SIZE, TEACHER_SIZE, contain_depth_valid_aware, make_rgb_pair

_NUMBER = re.compile(r"(\d+)")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path | str) -> tuple[Any, ...]:
    """Natural/numeric ordering, with path text as a deterministic tie-breaker."""
    text = Path(path).name.lower()
    return tuple(int(x) if x.isdigit() else x for x in _NUMBER.split(text)) + (text,)


def source_frame_id(path: Path | str, fallback: int) -> int:
    matches = _NUMBER.findall(Path(path).stem)
    return int(matches[-1]) if matches else fallback


def contiguous_runs(items: Iterable[Path]) -> list[list[Path]]:
    """Split only when numeric source IDs prove a gap; IDs without digits stay ordered."""
    ordered = sorted(items, key=natural_key)
    if not ordered:
        return []
    runs: list[list[Path]] = [[ordered[0]]]
    previous = source_frame_id(ordered[0], 0)
    for position, item in enumerate(ordered[1:], start=1):
        current = source_frame_id(item, position)
        if current != previous + 1:
            runs.append([])
        runs[-1].append(item)
        previous = current
    return runs


@dataclass(frozen=True)
class ProcessedFrame:
    source_file: Path
    source_frame_id: int
    source_timestamp: float | None = None


class SequenceWriter:
    """Writes a sequence atomically enough to make interruption resumable.

    A sequence is first created in a sibling temporary directory.  The complete
    marker is written only after all images and metadata have been fsynced; its
    final directory is then atomically renamed into place.  Existing completed
    outputs are never overwritten unless explicitly requested.
    """

    def __init__(self, output_dir: Path, *, dataset_name: str, sequence_id: str,
                 overwrite: bool, dry_run: bool, evaluation_only: bool = False) -> None:
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.sequence_id = sequence_id
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.evaluation_only = evaluation_only
        self.temp_dir: Path | None = None
        self.frames: list[dict[str, Any]] = []
        self.failed_frames: list[dict[str, str]] = []

    def __enter__(self) -> "SequenceWriter":
        marker = self.output_dir / "_preprocess_complete.json"
        if marker.exists() and not self.overwrite:
            raise FileExistsError(f"complete output exists (use --overwrite): {self.output_dir}")
        if self.output_dir.exists() and self.overwrite and not self.dry_run:
            # Keep a recoverable previous result rather than deleting it.
            backup = self.output_dir.with_name(self.output_dir.name + ".previous")
            if backup.exists():
                raise FileExistsError(f"refusing to replace existing backup: {backup}")
            os.replace(self.output_dir, backup)
        if not self.dry_run:
            self.temp_dir = Path(tempfile.mkdtemp(prefix=self.output_dir.name + ".partial-", dir=self.output_dir.parent))
            (self.temp_dir / "teacher_rgb").mkdir()
            (self.temp_dir / "student_rgb").mkdir()
        return self

    def write_rgb(self, frame: ProcessedFrame, processed_index: int, canonical: Image.Image) -> None:
        teacher_name = f"{processed_index:06d}.png"
        student_name = teacher_name
        teacher, student, geometry = make_rgb_pair(canonical)
        if self.temp_dir is not None:
            teacher.save(self.temp_dir / "teacher_rgb" / teacher_name, format="PNG")
            student.save(self.temp_dir / "student_rgb" / student_name, format="PNG")
        self.frames.append({
            "processed_index": processed_index,
            "source_frame_id": frame.source_frame_id,
            "source_timestamp": frame.source_timestamp,
            "source_file": str(frame.source_file),
            "teacher_rgb_file": f"teacher_rgb/{teacher_name}",
            "student_rgb_file": f"student_rgb/{student_name}",
            "geometry": geometry,
        })

    def fail(self, frame: ProcessedFrame, exc: Exception) -> None:
        self.failed_frames.append({"source_file": str(frame.source_file), "error": str(exc)})

    def write_depth_mm(self, processed_index: int, depth_mm: "Any") -> None:
        """Write evaluation depth on the student grid with zero as the invalid value."""
        if self.temp_dir is None:
            return
        depth_dir = self.temp_dir / "data" / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        output = contain_depth_valid_aware(depth_mm, STUDENT_SIZE)
        # .npy preserves millimetres and invalid zeros without PNG quantisation.
        import numpy as np
        np.save(depth_dir / f"{processed_index:06d}.npy", output.astype(np.float32))

    def complete(self, extra_metadata: dict[str, Any] | None = None) -> None:
        metadata = {
            "format_version": 1,
            "dataset_name": self.dataset_name,
            "sequence_id": self.sequence_id,
            "evaluation_only": self.evaluation_only,
            "teacher_size_wh": list(TEACHER_SIZE),
            "student_size_wh": list(STUDENT_SIZE),
            "reported_frame_count": len(self.frames) + len(self.failed_frames),
            "decoded_frame_count": len(self.frames),
            "written_teacher_frames": len(self.frames),
            "written_student_frames": len(self.frames),
            "failed_frames": self.failed_frames,
            "frames": self.frames,
        }
        metadata.update(extra_metadata or {})
        if self.temp_dir is not None:
            (self.temp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            marker = {k: metadata[k] for k in ("format_version", "dataset_name", "sequence_id", "reported_frame_count", "decoded_frame_count")}
            (self.temp_dir / "_preprocess_complete.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
            os.replace(self.temp_dir, self.output_dir)
            self.temp_dir = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.temp_dir is not None:
            # Preserve partial work for diagnosis/recovery but never mark complete.
            partial = self.temp_dir.with_name(self.temp_dir.name + ".failed")
            os.replace(self.temp_dir, partial)
            self.temp_dir = None


def image_files(directory: Path) -> list[Path]:
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)


def process_image_run(output_dir: Path, dataset_name: str, sequence_id: str, files: list[Path], *,
                      overwrite: bool, dry_run: bool, extra_metadata: dict[str, Any] | None = None,
                      evaluation_only: bool = False) -> dict[str, Any]:
    """Decode a run one image at a time and create its canonical RGB pair."""
    with SequenceWriter(output_dir, dataset_name=dataset_name, sequence_id=sequence_id,
                        overwrite=overwrite, dry_run=dry_run, evaluation_only=evaluation_only) as writer:
        for index, file in enumerate(files):
            frame = ProcessedFrame(file, source_frame_id(file, index))
            try:
                with Image.open(file) as image:
                    writer.write_rgb(frame, index, image.copy())
            except Exception as exc:  # Decode failures must never get a complete marker.
                writer.fail(frame, exc)
                raise RuntimeError(f"failed decoding {file}: {exc}") from exc
        writer.complete(extra_metadata)
        return {"sequence": sequence_id, "frames": len(writer.frames), "output": str(output_dir), "dry_run": dry_run}
