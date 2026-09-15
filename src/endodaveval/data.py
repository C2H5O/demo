"""SCARED dataset 8/9 discovery with numeric frame-ID matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEPTH_SUFFIXES = {".npy", ".png", ".tif", ".tiff", ".exr"}
DATASET_PATTERN = re.compile(r"^dataset[\s_-]*0*(\d+)$", re.IGNORECASE)
KEYFRAME_PATTERN = re.compile(r"^key[\s_-]*frame[\s_-]*(.+)$", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(\d+)")


class DiscoveryError(RuntimeError):
    pass


def natural_key(value: Union[str, Path]):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(value))]


def frame_id(path: Union[str, Path]) -> int:
    matches = NUMBER_PATTERN.findall(Path(path).stem)
    if not matches:
        raise DiscoveryError("No numeric frame ID in filename: {}".format(path))
    return int(matches[-1])


def index_by_frame_id(paths: Iterable[Path]) -> Dict[int, Path]:
    indexed: Dict[int, Path] = {}
    for path in paths:
        identifier = frame_id(path)
        if identifier in indexed:
            raise DiscoveryError("Duplicate frame ID {}".format(identifier))
        indexed[identifier] = path
    return indexed


def _files(directory: Path, suffixes: set) -> List[Path]:
    if not directory.is_dir():
        return []
    values = [p for p in directory.iterdir() if p.is_file() and p.suffix.casefold() in suffixes]
    values.sort(key=natural_key)
    return values


@dataclass(frozen=True)
class SequenceRecord:
    dataset_id: int
    keyframe_id: str
    sequence_id: str
    keyframe_directory: Path
    frame_directory: Path
    rgb_by_id: Mapping[int, Path]
    ground_truth_directory: Path
    ground_truth_by_id: Mapping[int, Path]

    @property
    def frame_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.rgb_by_id))

    def to_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "keyframe_id": self.keyframe_id,
            "sequence_id": self.sequence_id,
            "keyframe_directory": str(self.keyframe_directory),
            "rgb_directory": str(self.frame_directory),
            "ground_truth_directory": str(self.ground_truth_directory),
            "frame_ids": list(self.frame_ids),
            "frame_count": len(self.frame_ids),
        }


def discover_sequences(
    root: Path,
    dataset_ids: Sequence[int],
    frame_sources: Sequence[str],
    ground_truth_relative: str,
) -> Tuple[List[SequenceRecord], List[str]]:
    root = root.resolve()
    datasets: Dict[int, Path] = {}
    if not root.is_dir():
        raise DiscoveryError("SCARED root is not a directory: {}".format(root))
    for path in root.iterdir():
        match = DATASET_PATTERN.match(path.name) if path.is_dir() else None
        if match:
            datasets[int(match.group(1))] = path
    missing = sorted(set(dataset_ids) - set(datasets))
    if missing:
        raise DiscoveryError("Missing SCARED dataset IDs {}".format(missing))
    records: List[SequenceRecord] = []
    skipped: List[str] = []
    for dataset_id in dataset_ids:
        keyframes = [p for p in datasets[dataset_id].iterdir() if p.is_dir() and KEYFRAME_PATTERN.match(p.name)]
        keyframes.sort(key=natural_key)
        for keyframe in keyframes:
            selected: Optional[Tuple[Path, Dict[int, Path]]] = None
            for source in frame_sources:
                directory = keyframe / "data" / source
                paths = _files(directory, IMAGE_SUFFIXES)
                if paths:
                    selected = directory, index_by_frame_id(paths)
                    break
            gt_directory = keyframe / ground_truth_relative
            gt_paths = _files(gt_directory, DEPTH_SUFFIXES)
            if selected is None or not gt_paths:
                skipped.append("{}: missing RGB or GT".format(keyframe))
                continue
            frame_directory, rgb_by_id = selected
            records.append(
                SequenceRecord(
                    dataset_id,
                    keyframe.name,
                    "dataset_{}/{}".format(dataset_id, keyframe.name),
                    keyframe,
                    frame_directory,
                    rgb_by_id,
                    gt_directory,
                    index_by_frame_id(gt_paths),
                )
            )
    if not records:
        raise DiscoveryError("No usable SCARED sequences found below {}".format(root))
    records.sort(key=lambda item: (item.dataset_id, natural_key(item.keyframe_id)))
    return records, skipped


def matched_frame_ids(record: SequenceRecord, require_all: bool = True) -> Tuple[int, ...]:
    rgb_ids = set(record.rgb_by_id)
    gt_ids = set(record.ground_truth_by_id)
    if require_all and rgb_ids != gt_ids:
        raise DiscoveryError(
            "RGB/GT frame IDs differ for {}: missing GT {}; extra GT {}".format(
                record.sequence_id, sorted(rgb_ids - gt_ids)[:20], sorted(gt_ids - rgb_ids)[:20]
            )
        )
    matched = tuple(sorted(rgb_ids & gt_ids))
    if not matched:
        raise DiscoveryError("No matched RGB/GT frame IDs for {}".format(record.sequence_id))
    return matched
