"""Strict Hamlyn sequence discovery and numeric RGB/GT frame matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from evaluation.hamlyn.constants import (
    HAMLYN_GT_SCALE,
    HAMLYN_MAX_DEPTH,
    HAMLYN_MIN_DEPTH,
    HAMLYN_SEQUENCE_IDS,
)
from evaluation.hamlyn.gt import read_hamlyn_gt_uint16


PNG_SUFFIX = ".png"
NUMBER_PATTERN = re.compile(r"(\d+)")
SEQUENCE_DIRECTORY_PATTERN = re.compile(
    r"^(rectified|cropped|sequence|seq|hamlyn)[\s_-]*0*(\d+)$", re.IGNORECASE
)


class DiscoveryError(RuntimeError):
    """Raised when the fixed Hamlyn split cannot be discovered unambiguously."""


def natural_key(value: str | Path):
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    ]


def frame_id(path: str | Path) -> int:
    matches = NUMBER_PATTERN.findall(Path(path).stem)
    if not matches:
        raise DiscoveryError("No numeric frame ID in filename: {}".format(path))
    return int(matches[-1])


def index_by_frame_id(paths: Iterable[Path], label: str) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    for path in paths:
        identifier = frame_id(path)
        if identifier in result:
            raise DiscoveryError(
                "Duplicate {} frame ID {}: {} and {}".format(
                    label, identifier, result[identifier], path
                )
            )
        result[identifier] = path
    return result


def _png_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    values = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() == PNG_SUFFIX
    ]
    values.sort(key=natural_key)
    return values


def _directory_or_children(directory: Path, names: Sequence[str]) -> List[Path]:
    values = [directory / name for name in names]
    values.append(directory)
    return values


def _candidate_pairs(root: Path, sequence_id: int) -> List[Tuple[int, Path, Path]]:
    """Return common prepared layouts in deterministic priority order."""
    suffixes = ("{:02d}".format(sequence_id), str(sequence_id))
    pairs: List[Tuple[int, Path, Path]] = []
    # Support sequences directly below HAMLYN_ROOT and below one named container
    # (for example, HAMLYN_ROOT/prepared/rectified01) without recursively walking
    # the potentially very large frame tree.
    bases = [root]
    bases.extend(
        path
        for path in root.iterdir() if path.is_dir()
        and SEQUENCE_DIRECTORY_PATTERN.match(path.name) is None
        and not path.name.casefold().startswith("depth_cropped")
    )
    for base in bases:
        for suffix in suffixes:
            rectified = base / "rectified{}".format(suffix)
            # The server's native Hamlyn stereo layout stores the left stream
            # and its registered depth in image01/depth01. This evaluation is
            # monocular, so select camera 01 deterministically when both views
            # are present.
            pairs.append((-1, rectified / "image01", rectified / "depth01"))
            for rgb in _directory_or_children(rectified, ("color", "rgb", "images")):
                for depth in _directory_or_children(rectified, ("depth", "depth_gt")):
                    pairs.append((0, rgb, depth))
            cropped = base / "cropped{}".format(suffix)
            depth_cropped = base / "depth_cropped{}".format(suffix)
            for rgb in _directory_or_children(cropped, ("color", "rgb", "images")):
                for depth in _directory_or_children(depth_cropped, ("depth", "depth_gt")):
                    pairs.append((1, rgb, depth))

        for directory in base.iterdir() if base.is_dir() else ():
            match = SEQUENCE_DIRECTORY_PATTERN.match(directory.name) if directory.is_dir() else None
            if match is None or int(match.group(2)) != sequence_id:
                continue
            for rgb in _directory_or_children(directory, ("color", "rgb", "images", "left")):
                for depth in _directory_or_children(directory, ("depth", "depth_gt", "ground_truth")):
                    pairs.append((2, rgb, depth))
    return pairs


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: int
    rgb_directory: Path
    depth_directory: Path
    rgb_by_id: Mapping[int, Path]
    depth_by_id: Mapping[int, Path]

    @property
    def frame_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.rgb_by_id))

    @property
    def frame_count(self) -> int:
        return len(self.frame_ids)

    def to_dict(self) -> Dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "rgb_directory": str(self.rgb_directory),
            "depth_directory": str(self.depth_directory),
            "frame_ids": list(self.frame_ids),
            "frame_count": self.frame_count,
        }


def _discover_one(root: Path, sequence_id: int) -> SequenceRecord:
    usable = []
    seen = set()
    for priority, rgb_directory, depth_directory in _candidate_pairs(root, sequence_id):
        identity = (rgb_directory.resolve(), depth_directory.resolve())
        if identity in seen:
            continue
        seen.add(identity)
        rgb_paths = _png_files(rgb_directory)
        depth_paths = _png_files(depth_directory)
        if rgb_paths and depth_paths and rgb_directory.resolve() != depth_directory.resolve():
            usable.append((priority, rgb_directory, depth_directory, rgb_paths, depth_paths))
    if not usable:
        raise DiscoveryError(
            "Missing Hamlyn sequence {} below {}. Expected layouts such as "
            "rectified{:02d}/image01 + depth01, rectified{:02d}/color + depth, "
            "or cropped{:02d} + depth_cropped{:02d}.".format(
                sequence_id,
                root,
                sequence_id,
                sequence_id,
                sequence_id,
                sequence_id,
            )
        )
    best_priority = min(item[0] for item in usable)
    best = [item for item in usable if item[0] == best_priority]
    unique = {(item[1].resolve(), item[2].resolve()): item for item in best}
    if len(unique) != 1:
        raise DiscoveryError(
            "Ambiguous Hamlyn layout for sequence {}: {}".format(
                sequence_id, [(str(rgb), str(depth)) for rgb, depth in unique]
            )
        )
    _, rgb_directory, depth_directory, rgb_paths, depth_paths = next(iter(unique.values()))
    rgb_by_id = index_by_frame_id(rgb_paths, "RGB")
    depth_by_id = index_by_frame_id(depth_paths, "GT")
    rgb_ids, depth_ids = set(rgb_by_id), set(depth_by_id)
    if rgb_ids != depth_ids:
        raise DiscoveryError(
            "RGB/GT frame IDs differ for sequence {}: missing GT {}; extra GT {}".format(
                sequence_id,
                sorted(rgb_ids - depth_ids)[:20],
                sorted(depth_ids - rgb_ids)[:20],
            )
        )
    if not rgb_ids:
        raise DiscoveryError("No matched RGB/GT frames for sequence {}".format(sequence_id))
    return SequenceRecord(
        sequence_id=sequence_id,
        rgb_directory=rgb_directory.resolve(),
        depth_directory=depth_directory.resolve(),
        rgb_by_id=rgb_by_id,
        depth_by_id=depth_by_id,
    )


def discover_sequences(
    root: Path, sequence_ids: Sequence[int] = HAMLYN_SEQUENCE_IDS
) -> List[SequenceRecord]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise DiscoveryError("Hamlyn root is not a directory: {}".format(root))
    requested = tuple(int(value) for value in sequence_ids)
    if requested != HAMLYN_SEQUENCE_IDS:
        raise DiscoveryError("Evaluation requires the exact ordered 22-sequence Hamlyn split")
    records = [_discover_one(root, identifier) for identifier in requested]
    if tuple(item.sequence_id for item in records) != HAMLYN_SEQUENCE_IDS:
        raise DiscoveryError("Discovered Hamlyn sequence order escaped the formal protocol")
    return records


def inspect_sequence(record: SequenceRecord) -> Dict[str, object]:
    """Read every matched PNG so preflight fails before any model is loaded."""
    rgb_shapes = set()
    gt_shapes = set()
    gt_dtypes = set()
    gt_min = None
    gt_max = None
    valid_pixels = 0
    for identifier in record.frame_ids:
        with Image.open(record.rgb_by_id[identifier]) as image:
            rgb_shapes.add((image.height, image.width))
        depth = read_hamlyn_gt_uint16(record.depth_by_id[identifier])
        gt_shapes.add(tuple(depth.shape))
        gt_dtypes.add(str(depth.dtype))
        current_min, current_max = int(depth.min()), int(depth.max())
        gt_min = current_min if gt_min is None else min(gt_min, current_min)
        gt_max = current_max if gt_max is None else max(gt_max, current_max)
        valid_pixels += int(
            ((depth.astype(np.float32) * HAMLYN_GT_SCALE > HAMLYN_MIN_DEPTH)
             & (depth.astype(np.float32) * HAMLYN_GT_SCALE < HAMLYN_MAX_DEPTH)).sum()
        )
    if valid_pixels == 0:
        raise DiscoveryError(
            "Hamlyn sequence {} has no valid GT pixels; dtype={}, min={}, max={}".format(
                record.sequence_id, sorted(gt_dtypes), gt_min, gt_max
            )
        )
    return {
        **record.to_dict(),
        "rgb_frame_count": len(record.rgb_by_id),
        "gt_frame_count": len(record.depth_by_id),
        "matched_frame_count": record.frame_count,
        "rgb_original_resolutions_hw": [list(shape) for shape in sorted(rgb_shapes)],
        "gt_original_resolutions_hw": [list(shape) for shape in sorted(gt_shapes)],
        "gt_dtype": sorted(gt_dtypes),
        "gt_raw_min": gt_min,
        "gt_raw_max": gt_max,
        "valid_pixel_count": valid_pixels,
    }


def print_preflight(records: Sequence[SequenceRecord]) -> List[Dict[str, object]]:
    inspected = []
    for record in records:
        item = inspect_sequence(record)
        inspected.append(item)
        print(
            "[Hamlyn {:02d}] RGB={} GT={} rgb_frames={} gt_frames={} matched={} "
            "rgb_hw={} gt_hw={} gt_dtype={} gt_min/max={}/{}".format(
                record.sequence_id,
                record.rgb_directory,
                record.depth_directory,
                item["rgb_frame_count"],
                item["gt_frame_count"],
                item["matched_frame_count"],
                item["rgb_original_resolutions_hw"],
                item["gt_original_resolutions_hw"],
                item["gt_dtype"],
                item["gt_raw_min"],
                item["gt_raw_max"],
            ),
            flush=True,
        )
    return inspected
