"""Directory discovery for temporal left-camera SCARED RGB sequences."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FRAME_SOURCES = ("left", "left_finalpass", "rgb_data")
SUPPORTED_FRAME_SOURCES = FRAME_SOURCES + ("left_rectified",)
DATASET_PATTERN = re.compile(r"^dataset[\s_-]*0*(\d+)$", re.IGNORECASE)
KEYFRAME_PATTERN = re.compile(r"^key[\s_-]*frame[\s_-]*(.+)$", re.IGNORECASE)


class MissingRGBFramesError(RuntimeError):
    """Raised when a keyframe contains no supported temporal RGB frames."""


def natural_sort_key(value: Union[str, Path]) -> List[Union[int, str]]:
    """Return a case-insensitive key that sorts embedded integers naturally."""
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(value))]


def expected_dataset_ids(split: str) -> Tuple[int, ...]:
    """Return the official SCARED dataset IDs for a supported split."""
    split = split.lower()
    if split == "train":
        return tuple(range(1, 8))
    if split == "test":
        return (8, 9)
    if split == "all":
        return tuple(range(1, 10))
    raise ValueError("split must be one of: train, test, all; received {!r}".format(split))


def extract_dataset_id(name: str) -> Optional[int]:
    """Extract an integer ID from a direct SCARED dataset directory name."""
    match = DATASET_PATTERN.match(name)
    return int(match.group(1)) if match else None


def _relative_or_none(path: Path, root: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class SequenceRecord:
    """One temporal RGB sequence contained in a SCARED key-frame directory."""

    dataset_id: int
    keyframe_id: str
    sequence_id: str
    keyframe_directory: Path
    frame_directory: Path
    frame_paths: Tuple[Path, ...]
    calibration_path: Optional[Path]
    depth_directory: Optional[Path]
    disparity_directory: Optional[Path]
    frame_data_directory: Optional[Path]
    reprojection_directory: Optional[Path]
    scene_points_directory: Optional[Path]
    point_cloud_path: Optional[Path]
    video_path: Optional[Path]

    @property
    def sequence_length(self) -> int:
        return len(self.frame_paths)

    def to_manifest_dict(self, root: Path) -> Dict[str, object]:
        """Serialize portable paths relative to the supplied SCARED root."""
        return {
            "dataset_id": self.dataset_id,
            "keyframe_id": self.keyframe_id,
            "sequence_id": self.sequence_id,
            "keyframe_directory": _relative_or_none(self.keyframe_directory, root),
            "frame_directory": _relative_or_none(self.frame_directory, root),
            "frame_paths": [_relative_or_none(path, root) for path in self.frame_paths],
            "sequence_length": self.sequence_length,
            "calibration_path": _relative_or_none(self.calibration_path, root) if self.calibration_path else None,
            "depth_directory": _relative_or_none(self.depth_directory, root) if self.depth_directory else None,
            "disparity_directory": _relative_or_none(self.disparity_directory, root) if self.disparity_directory else None,
            "frame_data_directory": _relative_or_none(self.frame_data_directory, root) if self.frame_data_directory else None,
            "reprojection_directory": _relative_or_none(self.reprojection_directory, root) if self.reprojection_directory else None,
            "scene_points_directory": _relative_or_none(self.scene_points_directory, root) if self.scene_points_directory else None,
            "point_cloud_path": _relative_or_none(self.point_cloud_path, root) if self.point_cloud_path else None,
            "video_path": _relative_or_none(self.video_path, root) if self.video_path else None,
        }


def discover_dataset_directories(root: Union[str, Path], split: str) -> Dict[int, Path]:
    """Find official split directories and validate IDs before sequence discovery."""
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise FileNotFoundError("SCARED dataset root does not exist or is not a directory: {}".format(root_path))
    discovered: Dict[int, Path] = {}
    duplicate_ids: Dict[int, List[Path]] = {}
    for candidate in root_path.iterdir():
        if not candidate.is_dir():
            continue
        dataset_id = extract_dataset_id(candidate.name)
        if dataset_id is None:
            continue
        if dataset_id in discovered:
            duplicate_ids.setdefault(dataset_id, [discovered[dataset_id]]).append(candidate)
        else:
            discovered[dataset_id] = candidate
    if duplicate_ids:
        detail = "; ".join("{}: {}".format(dataset_id, [str(path) for path in paths]) for dataset_id, paths in sorted(duplicate_ids.items()))
        raise RuntimeError("Duplicate numerical SCARED dataset IDs under {}: {}".format(root_path, detail))
    expected = expected_dataset_ids(split)
    missing = sorted(set(expected) - set(discovered))
    if missing:
        raise FileNotFoundError("Missing required SCARED dataset IDs under {}. Expected IDs: {}; discovered IDs: {}; missing IDs: {}".format(root_path, list(expected), sorted(discovered), missing))
    return {dataset_id: discovered[dataset_id] for dataset_id in expected}


def _find_keyframe_directories(dataset_directory: Path) -> List[Path]:
    keyframes = [path for path in dataset_directory.iterdir() if path.is_dir() and KEYFRAME_PATTERN.match(path.name)]
    keyframes.sort(key=lambda path: natural_sort_key(path.name))
    if not keyframes:
        raise RuntimeError("No key_frame directories found in SCARED dataset directory: {}".format(dataset_directory))
    return keyframes


def _image_paths(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES]
    paths.sort(key=lambda path: natural_sort_key(path.name))
    if len({path.resolve() for path in paths}) != len(paths):
        raise RuntimeError("Duplicate resolved RGB frame paths found in {}".format(directory))
    return paths


def select_frame_directory(keyframe_directory: Path, frame_source: str) -> Tuple[Path, List[Path]]:
    """Select one non-recursive temporal RGB directory for a key-frame sequence.

    ``auto`` follows ``FRAME_SOURCES`` priority and therefore prefers the raw
    ``data/left`` frames whenever they are available.  It only falls back to a
    processed source when the higher-priority directory contains no images.
    """
    source = frame_source.lower()
    if source not in ("auto",) + SUPPORTED_FRAME_SOURCES:
        raise ValueError(
            "frame_source must be auto, left, left_finalpass, rgb_data, "
            "or left_rectified; received {!r}".format(frame_source)
        )
    data_directory = keyframe_directory / "data"
    names: Sequence[str] = FRAME_SOURCES if source == "auto" else (source,)
    checked_paths = [data_directory / name for name in names]
    for path in checked_paths:
        images = _image_paths(path)
        if images:
            return path, images
    raise MissingRGBFramesError(
        "No temporal RGB image sequence found for {}. Checked paths: {}".format(
            keyframe_directory,
            [str(path) for path in checked_paths],
        )
    )


def _optional_paths(keyframe_directory: Path) -> Dict[str, Optional[Path]]:
    data = keyframe_directory / "data"
    candidates = {
        "calibration_path": keyframe_directory / "endoscope_calibration.yaml",
        "depth_directory": data / "depth",
        "disparity_directory": data / "disparity",
        "frame_data_directory": data / "frame_data",
        "reprojection_directory": data / "reprojection_data",
        "scene_points_directory": data / "scene_points",
        "point_cloud_path": keyframe_directory / "point_cloud.obj",
        "video_path": data / "rgb.mp4",
    }
    return {name: path if path.exists() else None for name, path in candidates.items()}


def discover_scared_sequences(root: Union[str, Path], split: str = "train", frame_source: str = "auto", strict: bool = True) -> Tuple[List[SequenceRecord], List[str]]:
    """Discover RGB sequences and optionally collect malformed-sequence errors."""
    root_path = Path(root).expanduser().resolve()
    records: List[SequenceRecord] = []
    malformed: List[str] = []
    for dataset_id, dataset_directory in discover_dataset_directories(root_path, split).items():
        for keyframe_directory in _find_keyframe_directories(dataset_directory):
            try:
                frame_directory, frame_paths = select_frame_directory(keyframe_directory, frame_source)
                metadata = _optional_paths(keyframe_directory)
                keyframe_id = keyframe_directory.name
                records.append(SequenceRecord(dataset_id, keyframe_id, "dataset_{}/{}".format(dataset_id, keyframe_id), keyframe_directory, frame_directory, tuple(frame_paths), metadata["calibration_path"], metadata["depth_directory"], metadata["disparity_directory"], metadata["frame_data_directory"], metadata["reprojection_directory"], metadata["scene_points_directory"], metadata["point_cloud_path"], metadata["video_path"]))
            except MissingRGBFramesError as error:
                # Some official SCARED keyframes do not ship temporal RGB data.
                # They cannot form training clips, so skip them without aborting
                # discovery of the remaining usable sequences.
                message = "{}: {}".format(keyframe_directory, error)
                malformed.append(message)
                LOGGER.warning("Skipping keyframe without RGB frames: %s", keyframe_directory)
            except RuntimeError as error:
                message = "{}: {}".format(keyframe_directory, error)
                malformed.append(message)
                LOGGER.warning(message)
                if strict:
                    raise
    if not records:
        raise RuntimeError("No usable SCARED temporal RGB sequences discovered under {}".format(root_path))
    return records, malformed
