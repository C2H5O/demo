"""Safely convert complete cross-clip teacher NPZ files to ZIP_STORED in place."""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REQUIRED_CROSSCLIP_KEYS = frozenset(
    {
        "cache_format_version",
        "cache_stage",
        "sequence_id",
        "clip_start",
        "absolute_frame_ids",
        "depth",
        "xyz_local",
        "xyz_global",
        "confidence",
        "valid_mask",
        "highlight_mask",
        "intrinsics",
        "extrinsics",
        "metadata_json",
    }
)
LOCK_FILENAME = ".crossclip_cache_conversion.lock"
TEMPORARY_MARKER = ".crossclip-uncompressed-"
SUPPORTED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}


@dataclass(frozen=True)
class ConversionReport:
    discovered: int
    selected: int
    converted: int
    already_uncompressed: int
    source_bytes: int
    output_bytes: int
    dry_run: bool


def _archive_members(path: Path) -> Tuple[List[str], List[int]]:
    try:
        with zipfile.ZipFile(str(path), "r") as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            if not infos:
                raise ValueError("NPZ archive has no array members")
            if any(item.flag_bits & 0x1 for item in infos):
                raise ValueError("encrypted ZIP members are not supported")
            unsupported = {
                item.compress_type
                for item in infos
                if item.compress_type not in SUPPORTED_COMPRESSION
            }
            if unsupported:
                raise ValueError(
                    "unsupported ZIP compression methods: {}".format(
                        sorted(unsupported)
                    )
                )
            names = [item.filename for item in infos]
            if len(names) != len(set(names)):
                raise ValueError("NPZ archive contains duplicate members")
            return names, [item.compress_type for item in infos]
    except zipfile.BadZipFile as error:
        raise ValueError("invalid NPZ/ZIP archive: {}".format(path)) from error


def is_uncompressed_npz(path: Path) -> bool:
    """Return whether every array member uses ZIP_STORED."""
    _, methods = _archive_members(path)
    return all(method == zipfile.ZIP_STORED for method in methods)


def _npz_keys_from_members(members: Sequence[str]) -> List[str]:
    keys: List[str] = []
    for member in members:
        if not member.endswith(".npy") or member.startswith("/"):
            raise ValueError("unexpected NPZ member name: {!r}".format(member))
        key = member[:-4]
        if not key or ".." in Path(key).parts:
            raise ValueError("unsafe NPZ member name: {!r}".format(member))
        keys.append(key)
    return keys


def _assert_crossclip_keys(keys: Sequence[str], path: Path) -> None:
    missing = sorted(REQUIRED_CROSSCLIP_KEYS.difference(keys))
    if missing:
        raise ValueError(
            "refusing non-cross-clip or incomplete cache {}: missing {}".format(
                path, ", ".join(missing)
            )
        )


def _load_complete_cache(path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    try:
        with np.load(str(path), allow_pickle=False) as source:
            keys = list(source.files)
            _assert_crossclip_keys(keys, path)
            arrays = {key: np.asarray(source[key]).copy() for key in keys}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(
            "failed to read complete cache {}: {}".format(path, error)
        ) from error
    object_keys = [key for key, value in arrays.items() if value.dtype.hasobject]
    if object_keys:
        raise ValueError(
            "object arrays are forbidden with allow_pickle=False: {}".format(
                ", ".join(object_keys)
            )
        )
    return keys, arrays


def _source_signature(path: Path) -> Tuple[int, int, int, int]:
    metadata = path.stat()
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _estimated_npz_bytes(arrays: Mapping[str, np.ndarray]) -> int:
    # NPY headers and the ZIP central directory are tiny relative to dense maps,
    # but include a conservative fixed allowance per member.
    return sum(int(value.nbytes) + 4096 for value in arrays.values()) + 65536


def _check_temporary_space(parent: Path, estimated_bytes: int) -> None:
    free_bytes = int(shutil.disk_usage(str(parent)).free)
    safety_margin = max(64 * 1024 * 1024, estimated_bytes // 20)
    required = estimated_bytes + safety_margin
    if free_bytes < required:
        raise RuntimeError(
            "insufficient free space in {}: need at least {} bytes (including "
            "safety margin), have {}".format(parent, required, free_bytes)
        )


def _arrays_equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    if expected.dtype != actual.dtype or expected.shape != actual.shape:
        return False
    if expected.dtype.kind in "fc":
        return bool(np.array_equal(expected, actual, equal_nan=True))
    return bool(np.array_equal(expected, actual))


def _verify_uncompressed_copy(
    temporary: Path,
    expected_keys: Sequence[str],
    expected_arrays: Mapping[str, np.ndarray],
) -> None:
    members, methods = _archive_members(temporary)
    archive_keys = _npz_keys_from_members(members)
    if archive_keys != list(expected_keys):
        raise RuntimeError(
            "temporary archive keys/order changed: expected {}, got {}".format(
                list(expected_keys), archive_keys
            )
        )
    if any(method != zipfile.ZIP_STORED for method in methods):
        raise RuntimeError("temporary archive is not completely uncompressed")
    with zipfile.ZipFile(str(temporary), "r") as archive:
        corrupt_member = archive.testzip()
    if corrupt_member is not None:
        raise RuntimeError("temporary archive CRC failed for {}".format(corrupt_member))
    with np.load(str(temporary), allow_pickle=False) as converted:
        if list(converted.files) != list(expected_keys):
            raise RuntimeError("temporary archive changed its NumPy key list")
        for key in expected_keys:
            actual = np.asarray(converted[key])
            if not _arrays_equal(expected_arrays[key], actual):
                raise RuntimeError(
                    "temporary archive differs at key {!r} (shape/dtype/value)".format(
                        key
                    )
                )


def _sync_file(path: Path) -> None:
    # Windows rejects fsync on a read-only CRT descriptor; r+b is portable and
    # does not modify the already closed NumPy archive.
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def convert_one_cache_in_place(path: Path) -> str:
    """Atomically replace one complete compressed cache; return its result state."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("refusing symbolic-link cache: {}".format(path))
    if not path.is_file():
        raise FileNotFoundError("cache is not a regular file: {}".format(path))

    members, methods = _archive_members(path)
    _assert_crossclip_keys(_npz_keys_from_members(members), path)
    if all(method == zipfile.ZIP_STORED for method in methods):
        return "already_uncompressed"

    signature = _source_signature(path)
    source_mode = stat.S_IMODE(path.stat().st_mode)
    expected_keys, arrays = _load_complete_cache(path)
    estimated_bytes = _estimated_npz_bytes(arrays)
    _check_temporary_space(path.parent, estimated_bytes)
    temporary = path.with_name(
        ".{}{}{}-{}.tmp.npz".format(
            path.name,
            TEMPORARY_MARKER,
            os.getpid(),
            uuid.uuid4().hex,
        )
    )
    try:
        np.savez(str(temporary), **arrays)
        os.chmod(str(temporary), source_mode)
        _sync_file(temporary)
        _verify_uncompressed_copy(temporary, expected_keys, arrays)
        if _source_signature(path) != signature:
            raise RuntimeError(
                "source changed while converting; original was not replaced: {}".format(
                    path
                )
            )
        os.replace(str(temporary), str(path))
        _sync_directory(path.parent)
    except BaseException:
        # This process owns a UUID-named temporary. Removing it cannot affect the
        # original source, which remains intact until os.replace succeeds.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return "converted"


def _lock_payload(root: Path, token: str) -> str:
    return json.dumps(
        {
            "token": token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_unix": time.time(),
            "root": str(root),
        },
        sort_keys=True,
    )


@contextmanager
def _exclusive_conversion_lock(root: Path) -> Iterator[None]:
    lock_path = root / LOCK_FILENAME
    token = uuid.uuid4().hex
    payload = _lock_payload(root, token)
    try:
        descriptor = os.open(
            str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as error:
        try:
            owner = lock_path.read_text(encoding="utf-8")
        except OSError:
            owner = "<unreadable>"
        raise RuntimeError(
            "conversion lock already exists: {}\nowner: {}\n"
            "Do not remove it until that PID/job is confirmed stopped.".format(
                lock_path, owner
            )
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            current = lock_path.read_text(encoding="utf-8")
            if current == payload:
                lock_path.unlink()
                _sync_directory(root)
        except FileNotFoundError:
            pass


def _discover_cache_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for directory, directory_names, file_names in os.walk(
        str(root), followlinks=False
    ):
        parent = Path(directory)
        for name in list(directory_names):
            child = parent / name
            if child.is_symlink():
                raise ValueError("refusing symbolic-link directory: {}".format(child))
        for name in file_names:
            if not name.endswith(".npz"):
                continue
            path = parent / name
            if path.name.endswith(".tmp.npz") or TEMPORARY_MARKER in path.name:
                continue
            if path.is_symlink():
                raise ValueError("refusing symbolic-link cache: {}".format(path))
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    "cache escapes requested root: {}".format(path)
                ) from error
            if resolved.is_file():
                files.append(resolved)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _select_files(
    files: Sequence[Path],
    root: Path,
    start_at: Optional[str],
    limit: Optional[int],
) -> List[Path]:
    selected = list(files)
    if start_at is not None:
        requested = Path(start_at.replace("\\", "/"))
        if requested.is_absolute() or ".." in requested.parts:
            raise ValueError("--start-at must be a safe path relative to --root")
        requested_posix = requested.as_posix()
        positions = {
            path.relative_to(root).as_posix(): index
            for index, path in enumerate(selected)
        }
        if requested_posix not in positions:
            raise ValueError("--start-at cache was not found: {}".format(start_at))
        selected = selected[positions[requested_posix] :]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    return selected


def convert_crossclip_teacher_cache_root(
    root: Path,
    *,
    dry_run: bool = False,
    confirm_no_readers: bool = False,
    start_at: Optional[str] = None,
    limit: Optional[int] = None,
) -> ConversionReport:
    """Convert complete teacher caches under ``root`` sequentially and in place."""
    root = Path(root).expanduser()
    if root.is_symlink():
        raise ValueError("refusing symbolic-link cache root: {}".format(root))
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError("cache root is not a directory: {}".format(root))
    if root == Path(root.anchor):
        raise ValueError("refusing to scan a filesystem root")
    if not dry_run and not confirm_no_readers:
        raise ValueError(
            "conversion requires --confirm-no-readers after stopping every training, "
            "evaluation, visualization, and cache-generation process using this root"
        )

    files = _discover_cache_files(root)
    if not files:
        raise FileNotFoundError("no NPZ caches found under {}".format(root))
    selected = _select_files(files, root, start_at, limit)
    if not selected:
        raise ValueError("selection contains no cache files")

    converted = 0
    already = 0
    source_bytes = 0
    output_bytes = 0

    def process() -> None:
        nonlocal converted, already, source_bytes, output_bytes
        total = len(selected)
        for index, path in enumerate(selected, start=1):
            relative = path.relative_to(root).as_posix()
            before = path.stat().st_size
            source_bytes += before
            members, methods = _archive_members(path)
            _assert_crossclip_keys(_npz_keys_from_members(members), path)
            if all(method == zipfile.ZIP_STORED for method in methods):
                already += 1
                output_bytes += before
                print(
                    "[{}/{}] SKIP already uncompressed: {}".format(
                        index, total, relative
                    ),
                    flush=True,
                )
                continue
            if dry_run:
                print(
                    "[{}/{}] WOULD CONVERT: {} ({:.2f} MiB compressed)".format(
                        index, total, relative, before / (1024.0 ** 2)
                    ),
                    flush=True,
                )
                continue
            print(
                "[{}/{}] CONVERTING: {}".format(index, total, relative),
                flush=True,
            )
            state = convert_one_cache_in_place(path)
            if state != "converted":
                raise RuntimeError("unexpected conversion state: {}".format(state))
            after = path.stat().st_size
            converted += 1
            output_bytes += after
            print(
                "[{}/{}] VERIFIED + ATOMICALLY REPLACED: {} ({:.2f} MiB)".format(
                    index, total, relative, after / (1024.0 ** 2)
                ),
                flush=True,
            )

    if dry_run:
        process()
    else:
        with _exclusive_conversion_lock(root):
            # Validate every selected archive signature before making the first
            # replacement. This prevents discovering an unrelated/incomplete NPZ
            # only after earlier files have already been converted.
            for path in selected:
                members, _ = _archive_members(path)
                _assert_crossclip_keys(_npz_keys_from_members(members), path)
            process()

    return ConversionReport(
        discovered=len(files),
        selected=len(selected),
        converted=converted,
        already_uncompressed=already,
        source_bytes=source_bytes,
        output_bytes=output_bytes,
        dry_run=dry_run,
    )


__all__ = [
    "ConversionReport",
    "convert_crossclip_teacher_cache_root",
    "convert_one_cache_in_place",
    "is_uncompressed_npz",
]
