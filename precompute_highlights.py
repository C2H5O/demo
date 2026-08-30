"""Materialize per-frame highlight masks and inpainted student RGB."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from PIL import Image

from datasets.highlight import HighlightDetectionConfig, SpecularHighlightProcessor
from datasets.multidataset import discover_processed_scared_sequences
from datasets.precomputed_highlight import (
    HIGHLIGHT_MANIFEST_NAME,
    highlight_manifest_payload,
    parse_highlight_options,
    precomputed_highlight_paths,
)
from utils.config import load_config


_PROCESSOR: SpecularHighlightProcessor | None = None


def _initialize_worker(config: Dict[str, Any]) -> None:
    global _PROCESSOR
    import cv2

    cv2.setNumThreads(1)
    _PROCESSOR = SpecularHighlightProcessor(HighlightDetectionConfig(**config))


def _atomic_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp.png".format(path.name, os.getpid()))
    try:
        Image.fromarray(array).save(temporary, format="PNG", compress_level=1)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _process_frame(task: Tuple[str, str, str]) -> str:
    if _PROCESSOR is None:
        raise RuntimeError("Highlight worker was not initialized")
    source_value = task[0]
    source, mask_path, inpainted_path = map(Path, task)
    try:
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            if rgb.size != (560, 448):
                raise RuntimeError(
                    "student RGB {} has size {} but requires {}".format(
                        source, rgb.size, (560, 448)
                    )
                )
            array = np.asarray(rgb, dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise RuntimeError("Failed to decode student RGB {}: {}".format(source, error)) from error
    mask, inpainted = _PROCESSOR.process_numpy(array)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    inpainted_uint8 = np.clip(np.rint(inpainted * 255.0), 0, 255).astype(np.uint8)
    _atomic_png(mask_uint8, mask_path)
    _atomic_png(inpainted_uint8, inpainted_path)
    return source_value


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_tasks(
    tasks: Iterable[Tuple[str, str, str]], workers: int, config: Dict[str, Any]
) -> None:
    selected = list(tasks)
    if not selected:
        return
    if workers == 1:
        _initialize_worker(config)
        results = map(_process_frame, selected)
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers, initializer=_initialize_worker, initargs=(config,)
        )
        results = executor.map(_process_frame, selected, chunksize=1)
    try:
        for index, source in enumerate(results, start=1):
            if index == 1 or index % 10 == 0 or index == len(selected):
                print("[{}/{}] {}".format(index, len(selected), source), flush=True)
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute SCARED student highlight masks and inpainted RGB."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/vggtoda3.yaml"))
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    config = load_config(args.config)
    dataset_config = config["dataset"]
    enabled, detection, mask_directory, inpainted_directory = parse_highlight_options(
        dataset_config.get("highlight", {})
    )
    if not enabled:
        raise ValueError("dataset.highlight.enabled must be true for precomputation")
    sequences = discover_processed_scared_sequences(dataset_config["root"], args.split)
    if not sequences:
        raise RuntimeError("No processed SCARED sequences were discovered")

    tasks = []
    total_frames = 0
    for sequence in sequences:
        root = Path(sequence["keyframe_directory"])
        (root / mask_directory).mkdir(parents=True, exist_ok=True)
        (root / inpainted_directory).mkdir(parents=True, exist_ok=True)
        for source_value in sequence["frame_paths"]:
            source = Path(source_value)
            mask, inpainted = precomputed_highlight_paths(
                root, source.name, mask_directory, inpainted_directory
            )
            total_frames += 1
            if args.overwrite or not (mask.is_file() and inpainted.is_file()):
                tasks.append((str(source), str(mask), str(inpainted)))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    print(
        "highlight precompute: sequences={} frames={} selected={} workers={}".format(
            len(sequences), total_frames, len(tasks), args.workers
        ),
        flush=True,
    )
    detection_dict = {
        name: getattr(detection, name) for name in detection.__dataclass_fields__
    }
    _run_tasks(tasks, args.workers, detection_dict)

    if args.limit is not None:
        print("Partial run complete; no completion markers were written.", flush=True)
        return
    payloads = 0
    for sequence in sequences:
        root = Path(sequence["keyframe_directory"])
        for source_value in sequence["frame_paths"]:
            mask, inpainted = precomputed_highlight_paths(
                root, Path(source_value).name, mask_directory, inpainted_directory
            )
            if not mask.is_file() or not inpainted.is_file():
                raise RuntimeError("Precompute is incomplete under {}".format(root))
        _atomic_json(
            root / HIGHLIGHT_MANIFEST_NAME,
            highlight_manifest_payload(
                detection, int(sequence["sequence_length"]),
                mask_directory, inpainted_directory,
            ),
        )
        payloads += 1
    print("Highlight precompute complete: markers={}".format(payloads), flush=True)


if __name__ == "__main__":
    main()
