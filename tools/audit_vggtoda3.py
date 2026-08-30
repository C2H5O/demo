"""Fail-closed stride/cache/checkpoint audit without loading the DA3 model."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from datasets.crossclip_teacher_dataset import (
    build_neighbor_clip_indices,
    crossclip_teacher_cache_path,
    make_crossclip_rgb_dataset,
    validate_crossclip_teacher_cache,
)
from datasets.scared_clip_dataset import clip_metadata
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _audit_checkpoint(config: dict[str, Any]) -> None:
    student = config["student"]
    checkpoint = _project_path(str(student["checkpoint"]))
    model_config = _project_path(str(student["config_path"]))
    if not checkpoint.is_file() or not model_config.is_file():
        raise FileNotFoundError("DA3 checkpoint/config missing: {} {}".format(checkpoint, model_config))
    parsed = json.loads(model_config.read_text(encoding="utf-8"))
    if parsed.get("model_name") != "da3-small":
        raise RuntimeError("config.json is not da3-small")
    official = parsed.get("config", {})
    architecture = (
        official.get("net", {}).get("name"),
        official.get("head", {}).get("__object__", {}).get("name"),
        official.get("cam_enc", {}).get("__object__", {}).get("name"),
        official.get("cam_dec", {}).get("__object__", {}).get("name"),
    )
    if architecture != ("vits", "DualDPT", "CameraEnc", "CameraDec"):
        raise RuntimeError("Unexpected DA3-Small architecture {}".format(architecture))
    with checkpoint.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    keys = [key for key in header if key != "__metadata__"]
    counts = {
        "backbone": sum(key.startswith("model.backbone.") for key in keys),
        "depth_head": sum(key.startswith("model.head.") for key in keys),
        "camera_encoder": sum(key.startswith("model.cam_enc.") for key in keys),
        "camera_decoder": sum(key.startswith("model.cam_dec.") for key in keys),
    }
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError("Incomplete DA3 checkpoint: {}".format(counts))
    print("checkpoint={} format=safetensors architecture={} keys={} components={}".format(checkpoint, architecture, len(keys), counts))


def audit(config_path: Path, split: str, limit: int | None, dataset_only: bool) -> None:
    config = load_config(config_path)
    dataset_config = config["dataset"]
    if (int(dataset_config["clip_length"]), int(dataset_config["sample_stride"]), int(dataset_config["window_stride"])) != (16, 1, 8):
        raise RuntimeError("Dataset config must be clip_length=16 sample_stride=1 window_stride=8")
    teacher = config["teacher"]
    cache_root = Path(str(teacher["raw_cache_root"])) / split
    dataset = make_crossclip_rgb_dataset(
        dataset_config, split, cache_root=None if dataset_only else cache_root
    )
    neighbors = build_neighbor_clip_indices(dataset.clips, window_stride=8)
    count = len(dataset) if limit is None or limit == 0 else min(limit, len(dataset))
    print("split={} sequences={} stride8_samples={} auditing={}".format(split, len(dataset.sequences), len(dataset), count))
    expected_shape = (448, 560)
    for index in range(count):
        metadata = clip_metadata(dataset, index)
        start = int(metadata["clip_start"])
        if start % 8:
            raise RuntimeError("clip_start {} is not divisible by 8".format(start))
        ids = list(metadata["frame_indices"])
        if ids != list(range(ids[0], ids[0] + 16)):
            raise RuntimeError("Clip absolute frame IDs are not 16 consecutive IDs: {}".format(ids))
        missing_rgb = next(
            (path for path in metadata["frame_paths"] if not Path(path).is_file()),
            None,
        )
        if missing_rgb is not None:
            raise FileNotFoundError(
                "Student RGB frame referenced by the dataset/cache metadata is missing: {}. "
                "Teacher caches contain supervision, not RGB pixels; provide valid RGB "
                "frame_paths or regenerate the cache with paths valid on this machine.".format(
                    missing_rgb
                )
            )
        left_index, right_index = neighbors[index]
        side_records = []
        for side, neighbor_index in (("previous", left_index), ("next", right_index)):
            if neighbor_index is None:
                side_records.append("{}=None overlap=0".format(side))
                continue
            neighbor = clip_metadata(dataset, neighbor_index)
            expected_start = start + (-8 if side == "previous" else 8)
            if int(neighbor["clip_start"]) != expected_start:
                raise RuntimeError("{} neighbor start {} != {}".format(side, neighbor["clip_start"], expected_start))
            overlap = sorted(set(ids).intersection(neighbor["frame_indices"]))
            if len(overlap) != 8:
                raise RuntimeError("{} overlap is {} rather than 8: {}".format(side, len(overlap), overlap))
            neighbor_path = crossclip_teacher_cache_path(cache_root, neighbor)
            if not dataset_only and not neighbor_path.is_file():
                raise FileNotFoundError(
                    "Missing neighbor cache: dataset={} sequence={} clip_start={} expected_neighbor_start={} expected_cache_path={}".format(
                        metadata.get("dataset_name", metadata.get("dataset_id")), metadata["sequence_id"], start, expected_start, neighbor_path
                    )
                )
            side_records.append("{}_start={} overlap={} ids={}".format(side, expected_start, len(overlap), overlap))
        cache_path = crossclip_teacher_cache_path(cache_root, metadata)
        if not dataset_only:
            if not cache_path.is_file():
                raise FileNotFoundError("Current stride8 cache missing: {}".format(cache_path))
            with np.load(str(cache_path), allow_pickle=False) as cache:
                validate_crossclip_teacher_cache(
                    cache, metadata, expected_shape,
                    str(teacher["pretrained_checkpoint"]), "raw",
                )
                native = (
                    (int(cache["teacher_input_height"]), int(cache["teacher_input_width"]))
                    if "teacher_input_height" in cache else "not-recorded-v1"
                )
            side_records.append("cache_shape=448x560 teacher_native={}".format(native))
        print("sequence={} current_start={} absolute_ids={} {}".format(metadata["sequence_id"], start, ids, " | ".join(side_records)))
    _audit_checkpoint(config)
    print("VGGT-DA3 audit passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vggtoda3.yaml")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=5, help="0 validates all stride8 samples")
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()
    audit(Path(args.config), args.split, args.limit, args.dataset_only)


if __name__ == "__main__":
    main()
