"""Fail-closed same-clip cache/checkpoint audit without loading DA3."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from datasets.crossclip_teacher_dataset import (
    attention_cache_key,
    crossclip_teacher_cache_path,
    make_teacher_cache_rgb_dataset,
    validate_attention_teacher_cache,
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
    dataset = make_teacher_cache_rgb_dataset(
        dataset_config, split, cache_root=None if dataset_only else cache_root
    )
    count = len(dataset) if limit is None or limit == 0 else min(limit, len(dataset))
    print("split={} sequences={} cache_stride8_samples={} auditing={}".format(split, len(dataset.sequences), len(dataset), count))
    expected_shape = (448, 560)
    attention_enabled = bool(config.get("attention_distill", {}).get("enabled", False))
    attention_layers = tuple(
        int(value) for value in config.get("attention_distill", {}).get("teacher_layers", ())
    )
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
        records = []
        cache_path = crossclip_teacher_cache_path(cache_root, metadata)
        if not dataset_only:
            if not cache_path.is_file():
                raise FileNotFoundError("Current stride8 cache missing: {}".format(cache_path))
            with np.load(str(cache_path), allow_pickle=False) as cache:
                validate_crossclip_teacher_cache(
                    cache, metadata, expected_shape,
                    str(teacher["pretrained_checkpoint"]), "raw",
                )
                if attention_enabled:
                    validate_attention_teacher_cache(cache, attention_layers)
                    if index == 0:
                        print(
                            "attention sample {}: {}".format(
                                cache_path,
                                {
                                    "layer_{}".format(layer): {
                                        "q": list(cache[attention_cache_key(layer, "q")].shape),
                                        "k": list(cache[attention_cache_key(layer, "k")].shape),
                                        "dtype": str(cache[attention_cache_key(layer, "q")].dtype),
                                    }
                                    for layer in attention_layers
                                },
                            )
                        )
                native = (
                    (int(cache["teacher_input_height"]), int(cache["teacher_input_width"]))
                    if "teacher_input_height" in cache else "not-recorded-v1"
                )
                teacher_start = int(cache["clip_start"].item())
                teacher_ids = [
                    int(value) for value in cache["absolute_frame_ids"].tolist()
                ]
            records.append(
                "teacher_start={} teacher_ids={} cache_shape=448x560 teacher_native={}".format(
                    teacher_start, teacher_ids, native,
                )
            )
        print(
            "sequence={} student_start={} absolute_ids={} {}".format(
                metadata["sequence_id"], start, ids, " | ".join(records)
            )
        )
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
