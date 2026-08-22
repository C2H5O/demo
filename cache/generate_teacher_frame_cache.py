"""Generate one frozen base VGGT-Omega cache per source RGB frame."""

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from datasets.teacher_frame_cache import (
    FRAME_CACHE_FORMAT_VERSION,
    FRAME_COORDINATE_CONVENTION,
    frame_metadata,
    make_scared_frame_rgb_dataset,
    teacher_frame_cache_path,
    validate_teacher_frame_cache,
)
from models.teacher.output_adapter import adapt_teacher_outputs
from models.teacher.vggt_omega_wrapper import VGGTOmegaTeacher
from utils.config import load_config


def _write_cache(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    temporary.replace(path)


def generate_teacher_frame_cache(
    config_path: Path,
    split: str,
    limit: Optional[int] = None,
    overwrite: bool = False,
    cache_root_override: Optional[Path] = None,
) -> None:
    config = load_config(config_path)
    teacher_config = dict(config["teacher"])
    if not bool(teacher_config.get("frozen", True)):
        raise ValueError("Frame cache generation requires teacher.frozen=true")
    if str(teacher_config.get("variant", "base")) != "base":
        raise ValueError("Frame cache generation currently supports only teacher.variant=base")
    if teacher_config.get("lora_checkpoint"):
        raise ValueError("Per-frame cache generation must not use teacher.lora_checkpoint")
    if str(teacher_config.get("cache_protocol")) != "frame_local_v1":
        raise ValueError("Per-frame cache generation requires cache_protocol=frame_local_v1")
    if str(teacher_config.get("cache_dtype", "float32")).lower() != "float32":
        raise ValueError("Per-frame caches require cache_dtype=float32 to preserve detail")
    cache_root_value = str(cache_root_override) if cache_root_override is not None else teacher_config.get("cache_root")
    if not cache_root_value:
        raise ValueError("teacher.cache_root must be configured")

    dataset = make_scared_frame_rgb_dataset(config["dataset"], split)
    expected_shape = (
        int(teacher_config.get("image_height", 448)),
        int(teacher_config.get("image_width", 560)),
    )
    min_depth = float(teacher_config.get("min_depth", 0.1))
    max_depth = float(teacher_config.get("max_depth", 150.0))
    base_checkpoint = str(teacher_config["pretrained_checkpoint"])
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda":
        raise RuntimeError("VGGT-Omega frame cache generation requires CUDA")
    teacher = VGGTOmegaTeacher.from_config(
        teacher_config, device=device, load_lora=False, inject_lora=False
    ).freeze_for_distillation()
    if teacher.uses_lora:
        raise RuntimeError("Frame cache teacher unexpectedly contains LoRA adapters")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Teacher must be completely frozen")

    load_module = importlib.import_module("vggt_omega.utils.load_fn")
    cache_root = Path(cache_root_value) / split
    total = len(dataset) if limit is None else min(len(dataset), limit)
    written = skipped = 0
    for index in range(total):
        metadata = frame_metadata(dataset, index)
        path = teacher_frame_cache_path(cache_root, metadata)
        if path.is_file() and not overwrite:
            try:
                with np.load(str(path), allow_pickle=False) as existing:
                    validate_teacher_frame_cache(existing, metadata, expected_shape, base_checkpoint)
            except (OSError, RuntimeError, ValueError) as error:
                raise RuntimeError(
                    "Existing frame cache is stale or invalid; rerun with --overwrite: {} ({})".format(path, error)
                ) from error
            skipped += 1
            continue

        images = load_module.load_and_preprocess_images(
            [metadata["frame_path"]],
            mode=teacher_config.get("preprocess_mode", "max_size"),
            image_resolution=int(teacher_config.get("image_resolution", 560)),
        ).to(device, non_blocking=True)
        if tuple(images.shape) != (1, 3) + expected_shape:
            raise RuntimeError(
                "Frame preprocessing produced {} instead of [1,3,{},{}]".format(tuple(images.shape), *expected_shape)
            )
        with torch.inference_mode():
            raw = teacher(images)
            adapted = adapt_teacher_outputs(raw, expected_shape, min_depth, max_depth)
        expected_outputs = {
            "depth": (1, 1) + expected_shape,
            "xyz_local": (1, 1) + expected_shape + (3,),
            "conf_local": (1, 1) + expected_shape,
            "valid_mask": (1, 1) + expected_shape,
            "intrinsics": (1, 1, 3, 3),
            "extrinsics": (1, 1, 3, 4),
        }
        wrong = {
            key: tuple(adapted[key].shape)
            for key, expected in expected_outputs.items()
            if tuple(adapted[key].shape) != expected
        }
        if wrong:
            raise RuntimeError(
                "Single-frame teacher output contract failed: {} expected {}".format(
                    wrong, expected_outputs
                )
            )
        depth = adapted["depth"][0, 0].detach().cpu().float().numpy()
        xyz_local = adapted["xyz_local"][0, 0].detach().cpu().float().numpy()
        confidence = adapted["conf_local"][0, 0].detach().cpu().float().numpy()
        valid_mask = adapted["valid_mask"][0, 0].detach().cpu().numpy().astype(np.bool_)
        finite_outputs = {
            "depth": depth,
            "xyz_local": xyz_local,
            "confidence": confidence,
            "intrinsics": adapted["intrinsics"][0, 0].detach().cpu().float().numpy(),
            "extrinsics": adapted["extrinsics"][0, 0].detach().cpu().float().numpy(),
        }
        nonfinite = [
            name for name, value in finite_outputs.items() if not np.isfinite(value).all()
        ]
        if nonfinite:
            raise FloatingPointError(
                "Teacher produced non-finite {} for {}".format(
                    nonfinite, metadata["frame_path"]
                )
            )
        _write_cache(
            path,
            {
                "dataset_id": np.asarray(metadata["dataset_id"], dtype=np.int64),
                "keyframe_id": np.asarray(metadata["keyframe_id"], dtype=np.str_),
                "sequence_id": np.asarray(metadata["sequence_id"], dtype=np.str_),
                "frame_id": np.asarray(metadata["frame_id"], dtype=np.int64),
                "frame_index": np.asarray(metadata["frame_index"], dtype=np.int64),
                "frame_name": np.asarray(metadata["frame_name"], dtype=np.str_),
                "image_shape": np.asarray(expected_shape, dtype=np.int64),
                "teacher_variant": np.asarray("base", dtype=np.str_),
                "inference_frame_count": np.asarray(1, dtype=np.int64),
                "depth": depth,
                "xyz_local": xyz_local,
                "confidence": confidence,
                "valid_mask": valid_mask,
                "intrinsics": finite_outputs["intrinsics"],
                "extrinsics": finite_outputs["extrinsics"],
                "coordinate_convention": np.asarray(FRAME_COORDINATE_CONVENTION, dtype=np.str_),
                "cache_format_version": np.asarray(FRAME_CACHE_FORMAT_VERSION, dtype=np.str_),
                "base_checkpoint": np.asarray(base_checkpoint, dtype=np.str_),
                "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=np.str_),
            },
        )
        written += 1
        del images, raw, adapted, depth, xyz_local, confidence, valid_mask, finite_outputs
        if (index + 1) % 25 == 0 or index + 1 == total:
            print("[{}/{}] wrote={} skipped={} latest={}".format(index + 1, total, written, skipped, path))
        if (index + 1) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    print(
        "Frame cache complete: variant=base split={} frames={} written={} skipped={} root={}".format(
            split, total, written, skipped, cache_root
        )
    )


__all__ = ["generate_teacher_frame_cache"]
