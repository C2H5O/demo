"""Generate strict two-view VGGT-Omega caches in reference-camera coordinates."""

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from datasets.scared_pair_dataset import (
    PAIR_CACHE_FORMAT_VERSION,
    PAIR_COORDINATE_CONVENTION,
    make_scared_pair_rgb_dataset,
    pair_metadata,
    teacher_pair_cache_path,
)
from models.teacher.output_adapter import adapt_teacher_outputs
from models.teacher.vggt_omega_wrapper import VGGTOmegaTeacher
from utils.config import load_config
from utils.geometry import world_to_camera


def _write_cache(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    temporary.replace(path)


def _fp16(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().to(torch.float16).numpy()


def generate_teacher_pair_cache(
    config_path: Path,
    split: str,
    limit: Optional[int] = None,
    overwrite: bool = False,
    base_teacher: bool = False,
    cache_root_override: Optional[Path] = None,
) -> None:
    config = load_config(config_path)
    dataset_config = dict(config["dataset"])
    teacher_config = dict(config["teacher"])
    if not bool(teacher_config.get("frozen", True)):
        raise ValueError("Pair cache generation requires teacher.frozen=true")
    cache_root_value = (
        str(cache_root_override)
        if cache_root_override is not None
        else teacher_config.get("cache_root")
    )
    if not cache_root_value:
        raise ValueError("teacher.cache_root must be configured")
    if "teacher_cache_endodac_lora_448x560" in str(cache_root_value):
        raise ValueError("Refusing to overwrite the legacy eight-frame teacher cache")
    if base_teacher and cache_root_override is None:
        raise ValueError("--base-teacher requires a separate --cache-root")
    if not base_teacher and not teacher_config.get("lora_checkpoint"):
        raise ValueError("LoRA pair cache requires teacher.lora_checkpoint")

    dataset = make_scared_pair_rgb_dataset(dataset_config, split)
    expected_shape = (
        int(teacher_config.get("image_height", 448)),
        int(teacher_config.get("image_width", 560)),
    )
    min_depth = float(teacher_config.get("min_depth", 0.1))
    max_depth = float(teacher_config.get("max_depth", 150.0))
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda":
        raise RuntimeError("VGGT-Omega pair cache generation requires CUDA")
    teacher = VGGTOmegaTeacher.from_config(
        teacher_config,
        device=device,
        load_lora=not base_teacher,
        inject_lora=not base_teacher,
    ).freeze_for_distillation()
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Teacher must be completely frozen")

    load_module = importlib.import_module("vggt_omega.utils.load_fn")
    cache_root = Path(cache_root_value) / split
    total = len(dataset) if limit is None else min(len(dataset), limit)
    written = skipped = 0
    for index in range(total):
        metadata = pair_metadata(dataset, index)
        path = teacher_pair_cache_path(cache_root, metadata)
        if path.is_file() and not overwrite:
            skipped += 1
            continue
        images = load_module.load_and_preprocess_images(
            metadata["frame_paths"],
            mode=teacher_config.get("preprocess_mode", "max_size"),
            image_resolution=int(teacher_config.get("image_resolution", 560)),
        ).to(device, non_blocking=True)
        if tuple(images.shape) != (2, 3) + expected_shape:
            raise RuntimeError(
                "Pair preprocessing produced {} instead of [2,3,{},{}]".format(
                    tuple(images.shape), *expected_shape
                )
            )
        with torch.inference_mode():
            raw = teacher(images)
            adapted = adapt_teacher_outputs(
                raw,
                image_shape=expected_shape,
                min_depth=min_depth,
                max_depth=max_depth,
            )
        local = adapted["xyz_local"][0]
        world = adapted["xyz_global"][0]
        extrinsics = adapted["extrinsics"][0]
        if tuple(local.shape) != (2,) + expected_shape + (3,):
            raise RuntimeError("Teacher pair point-map shape mismatch: {}".format(tuple(local.shape)))
        pts_a_in_a = local[0]
        pts_b_in_a = world_to_camera(world[1], extrinsics[0])
        confidence = adapted["conf_local"][0]
        valid = adapted["valid_mask"][0]
        variant = "base" if base_teacher else "lora"
        _write_cache(
            path,
            {
                "frame_id_a": np.asarray(metadata["frame_id_a"], dtype=np.int64),
                "frame_id_b": np.asarray(metadata["frame_id_b"], dtype=np.int64),
                "frame_name_a": np.asarray(metadata["frame_name_a"], dtype=np.str_),
                "frame_name_b": np.asarray(metadata["frame_name_b"], dtype=np.str_),
                "pair_stride": np.asarray(metadata["pair_stride"], dtype=np.int64),
                "image_shape": np.asarray(expected_shape, dtype=np.int64),
                "teacher_variant": np.asarray(variant, dtype=np.str_),
                "depth_a": _fp16(adapted["depth"][0, 0]),
                "depth_b": _fp16(adapted["depth"][0, 1]),
                "xyz_local_a": _fp16(local[0]),
                "xyz_local_b": _fp16(local[1]),
                "xyz_global_a": _fp16(world[0]),
                "xyz_global_b": _fp16(world[1]),
                "pts3d_a_in_a": _fp16(pts_a_in_a),
                "pts3d_b_in_a": _fp16(pts_b_in_a),
                "confidence_a": _fp16(confidence[0]),
                "confidence_b": _fp16(confidence[1]),
                "valid_mask_a": valid[0].detach().cpu().numpy().astype(np.bool_),
                "valid_mask_b": valid[1].detach().cpu().numpy().astype(np.bool_),
                "intrinsics_a": adapted["intrinsics"][0, 0].detach().cpu().float().numpy(),
                "intrinsics_b": adapted["intrinsics"][0, 1].detach().cpu().float().numpy(),
                "extrinsics_a": extrinsics[0].detach().cpu().float().numpy(),
                "extrinsics_b": extrinsics[1].detach().cpu().float().numpy(),
                "coordinate_convention": np.asarray(PAIR_COORDINATE_CONVENTION, dtype=np.str_),
                "cache_format_version": np.asarray(PAIR_CACHE_FORMAT_VERSION, dtype=np.str_),
                "lora_checkpoint": np.asarray("" if base_teacher else str(teacher_config.get("lora_checkpoint", "")), dtype=np.str_),
                "metadata_json": np.asarray(json.dumps(metadata), dtype=np.str_),
            },
        )
        written += 1
        del images, raw, adapted, local, world, extrinsics
        if (index + 1) % 10 == 0 or index + 1 == total:
            print("[{}/{}] wrote={} skipped={} latest={}".format(index + 1, total, written, skipped, path))
        if (index + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    print(
        "Pair cache complete: variant={} split={} pairs={} written={} skipped={} root={}".format(
            "base" if base_teacher else "lora", split, total, written, skipped, cache_root
        )
    )


__all__ = ["generate_teacher_pair_cache"]
