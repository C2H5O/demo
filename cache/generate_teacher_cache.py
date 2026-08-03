"""Generate frozen VGGT-Omega + LoRA caches for configured SCARED clips."""

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from datasets.scared_clip_dataset import (
    clip_metadata,
    make_scared_rgb_dataset,
    teacher_cache_path,
)
from models.teacher.output_adapter import adapt_teacher_outputs
from models.teacher.vggt_omega_wrapper import VGGTOmegaTeacher
from utils.config import load_config


def _write_cache(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    temporary.replace(path)


def generate_teacher_cache(
    config_path: Path,
    split: str,
    limit: Optional[int] = None,
    overwrite: bool = False,
    base_teacher: bool = False,
    cache_root_override: Optional[Path] = None,
) -> None:
    config = load_config(config_path)
    teacher_config = dict(config["teacher"])
    if not bool(teacher_config.get("frozen", True)):
        raise ValueError("Cache generation requires teacher.frozen=true")
    cache_root_value = (
        str(cache_root_override)
        if cache_root_override is not None
        else teacher_config.get("cache_root")
    )
    if not cache_root_value:
        raise ValueError("teacher.cache_root must be configured")
    if base_teacher and cache_root_override is None:
        raise ValueError(
            "Base-teacher cache generation requires a separate "
            "cache_root_override to avoid mixing base and LoRA caches"
        )
    if not base_teacher and not teacher_config.get("lora_checkpoint"):
        raise ValueError(
            "teacher.lora_checkpoint must point to the adapted stage-one LoRA checkpoint"
        )
    dataset = make_scared_rgb_dataset(config["dataset"], split)
    min_depth = float(teacher_config.get("min_depth", 0.1))
    max_depth = float(teacher_config.get("max_depth", 150.0))
    expected_shape = (
        int(teacher_config.get("image_height", 448)),
        int(teacher_config.get("image_width", 560)),
    )
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda":
        raise RuntimeError("VGGT-Omega cache generation requires CUDA")
    teacher = VGGTOmegaTeacher.from_config(
        teacher_config,
        device=device,
        load_lora=not base_teacher,
        inject_lora=not base_teacher,
    ).freeze_for_distillation()
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Teacher must be completely frozen during cache generation")

    load_module = importlib.import_module("vggt_omega.utils.load_fn")
    cache_root = Path(cache_root_value) / split
    total = len(dataset) if limit is None else min(len(dataset), limit)
    written = 0
    skipped = 0
    for index in range(total):
        metadata = clip_metadata(dataset, index)
        path = teacher_cache_path(cache_root, metadata)
        if path.is_file() and not overwrite:
            skipped += 1
            continue
        images = load_module.load_and_preprocess_images(
            metadata["frame_paths"],
            mode=teacher_config.get("preprocess_mode", "balanced"),
            image_resolution=int(teacher_config.get("image_resolution", 512)),
        ).to(device, non_blocking=True)
        if tuple(images.shape[-2:]) != expected_shape:
            raise RuntimeError(
                "Teacher preprocessing produced {} instead of configured {} "
                "for {}. Check preprocess_mode, image_resolution, and source "
                "RGB aspect ratio.".format(
                    tuple(images.shape[-2:]), expected_shape, path
                )
            )
        with torch.inference_mode():
            raw = teacher(images)
            adapted = adapt_teacher_outputs(
                raw,
                image_shape=tuple(images.shape[-2:]),
                min_depth=min_depth,
                max_depth=max_depth,
            )
        if tuple(adapted["xyz_local"].shape[-3:-1]) != expected_shape:
            raise RuntimeError(
                "VGGT-Omega output produced {} instead of configured {} for {}".format(
                    tuple(adapted["xyz_local"].shape[-3:-1]),
                    expected_shape,
                    path,
                )
            )
        _write_cache(
            path,
            {
                "xyz_local": adapted["xyz_local"][0].detach().cpu().to(torch.float16).numpy(),
                "xyz_global": adapted["xyz_global"][0].detach().cpu().to(torch.float16).numpy(),
                "conf_local": adapted["conf_local"][0].detach().cpu().to(torch.float16).numpy(),
                "conf_global": adapted["conf_global"][0].detach().cpu().to(torch.float16).numpy(),
                "valid_mask": adapted["valid_mask"][0].detach().cpu().numpy().astype(np.bool_),
                "frame_names": np.asarray(metadata["frame_names"], dtype=np.str_),
                "frame_indices": np.asarray(metadata["frame_indices"], dtype=np.int64),
                "dataset_id": np.asarray(metadata["dataset_id"], dtype=np.int64),
                "clip_start": np.asarray(metadata["clip_start"], dtype=np.int64),
                "teacher_image_shape": np.asarray(images.shape[-2:], dtype=np.int64),
                "teacher_output_shape": np.asarray(
                    adapted["xyz_local"].shape[-3:-1], dtype=np.int64
                ),
                "teacher_depth_range": np.asarray([min_depth, max_depth], dtype=np.float32),
                "teacher_variant": np.asarray(
                    "base" if base_teacher else "lora", dtype=np.str_
                ),
                "lora_checkpoint": np.asarray(
                    ""
                    if base_teacher
                    else str(teacher_config.get("lora_checkpoint", "")),
                    dtype=np.str_,
                ),
                "intrinsics": adapted["intrinsics"][0].detach().cpu().float().numpy(),
                "extrinsics": adapted["extrinsics"][0].detach().cpu().float().numpy(),
                "metadata_json": np.asarray(json.dumps(metadata), dtype=np.str_),
            },
        )
        written += 1
        del images, raw, adapted
        if (index + 1) % 10 == 0 or index + 1 == total:
            print(
                "[{}/{}] wrote={} skipped={} latest={}".format(
                    index + 1, total, written, skipped, path
                )
            )
        if (index + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    print(
        "Cache generation complete: variant={} split={} clips={} written={} "
        "skipped={} root={}".format(
            "base" if base_teacher else "lora",
            split,
            total,
            written,
            skipped,
            cache_root,
        )
    )
