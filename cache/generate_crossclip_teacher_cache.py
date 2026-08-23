"""Generate raw frozen-base VGGT-Omega caches for every stride-one 16-frame clip."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from datasets.crossclip_teacher_dataset import (
    CROSSCLIP_CACHE_FORMAT_VERSION,
    CROSSCLIP_CACHE_PROTOCOL,
    LOCAL_CAMERA_COORDINATE_SYSTEM,
    WORLD_TO_CAMERA_POSE_CONVENTION,
    crossclip_teacher_cache_path,
    make_crossclip_rgb_dataset,
    validate_crossclip_teacher_cache,
)
from datasets.scared_clip_dataset import clip_metadata
from datasets.transforms import unnormalize_image
from models.teacher.output_adapter import adapt_teacher_outputs
from models.teacher.vggt_omega_wrapper import VGGTOmegaTeacher
from utils.config import load_config


def _atomic_npz(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    temporary.replace(path)


def generate_crossclip_teacher_cache(
    config_path: Path,
    split: str,
    limit: Optional[int] = None,
    overwrite: bool = False,
    cache_root_override: Optional[Path] = None,
) -> None:
    config = load_config(config_path)
    teacher_config = dict(config["teacher"])
    dataset_config = dict(config["dataset"])
    if str(teacher_config.get("cache_protocol")) != CROSSCLIP_CACHE_PROTOCOL:
        raise ValueError("Teacher cache_protocol must be crossclip_local_v1")
    if str(teacher_config.get("variant")) != "base":
        raise ValueError("Cross-clip caches require the pretrained base teacher")
    if not bool(teacher_config.get("frozen", True)):
        raise ValueError("Cross-clip teacher must be frozen")
    if teacher_config.get("lora_checkpoint"):
        raise ValueError("Cross-clip teacher cache generation forbids LoRA/fine-tuning")
    if str(teacher_config.get("cache_dtype", "float32")).lower() != "float32":
        raise ValueError("Cross-clip teacher caches require FP32 storage")
    if int(dataset_config.get("clip_length", 16)) != 16:
        raise ValueError("dataset.clip_length must be 16")
    raw_root = (
        Path(cache_root_override)
        if cache_root_override is not None
        else Path(str(teacher_config["raw_cache_root"]))
    ) / split
    dataset = make_crossclip_rgb_dataset(dataset_config, split)
    expected_shape = (
        int(dataset_config["image_height"]),
        int(dataset_config["image_width"]),
    )
    base_checkpoint = str(teacher_config["pretrained_checkpoint"])
    min_depth = float(teacher_config.get("min_depth", 0.1))
    max_depth = float(teacher_config.get("max_depth", 150.0))
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda":
        raise RuntimeError("VGGT-Omega cross-clip cache generation requires CUDA")
    teacher = VGGTOmegaTeacher.from_config(
        teacher_config,
        device=device,
        load_lora=False,
        inject_lora=False,
    ).freeze_for_distillation()
    if teacher.uses_lora or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Cross-clip teacher is not the fully frozen base model")

    total = len(dataset) if limit is None else min(len(dataset), int(limit))
    written = skipped = 0
    for index in range(total):
        metadata = clip_metadata(dataset, index)
        path = crossclip_teacher_cache_path(raw_root, metadata)
        if path.is_file() and not overwrite:
            with np.load(str(path), allow_pickle=False) as existing:
                validate_crossclip_teacher_cache(
                    existing,
                    metadata,
                    expected_shape,
                    base_checkpoint,
                    "raw",
                )
            skipped += 1
            continue

        sample = dataset[index]
        # This is the exact deterministic student resize/crop result, converted
        # from configured [-1,1] to VGGT-Omega's [0,1] RGB input.
        images = unnormalize_image(
            sample["images"], dataset.normalize_mode
        ).to(device, non_blocking=True)
        if tuple(images.shape) != (16, 3) + expected_shape:
            raise RuntimeError("Cross-clip RGB shape contract failed: {}".format(tuple(images.shape)))
        with torch.inference_mode():
            raw = teacher(images)
            adapted = adapt_teacher_outputs(
                raw,
                image_shape=expected_shape,
                min_depth=min_depth,
                max_depth=max_depth,
            )
        expected_outputs = {
            "depth": (1, 16) + expected_shape,
            "xyz_local": (1, 16) + expected_shape + (3,),
            "xyz_global": (1, 16) + expected_shape + (3,),
            "conf_local": (1, 16) + expected_shape,
            "valid_mask": (1, 16) + expected_shape,
            "intrinsics": (1, 16, 3, 3),
            "extrinsics": (1, 16, 3, 4),
        }
        wrong = {
            key: tuple(adapted[key].shape)
            for key, expected in expected_outputs.items()
            if tuple(adapted[key].shape) != expected
        }
        if wrong:
            raise RuntimeError(
                "16-frame teacher output contract failed: {} expected {}".format(
                    wrong, expected_outputs
                )
            )

        def fp32(name: str) -> np.ndarray:
            value = adapted[name][0].detach().cpu().float().numpy()
            return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32, copy=False
            )

        highlight = sample.get("highlight_masks")
        if highlight is None:
            highlight_array = np.zeros((16,) + expected_shape, dtype=np.bool_)
        else:
            highlight_array = highlight[:, 0].cpu().numpy().astype(np.bool_)
        metadata_record = {
            **metadata,
            "input_height": expected_shape[0],
            "input_width": expected_shape[1],
            "resize_mode": dataset.resize_mode,
            "normalize_mode": dataset.normalize_mode,
            "pose_convention": WORLD_TO_CAMERA_POSE_CONVENTION,
            "point_coordinate_system": LOCAL_CAMERA_COORDINATE_SYSTEM,
            "teacher_variant": "base",
            "base_checkpoint": base_checkpoint,
            "cache_stage": "raw",
        }
        _atomic_npz(
            path,
            {
                "sequence_id": np.asarray(metadata["sequence_id"], dtype=np.str_),
                "clip_start": np.asarray(metadata["clip_start"], dtype=np.int64),
                "absolute_frame_ids": np.asarray(metadata["frame_indices"], dtype=np.int64),
                "frame_names": np.asarray(metadata["frame_names"], dtype=np.str_),
                "input_height": np.asarray(expected_shape[0], dtype=np.int64),
                "input_width": np.asarray(expected_shape[1], dtype=np.int64),
                "depth": fp32("depth"),
                "xyz_local": fp32("xyz_local"),
                "xyz_global": fp32("xyz_global"),
                "confidence": fp32("conf_local"),
                "valid_mask": adapted["valid_mask"][0].detach().cpu().numpy().astype(np.bool_),
                "highlight_mask": highlight_array,
                "intrinsics": fp32("intrinsics"),
                "extrinsics": fp32("extrinsics"),
                "pose_convention": np.asarray(WORLD_TO_CAMERA_POSE_CONVENTION, dtype=np.str_),
                "point_coordinate_system": np.asarray(LOCAL_CAMERA_COORDINATE_SYSTEM, dtype=np.str_),
                "teacher_variant": np.asarray("base", dtype=np.str_),
                "base_checkpoint": np.asarray(base_checkpoint, dtype=np.str_),
                "cache_stage": np.asarray("raw", dtype=np.str_),
                "alignment_scale": np.asarray(1.0, dtype=np.float32),
                "cache_format_version": np.asarray(CROSSCLIP_CACHE_FORMAT_VERSION, dtype=np.str_),
                "metadata_json": np.asarray(
                    json.dumps(metadata_record, ensure_ascii=False), dtype=np.str_
                ),
            },
        )
        written += 1
        del sample, images, raw, adapted
        if (index + 1) % 10 == 0 or index + 1 == total:
            print(
                "[{}/{}] raw cross-clip caches: wrote={} skipped={} latest={}".format(
                    index + 1, total, written, skipped, path
                )
            )
        if (index + 1) % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    print(
        "Raw cross-clip teacher cache complete: split={} clips={} wrote={} skipped={} root={}".format(
            split, total, written, skipped, raw_root
        )
    )


__all__ = ["generate_crossclip_teacher_cache"]
