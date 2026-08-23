"""Generate raw frozen-base VGGT-Omega caches for every stride-one 16-frame clip."""

from __future__ import annotations

import gc
import json
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

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
from datasets.scared_dataset import build_scared_dataloader
from datasets.transforms import unnormalize_image
from models.teacher.output_adapter import adapt_teacher_outputs
from models.teacher.vggt_omega_wrapper import VGGTOmegaTeacher
from utils.config import load_config


def _atomic_npz(
    path: Path, arrays: Dict[str, Any], compressed: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    save = np.savez_compressed if compressed else np.savez
    save(str(temporary), **arrays)
    temporary.replace(path)


def _teacher_amp_settings(
    teacher_config: Dict[str, Any], device: torch.device
) -> Tuple[bool, torch.dtype]:
    enabled = bool(teacher_config.get("amp", True))
    name = str(teacher_config.get("amp_dtype", "auto")).lower()
    if name == "auto":
        name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    dtypes = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in dtypes:
        raise ValueError("teacher.amp_dtype must be auto, bfloat16, or float16")
    if enabled and device.type != "cuda":
        raise ValueError("Teacher AMP is only supported on CUDA")
    return enabled, dtypes[name]


def _drain_completed_writes(
    futures: List[Future[None]], block: bool
) -> List[Future[None]]:
    if not futures:
        return futures
    done, pending = wait(
        futures,
        return_when=ALL_COMPLETED if block else FIRST_COMPLETED,
    )
    for future in done:
        future.result()
    return list(pending)


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
    minimum_valid_fraction = float(
        teacher_config.get("minimum_valid_fraction", 1.0e-3)
    )
    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise ValueError("teacher.minimum_valid_fraction must be in (0,1]")
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda":
        raise RuntimeError("VGGT-Omega cross-clip cache generation requires CUDA")
    teacher = VGGTOmegaTeacher.from_config(teacher_config, device=device)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Cross-clip teacher is not the fully frozen base model")

    amp_enabled, amp_dtype = _teacher_amp_settings(teacher_config, device)
    inference_batch_size = int(teacher_config.get("inference_batch_size", 1))
    if inference_batch_size <= 0:
        raise ValueError("teacher.inference_batch_size must be positive")
    loader_config = dict(config.get("teacher_dataloader", {}))
    write_workers = int(teacher_config.get("cache_write_workers", 2))
    if write_workers <= 0:
        raise ValueError("teacher.cache_write_workers must be positive")
    compressed = bool(teacher_config.get("cache_compressed", True))
    print(
        "Teacher inference: batch_size={} amp={} amp_dtype={} loader_workers={} "
        "write_workers={} compressed={}".format(
            inference_batch_size,
            amp_enabled,
            str(amp_dtype).replace("torch.", ""),
            int(loader_config.get("num_workers", 4)),
            write_workers,
            compressed,
        )
    )

    total = len(dataset) if limit is None else min(len(dataset), int(limit))
    skipped = 0
    pending_items = []
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
        pending_items.append((index, metadata, path))

    if not pending_items:
        print(
            "Raw cross-clip teacher cache complete: split={} clips={} wrote=0 skipped={} root={}".format(
                split, total, skipped, raw_root
            )
        )
        return

    loader = build_scared_dataloader(
        Subset(dataset, [item[0] for item in pending_items]),
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=int(loader_config.get("num_workers", 4)),
        pin_memory=bool(loader_config.get("pin_memory", True)),
        persistent_workers=bool(loader_config.get("persistent_workers", True)),
        prefetch_factor=int(loader_config.get("prefetch_factor", 2)),
        drop_last=False,
        seed=int(config.get("seed", 42)),
    )

    written = 0
    cursor = 0
    outstanding: List[Future[None]] = []
    max_outstanding = max(write_workers * 2, inference_batch_size)
    executor = ThreadPoolExecutor(max_workers=write_workers)
    try:
        for batch in loader:
            batch_size = int(batch["images"].shape[0])
            batch_items = pending_items[cursor : cursor + batch_size]
            cursor += batch_size

            # This is the exact deterministic student resize/crop result,
            # converted from configured [-1,1] to VGGT-Omega's [0,1] RGB input.
            images = unnormalize_image(
                batch["images"], dataset.normalize_mode
            ).to(device, non_blocking=True)
            if tuple(images.shape) != (batch_size, 16, 3) + expected_shape:
                raise RuntimeError(
                    "Cross-clip RGB shape contract failed: {}".format(
                        tuple(images.shape)
                    )
                )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                raw = teacher(images)
            # Camera decoding and unprojection stay in FP32 even when the network
            # forward uses BF16/FP16 Tensor Cores.
            adapted = adapt_teacher_outputs(
                {
                    name: raw[name].float()
                    for name in ("pose_enc", "depth", "depth_conf")
                },
                image_shape=expected_shape,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            expected_outputs = {
                "depth": (batch_size, 16) + expected_shape,
                "xyz_local": (batch_size, 16) + expected_shape + (3,),
                "xyz_global": (batch_size, 16) + expected_shape + (3,),
                "conf_local": (batch_size, 16) + expected_shape,
                "valid_mask": (batch_size, 16) + expected_shape,
                "intrinsics": (batch_size, 16, 3, 3),
                "extrinsics": (batch_size, 16, 3, 4),
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

            for batch_index, (_, metadata, path) in enumerate(batch_items):
                valid_array = (
                    adapted["valid_mask"][batch_index]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.bool_)
                )
                valid_fraction = (
                    valid_array.reshape(16, -1).mean(axis=1).astype(np.float32)
                )
                bad_frames = np.flatnonzero(
                    valid_fraction < minimum_valid_fraction
                ).tolist()
                if bad_frames:
                    raise RuntimeError(
                        "Teacher valid fraction fell below {} for frames {} in {}".format(
                            minimum_valid_fraction, bad_frames, path
                        )
                    )

                def dense_fp32(name: str) -> np.ndarray:
                    value = (
                        adapted[name][batch_index]
                        .detach()
                        .cpu()
                        .float()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    valid = valid_array[..., None] if value.ndim == 4 else valid_array
                    valid = np.broadcast_to(valid, value.shape)
                    if not np.isfinite(value[valid]).all():
                        raise FloatingPointError(
                            "Teacher {} is non-finite at valid pixels".format(name)
                        )
                    return np.where(valid, value, 0.0).astype(
                        np.float32, copy=False
                    )

                def camera_fp32(name: str) -> np.ndarray:
                    value = (
                        adapted[name][batch_index]
                        .detach()
                        .cpu()
                        .float()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    if not np.isfinite(value).all():
                        raise FloatingPointError(
                            "Teacher {} contains a non-finite camera matrix".format(
                                name
                            )
                        )
                    return value

                depth_array = dense_fp32("depth")
                local_array = dense_fp32("xyz_local")
                global_array = dense_fp32("xyz_global")
                confidence_array = dense_fp32("conf_local")
                intrinsics_array = camera_fp32("intrinsics")
                extrinsics_array = camera_fp32("extrinsics")
                valid_depth = depth_array[valid_array]
                valid_confidence = confidence_array[valid_array]

                highlight = batch.get("highlight_masks")
                if highlight is None:
                    highlight_array = np.zeros(
                        (16,) + expected_shape, dtype=np.bool_
                    )
                else:
                    highlight_array = (
                        highlight[batch_index, :, 0].cpu().numpy().astype(np.bool_)
                    )
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
                    "minimum_valid_fraction": minimum_valid_fraction,
                    "valid_fraction_per_frame": valid_fraction.tolist(),
                    "valid_depth_min": float(valid_depth.min()),
                    "valid_depth_max": float(valid_depth.max()),
                    "valid_confidence_mean": float(valid_confidence.mean()),
                }
                arrays = {
                    "sequence_id": np.asarray(
                        metadata["sequence_id"], dtype=np.str_
                    ),
                    "clip_start": np.asarray(
                        metadata["clip_start"], dtype=np.int64
                    ),
                    "absolute_frame_ids": np.asarray(
                        metadata["frame_indices"], dtype=np.int64
                    ),
                    "frame_names": np.asarray(
                        metadata["frame_names"], dtype=np.str_
                    ),
                    "input_height": np.asarray(
                        expected_shape[0], dtype=np.int64
                    ),
                    "input_width": np.asarray(
                        expected_shape[1], dtype=np.int64
                    ),
                    "depth": depth_array,
                    "xyz_local": local_array,
                    "xyz_global": global_array,
                    "confidence": confidence_array,
                    "valid_mask": valid_array,
                    "highlight_mask": highlight_array,
                    "intrinsics": intrinsics_array,
                    "extrinsics": extrinsics_array,
                    "pose_convention": np.asarray(
                        WORLD_TO_CAMERA_POSE_CONVENTION, dtype=np.str_
                    ),
                    "point_coordinate_system": np.asarray(
                        LOCAL_CAMERA_COORDINATE_SYSTEM, dtype=np.str_
                    ),
                    "teacher_variant": np.asarray("base", dtype=np.str_),
                    "base_checkpoint": np.asarray(
                        base_checkpoint, dtype=np.str_
                    ),
                    "cache_stage": np.asarray("raw", dtype=np.str_),
                    "alignment_scale": np.asarray(1.0, dtype=np.float32),
                    "cache_format_version": np.asarray(
                        CROSSCLIP_CACHE_FORMAT_VERSION, dtype=np.str_
                    ),
                    "metadata_json": np.asarray(
                        json.dumps(metadata_record, ensure_ascii=False),
                        dtype=np.str_,
                    ),
                }
                outstanding.append(
                    executor.submit(_atomic_npz, path, arrays, compressed)
                )
                written += 1
                if len(outstanding) >= max_outstanding:
                    outstanding = _drain_completed_writes(
                        outstanding, block=False
                    )

            del batch, images, raw, adapted
            processed = skipped + written
            if processed % 10 < batch_size or written == len(pending_items):
                print(
                    "[{}/{}] raw cross-clip caches: wrote={} skipped={} latest={}".format(
                        processed, total, written, skipped, batch_items[-1][2]
                    )
                )
            if written % 25 < batch_size:
                gc.collect()
        outstanding = _drain_completed_writes(outstanding, block=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
    print(
        "Raw cross-clip teacher cache complete: split={} clips={} wrote={} skipped={} root={}".format(
            split, total, written, skipped, raw_root
        )
    )


__all__ = ["generate_crossclip_teacher_cache"]
