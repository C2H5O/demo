"""Generate raw frozen-base VGGT-Omega caches for legal 16-frame clip starts."""

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
import torch.nn.functional as F
from torch.utils.data import Subset

from datasets.crossclip_teacher_dataset import (
    ATTENTION_CACHE_SCHEMA_VERSION,
    CROSSCLIP_ATTENTION_CACHE_FORMAT_VERSION,
    CROSSCLIP_CACHE_FORMAT_VERSION,
    CROSSCLIP_CACHE_PROTOCOL,
    LOCAL_CAMERA_COORDINATE_SYSTEM,
    WORLD_TO_CAMERA_POSE_CONVENTION,
    attention_cache_key,
    crossclip_teacher_cache_path,
    make_teacher_cache_rgb_dataset,
    validate_crossclip_teacher_cache,
    validate_attention_teacher_cache,
)
from datasets.scared_clip_dataset import clip_metadata
from datasets.scared_dataset import build_scared_dataloader
from datasets.multidataset import TeacherClipInputDataset
from models.teacher.output_adapter import adapt_teacher_outputs
from models.teacher.vggt_omega_wrapper import VGGTOmegaTeacher
from utils.config import load_config


TEACHER_SHAPE = (1024, 1280)
SUPERVISION_SHAPE = (448, 560)


def attention_cache_bytes_per_clip(
    layers: int,
    frames: int,
    tokens: int,
    heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    return layers * 2 * frames * tokens * heads * head_dim * bytes_per_element


def canonicalize_teacher_outputs(adapted: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Project native teacher maps onto the immutable student/cache grid.

    Continuous maps use a valid-aware bilinear sampling grid so invalid zeros do
    not bleed into valid geometry.  Binary masks use nearest-neighbour sampling.
    Depth is set from the resampled local XYZ Z component to preserve the cache
    validator's camera-coordinate invariant exactly.
    """
    valid = adapted["valid_mask"].bool()
    if tuple(valid.shape[-2:]) != TEACHER_SHAPE:
        raise ValueError("teacher outputs must be native 1024x1280")
    batch, frames = valid.shape[:2]
    flat_valid = valid.reshape(batch * frames, 1, *TEACHER_SHAPE).float()

    def continuous(value: torch.Tensor) -> torch.Tensor:
        channels_last = value.ndim == 5
        if channels_last:
            value = value.permute(0, 1, 4, 2, 3)
        if not channels_last:
            value = value.unsqueeze(2)
        flat = value.reshape(batch * frames, value.shape[2], *TEACHER_SHAPE).float()
        weight = F.interpolate(flat_valid, size=SUPERVISION_SHAPE, mode="bilinear", align_corners=False)
        sampled = F.interpolate(flat * flat_valid, size=SUPERVISION_SHAPE, mode="bilinear", align_corners=False) / weight.clamp_min(1.0e-6)
        sampled = torch.where(weight > 1.0e-6, sampled, torch.zeros_like(sampled))
        sampled = sampled.reshape(batch, frames, sampled.shape[1], *SUPERVISION_SHAPE)
        return sampled.permute(0, 1, 3, 4, 2).contiguous() if channels_last else sampled[:, :, 0]

    output_valid = F.interpolate(flat_valid, size=SUPERVISION_SHAPE, mode="nearest").reshape(batch, frames, *SUPERVISION_SHAPE) > 0.5
    local = continuous(adapted["xyz_local"])
    output = {
        "xyz_local": local,
        "xyz_global": continuous(adapted["xyz_global"]),
        "confidence": continuous(adapted["conf_local"]),
        "valid_mask": output_valid,
        "intrinsics": adapted["intrinsics"].float().clone(),
        "extrinsics": adapted["extrinsics"].float(),
    }
    output["depth"] = torch.where(output_valid, local[..., 2], torch.zeros_like(local[..., 2]))
    output["xyz_local"] = torch.where(output_valid[..., None], local, torch.zeros_like(local))
    output["xyz_global"] = torch.where(output_valid[..., None], output["xyz_global"], torch.zeros_like(output["xyz_global"]))
    output["confidence"] = torch.where(output_valid, output["confidence"], torch.zeros_like(output["confidence"]))
    sx, sy = SUPERVISION_SHAPE[1] / TEACHER_SHAPE[1], SUPERVISION_SHAPE[0] / TEACHER_SHAPE[0]
    output["intrinsics"][..., 0, 0] *= sx
    output["intrinsics"][..., 1, 1] *= sy
    output["intrinsics"][..., 0, 2] *= sx
    output["intrinsics"][..., 1, 2] *= sy
    return output


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


def _resolve_start_index(
    dataset: Any,
    start_index: Optional[int],
    start_dataset_id: Optional[int],
    start_keyframe_id: Optional[str],
    start_clip_start: Optional[int],
) -> int:
    has_location = start_dataset_id is not None or start_keyframe_id is not None
    if start_index is not None and has_location:
        raise ValueError(
            "Use either --start-index or dataset/keyframe location, not both"
        )
    if start_index is not None:
        resolved = int(start_index)
        if not 0 <= resolved < len(dataset):
            raise ValueError(
                "start_index {} is outside [0,{})".format(resolved, len(dataset))
            )
        return resolved
    if not has_location:
        if start_clip_start is not None:
            raise ValueError(
                "--start-clip-start requires --start-dataset-id and "
                "--start-keyframe-id"
            )
        return 0
    if start_dataset_id is None or start_keyframe_id is None:
        raise ValueError(
            "--start-dataset-id and --start-keyframe-id must be used together"
        )
    requested_clip_start = 0 if start_clip_start is None else int(start_clip_start)
    for index in range(len(dataset)):
        metadata = clip_metadata(dataset, index)
        if (
            int(metadata["dataset_id"]) == int(start_dataset_id)
            and str(metadata["keyframe_id"]) == str(start_keyframe_id)
            and int(metadata["clip_start"]) == requested_clip_start
        ):
            return index
    raise ValueError(
        "No clip matches dataset_id={} keyframe_id={!r} clip_start={}".format(
            start_dataset_id, start_keyframe_id, requested_clip_start
        )
    )


def generate_crossclip_teacher_cache(
    config_path: Path,
    split: str,
    limit: Optional[int] = None,
    overwrite: bool = False,
    cache_root_override: Optional[Path] = None,
    start_index: Optional[int] = None,
    start_dataset_id: Optional[int] = None,
    start_keyframe_id: Optional[str] = None,
    start_clip_start: Optional[int] = None,
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
    save_attention = bool(teacher_config.get("save_attention", False))
    attention_layers = tuple(
        int(value) for value in teacher_config.get("attention_layers", (4, 11, 17, 23))
    )
    attention_dtype = str(teacher_config.get("attention_cache_dtype", "float16")).lower()
    if save_attention and attention_dtype not in {"float16", "fp16", "float32"}:
        raise ValueError("teacher.attention_cache_dtype must be float16 or float32")
    if int(dataset_config.get("clip_length", 16)) != 16:
        raise ValueError("dataset.clip_length must be 16")
    raw_root = (
        Path(cache_root_override)
        if cache_root_override is not None
        else Path(str(teacher_config["raw_cache_root"]))
    ) / split
    dataset = make_teacher_cache_rgb_dataset(dataset_config, split)
    expected_shape = (
        int(dataset_config["image_height"]),
        int(dataset_config["image_width"]),
    )
    if expected_shape != SUPERVISION_SHAPE:
        raise ValueError("cross-clip supervision must remain 448x560")
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
    if save_attention:
        bytes_per_element = 2 if attention_dtype in {"float16", "fp16"} else 4
        estimated = attention_cache_bytes_per_clip(
            len(attention_layers), 16, 64 * 80, 16, 64, bytes_per_element
        )
        print(
            "Teacher attention cache: layers={} Q/K grid=64x80 heads=16 head_dim=64 "
            "dtype={} uncompressed_per_clip={:.3f} GiB".format(
                list(attention_layers), attention_dtype, estimated / float(1024 ** 3)
            )
        )

    first_index = _resolve_start_index(
        dataset,
        start_index,
        start_dataset_id,
        start_keyframe_id,
        start_clip_start,
    )
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive")
    stop_index = (
        len(dataset)
        if limit is None
        else min(len(dataset), first_index + int(limit))
    )
    total = stop_index - first_index
    first_metadata = clip_metadata(dataset, first_index)
    print(
        "Selected cache range: global_index=[{},{}) clips={} starts_at={}/{} "
        "clip_start={}".format(
            first_index,
            stop_index,
            total,
            first_metadata["dataset_id"],
            first_metadata["keyframe_id"],
            first_metadata["clip_start"],
        )
    )
    skipped = 0
    pending_items = []
    for index in range(first_index, stop_index):
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
                if save_attention:
                    validate_attention_teacher_cache(
                        existing, attention_layers, check_finite=True
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

    teacher_dataset = TeacherClipInputDataset(dataset)
    loader = build_scared_dataloader(
        Subset(teacher_dataset, [item[0] for item in pending_items]),
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
    printed_attention_sample = False
    cursor = 0
    outstanding: List[Future[None]] = []
    max_outstanding = max(write_workers * 2, inference_batch_size)
    executor = ThreadPoolExecutor(max_workers=write_workers)
    try:
        for batch in loader:
            batch_size = int(batch["images"].shape[0])
            batch_items = pending_items[cursor : cursor + batch_size]
            cursor += batch_size

            # Teacher RGB is independently decoded at the canonical native grid;
            # never upsample student RGB to manufacture a teacher input.
            images = batch["images"].to(device, non_blocking=True)
            if tuple(images.shape) != (batch_size, 16, 3) + TEACHER_SHAPE:
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
                image_shape=TEACHER_SHAPE,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            expected_outputs = {
                "depth": (batch_size, 16) + TEACHER_SHAPE,
                "xyz_local": (batch_size, 16) + TEACHER_SHAPE + (3,),
                "xyz_global": (batch_size, 16) + TEACHER_SHAPE + (3,),
                "conf_local": (batch_size, 16) + TEACHER_SHAPE,
                "valid_mask": (batch_size, 16) + TEACHER_SHAPE,
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
            adapted = canonicalize_teacher_outputs(adapted)

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
                confidence_array = dense_fp32("confidence")
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
                    "teacher_input_height": TEACHER_SHAPE[0],
                    "teacher_input_width": TEACHER_SHAPE[1],
                    "supervision_height": SUPERVISION_SHAPE[0],
                    "supervision_width": SUPERVISION_SHAPE[1],
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
                if save_attention:
                    captured = raw.get("attention")
                    if not isinstance(captured, dict):
                        raise RuntimeError("VGGT-Omega forward did not return attention Q/K")
                    metadata_record["attention"] = {
                        "schema_version": ATTENTION_CACHE_SCHEMA_VERSION,
                        "layers": list(attention_layers),
                        "dtype": (
                            "float16" if attention_dtype in {"float16", "fp16"} else "float32"
                        ),
                        "num_frames": int(captured[attention_layers[0]]["metadata"]["num_frames"]),
                        "patch_grid_h": int(captured[attention_layers[0]]["metadata"]["patch_grid_h"]),
                        "patch_grid_w": int(captured[attention_layers[0]]["metadata"]["patch_grid_w"]),
                        "patch_size": int(captured[attention_layers[0]]["metadata"]["patch_size"]),
                        "image_height": int(captured[attention_layers[0]]["metadata"]["image_height"]),
                        "image_width": int(captured[attention_layers[0]]["metadata"]["image_width"]),
                        "qk_stage": str(captured[attention_layers[0]]["metadata"]["qk_stage"]),
                    }
                arrays = {
                    "dataset_name": np.asarray(metadata["dataset_name"], dtype=np.str_),
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
                    "teacher_input_height": np.asarray(TEACHER_SHAPE[0], dtype=np.int64),
                    "teacher_input_width": np.asarray(TEACHER_SHAPE[1], dtype=np.int64),
                    "supervision_height": np.asarray(SUPERVISION_SHAPE[0], dtype=np.int64),
                    "supervision_width": np.asarray(SUPERVISION_SHAPE[1], dtype=np.int64),
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
                        (
                            CROSSCLIP_ATTENTION_CACHE_FORMAT_VERSION
                            if save_attention
                            else CROSSCLIP_CACHE_FORMAT_VERSION
                        ),
                        dtype=np.str_,
                    ),
                    "metadata_json": np.asarray(
                        json.dumps(metadata_record, ensure_ascii=False),
                        dtype=np.str_,
                    ),
                }
                if save_attention:
                    first_feature = captured[attention_layers[0]]
                    common_metadata = first_feature["metadata"]
                    arrays.update(
                        {
                            "attention_schema_version": np.asarray(
                                ATTENTION_CACHE_SCHEMA_VERSION, dtype=np.str_
                            ),
                            "attention_num_frames": np.asarray(
                                common_metadata["num_frames"], dtype=np.int64
                            ),
                            "attention_patch_grid_h": np.asarray(
                                common_metadata["patch_grid_h"], dtype=np.int64
                            ),
                            "attention_patch_grid_w": np.asarray(
                                common_metadata["patch_grid_w"], dtype=np.int64
                            ),
                            "attention_patch_size": np.asarray(
                                common_metadata["patch_size"], dtype=np.int64
                            ),
                            "attention_image_height": np.asarray(
                                common_metadata["image_height"], dtype=np.int64
                            ),
                            "attention_image_width": np.asarray(
                                common_metadata["image_width"], dtype=np.int64
                            ),
                            "attention_dtype": np.asarray(
                                "float16" if attention_dtype in {"float16", "fp16"} else "float32",
                                dtype=np.str_,
                            ),
                            "attention_qk_stage": np.asarray(
                                common_metadata["qk_stage"], dtype=np.str_
                            ),
                        }
                    )
                    for layer in attention_layers:
                        feature = captured[layer]
                        layer_metadata = feature["metadata"]
                        arrays[attention_cache_key(layer, "q")] = (
                            feature["q"][batch_index].numpy()
                        )
                        arrays[attention_cache_key(layer, "k")] = (
                            feature["k"][batch_index].numpy()
                        )
                        arrays[attention_cache_key(layer, "layer_index")] = np.asarray(
                            layer_metadata["layer_index"], dtype=np.int64
                        )
                        arrays[attention_cache_key(layer, "num_heads")] = np.asarray(
                            layer_metadata["num_heads"], dtype=np.int64
                        )
                        arrays[attention_cache_key(layer, "head_dim")] = np.asarray(
                            layer_metadata["head_dim"], dtype=np.int64
                        )
                    if not printed_attention_sample:
                        print(
                            "Attention cache sample {}: {}".format(
                                path,
                                {
                                    "layer_{}".format(layer): {
                                        "q": list(arrays[attention_cache_key(layer, "q")].shape),
                                        "k": list(arrays[attention_cache_key(layer, "k")].shape),
                                    }
                                    for layer in attention_layers
                                },
                            )
                        )
                        printed_attention_sample = True
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


__all__ = ["attention_cache_bytes_per_clip", "generate_crossclip_teacher_cache"]
