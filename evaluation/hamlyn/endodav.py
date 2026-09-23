"""Read-only adapter for official EndoDAV full-video inference."""

from __future__ import annotations

import importlib
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from evaluation.hamlyn.cache import (
    cache_is_complete,
    finalize_cache,
    prepare_cache,
    save_prediction,
)
from evaluation.hamlyn.config import RuntimeConfig
from evaluation.hamlyn.constants import INFERENCE_RESOLUTIONS_HW
from evaluation.hamlyn.data import SequenceRecord


OFFICIAL_REPOSITORY = "https://github.com/Zanue/EndoDAV"
OFFICIAL_CHECKPOINT_METADATA_KEYS = frozenset({"height", "width", "use_stereo"})
TEMPORAL_LORA_KEY_PREFIX = "head.motion_modules."
OFFICIAL_DEPTH_RANGE = (0.1, 150.0)


class AdapterError(RuntimeError):
    pass


def official_constructor_kwargs(
    pretrained_path: Path, temporal_lora: bool
) -> Mapping[str, Any]:
    return {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "r": 4,
        "lora_type": "ssb",
        "image_shape": INFERENCE_RESOLUTIONS_HW["endodav"],
        "pretrained_path": str(pretrained_path),
        "residual_block_indexes": [],
        "include_cls_token": True,
        "inv_sigmoid": False,
        "temporal_lora": bool(temporal_lora),
        "disable_conv_head": True,
        "out_sigmoid": False,
    }


@contextmanager
def _repository_import_path(repository: Path):
    value = str(repository.resolve())
    sys.path.insert(0, value)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        if value in sys.path:
            sys.path.remove(value)


def _state_dict(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping) and isinstance(value.get("state_dict"), Mapping):
        value = value["state_dict"]
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AdapterError("EndoDAV depth checkpoint is not a state dict")
    return value


def _uses_temporal_lora(state: Mapping[str, Any]) -> bool:
    return any(
        key.startswith(TEMPORAL_LORA_KEY_PREFIX)
        and (key.endswith(".lora_A") or key.endswith(".lora_B"))
        for key in state
    )


def load_official_model(runtime: RuntimeConfig):
    repository = runtime.endodav_repository
    required = (
        repository / "models/endodav/endodav.py",
        repository / "models/endodav/__init__.py",
        repository / "utils/layers.py",
        repository / "evaluate_depth_video_hamlyn.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AdapterError("Official EndoDAV checkout is incomplete: {}".format(missing))
    try:
        checkpoint_value = torch.load(
            str(runtime.endodav_checkpoint), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint_value = torch.load(str(runtime.endodav_checkpoint), map_location="cpu")
    state = _state_dict(checkpoint_value)
    temporal_lora = _uses_temporal_lora(state)
    with _repository_import_path(repository):
        module = importlib.import_module("models.endodav")
    module_path = Path(str(module.__file__)).resolve()
    try:
        module_path.relative_to(repository.resolve())
    except ValueError as error:
        raise AdapterError(
            "Imported models.endodav from another checkout: {}".format(module_path)
        ) from error
    constructor = getattr(module, "endodav", None)
    if constructor is None:
        raise AdapterError("Official models.endodav.endodav is unavailable")
    model = constructor(
        **official_constructor_kwargs(
            runtime.endodav_pretrained_path, temporal_lora=temporal_lora
        )
    )
    expected = set(model.state_dict())
    weights = {}
    unexpected = []
    for key, value in state.items():
        if key in OFFICIAL_CHECKPOINT_METADATA_KEYS:
            continue
        if key not in expected or not isinstance(value, torch.Tensor):
            unexpected.append(key)
        else:
            weights[key] = value
    incompatible = model.load_state_dict(weights, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected.extend(incompatible.unexpected_keys)
    if missing or unexpected:
        raise AdapterError(
            "EndoDAV checkpoint mismatch; missing keys {}; unexpected keys {}".format(
                missing[:20], unexpected[:20]
            )
        )
    model = model.to(torch.device(runtime.device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, temporal_lora


def _resample_bicubic():
    return getattr(Image, "Resampling", Image).BICUBIC


def load_resized_rgb(paths: Sequence[Path]) -> np.ndarray:
    """Use the same direct PIL bicubic resize as baseline-J SequenceFrames."""
    height, width = INFERENCE_RESOLUTIONS_HW["endodav"]
    frames = []
    for path in paths:
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB").resize(
                    (width, height), _resample_bicubic()
                )
                frames.append(np.asarray(rgb, dtype=np.uint8))
        except (OSError, ValueError) as error:
            raise AdapterError("Failed to decode Hamlyn RGB {}: {}".format(path, error))
    if not frames:
        raise AdapterError("Cannot run EndoDAV on an empty sequence")
    return np.stack(frames)


def official_output_to_depth(output_disp: np.ndarray) -> np.ndarray:
    disparity = np.asarray(output_disp, dtype=np.float32)
    if not np.isfinite(disparity).all():
        raise AdapterError("Official EndoDAV output contains NaN or Inf")
    min_depth, max_depth = OFFICIAL_DEPTH_RANGE
    scaled_disparity = 1.0 / max_depth + (1.0 / min_depth - 1.0 / max_depth) * disparity
    depth = np.reciprocal(scaled_disparity)
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise AdapterError("Converted EndoDAV depth is invalid")
    return depth.astype(np.float32, copy=False)


def infer_endodav_sequences(
    records: Sequence[SequenceRecord],
    runtime: RuntimeConfig,
    force: bool = False,
) -> None:
    method = "endodav"
    shape = INFERENCE_RESOLUTIONS_HW[method]
    pending = [
        record
        for record in records
        if force or not cache_is_complete(runtime.output_root, method, record, shape)
    ]
    for record in records:
        if record not in pending:
            print("[endodav] reuse sequence {:02d} cache".format(record.sequence_id), flush=True)
    if not pending:
        return
    if not torch.cuda.is_available() or not runtime.device.startswith("cuda"):
        raise RuntimeError("EndoDAV inference requires an available CUDA device")
    model, temporal_lora = load_official_model(runtime)
    for position, record in enumerate(pending, start=1):
        print(
            "[endodav] {}/{} sequence {:02d}".format(
                position, len(pending), record.sequence_id
            ),
            flush=True,
        )
        directory = prepare_cache(runtime.output_root, method, record, force=force)
        frames = load_resized_rgb(
            [record.rgb_by_id[identifier] for identifier in record.frame_ids]
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.infer_video_depth(frames)
        elapsed = time.perf_counter() - started
        output = np.asarray(output, dtype=np.float32)
        expected = (record.frame_count, *shape)
        if output.shape != expected:
            raise AdapterError(
                "Official EndoDAV output must have shape {}; found {}".format(
                    expected, output.shape
                )
            )
        depth = official_output_to_depth(output)
        for identifier, value in zip(record.frame_ids, depth):
            save_prediction(directory, identifier, value)
        finalize_cache(
            directory,
            method,
            record,
            shape,
            {
                "official_inference": "one model.infer_video_depth(frames) call",
                "external_windowing": False,
                "sequence_pipeline_seconds": elapsed,
                "temporal_lora": temporal_lora,
                "normalized_disparity_conversion": "official disp_to_depth range 0.1..150.0 m",
                "ground_truth_used_for_inference": False,
            },
        )
