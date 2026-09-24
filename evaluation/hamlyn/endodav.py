"""Read-only adapter for official EndoDAV full-video inference."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
    package_roots = {
        "models": repository.resolve() / "models",
        "utils": repository.resolve() / "utils",
    }
    displaced_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if any(name == root or name.startswith(f"{root}.") for root in package_roots)
    }
    for name in displaced_modules:
        del sys.modules[name]

    for name, package_path in package_roots.items():
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(package_path)]
        sys.modules[name] = importlib.util.module_from_spec(spec)

    sys.path.insert(0, value)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        for name in tuple(sys.modules):
            if any(
                name == root or name.startswith(f"{root}.")
                for root in package_roots
            ):
                del sys.modules[name]
        sys.modules.update(displaced_modules)
        if value in sys.path:
            sys.path.remove(value)
        importlib.invalidate_caches()


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


def _load_resized_rgb_worker(task) -> np.ndarray:
    """Pickleable CPU-only EndoDAV preprocessing worker."""
    path, height, width = task
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize(
                (width, height), _resample_bicubic()
            )
            return np.asarray(rgb, dtype=np.uint8).copy()
    except (OSError, ValueError) as error:
        raise AdapterError("Failed to decode Hamlyn RGB {}: {}".format(path, error))


class EndoDAVRGBLoader:
    def __init__(self, num_workers: int = 4) -> None:
        self.num_workers = int(num_workers)
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.multiprocessing_context = "spawn" if self.num_workers > 1 else None
        self.loader_backend = (
            "ProcessPoolExecutor(spawn)" if self.num_workers > 1 else "serial"
        )
        self._executor = (
            ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            if self.num_workers > 1
            else None
        )
        self.closed = False

    def load(self, paths: Sequence[Path]) -> np.ndarray:
        if self.closed:
            raise RuntimeError("EndoDAVRGBLoader is closed")
        if not paths:
            raise AdapterError("Cannot run EndoDAV on an empty sequence")
        height, width = INFERENCE_RESOLUTIONS_HW["endodav"]
        tasks = [(str(path), height, width) for path in paths]
        if self._executor is None:
            frames = [_load_resized_rgb_worker(task) for task in tasks]
        else:
            frames = list(self._executor.map(_load_resized_rgb_worker, tasks))
        return np.stack(frames)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "EndoDAVRGBLoader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def load_resized_rgb(paths: Sequence[Path], num_workers: int = 4) -> np.ndarray:
    """Decode/resize in input order with a spawn pool when workers exceed one."""
    with EndoDAVRGBLoader(num_workers) as loader:
        return loader.load(paths)


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
            print(
                "[endodav] reuse sequence {:02d} cache".format(record.sequence_id),
                flush=True,
            )
    if not pending:
        return
    if not torch.cuda.is_available() or not runtime.device.startswith("cuda"):
        raise RuntimeError("EndoDAV inference requires an available CUDA device")
    model, temporal_lora = load_official_model(runtime)
    rgb_loader = EndoDAVRGBLoader(runtime.resize_workers)
    try:
        for position, record in enumerate(pending, start=1):
            print(
                "[endodav] {}/{} sequence {:02d}".format(
                    position, len(pending), record.sequence_id
                ),
                flush=True,
            )
            directory = prepare_cache(runtime.output_root, method, record, force=force)
            pipeline_started = time.perf_counter()
            rgb_started = time.perf_counter()
            frames = rgb_loader.load(
                [record.rgb_by_id[identifier] for identifier in record.frame_ids]
            )
            rgb_seconds = time.perf_counter() - rgb_started
            forward_started = time.perf_counter()
            with torch.inference_mode():
                output = model.infer_video_depth(frames)
            forward_seconds = time.perf_counter() - forward_started
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
            pipeline_seconds = time.perf_counter() - pipeline_started
            finalize_cache(
                directory,
                method,
                record,
                shape,
                {
                    "official_inference": "one model.infer_video_depth(frames) call",
                    "external_windowing": False,
                    "model_forward_seconds": forward_seconds,
                    "sequence_pipeline_seconds": pipeline_seconds,
                    "rgb_decode_resize_seconds": rgb_seconds,
                    "rgb_loader_workers": runtime.resize_workers,
                    "rgb_loader_backend": rgb_loader.loader_backend,
                    "prefetch_windows": 0,
                    "frame_cache_size": 0,
                    "temporal_lora": temporal_lora,
                    "normalized_disparity_conversion": "official disp_to_depth range 0.1..150.0 m",
                    "ground_truth_used_for_inference": False,
                },
            )
    finally:
        rgb_loader.close()
