"""Read-only adapter around the official Zanue/EndoDAV checkout."""

from __future__ import annotations

import importlib
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

from endodaveval.config import INTERNAL_MODEL_RESOLUTION_HW


OFFICIAL_REPOSITORY = "https://github.com/Zanue/EndoDAV"
AUDITED_OFFICIAL_COMMIT = "8a9681b43d9b3b1600ba5f389e1ef69120af37cf"
OFFICIAL_DEPTH_RANGE = (0.1, 150.0)
OFFICIAL_CHECKPOINT_METADATA_KEYS = frozenset({"height", "width", "use_stereo"})
TEMPORAL_LORA_KEY_PREFIX = "head.motion_modules."


class AdapterError(RuntimeError):
    pass


def official_constructor_kwargs(
    pretrained_path: Path, *, temporal_lora: bool = False
) -> Dict[str, Any]:
    """Exact paper-evaluation model settings for the selected checkpoint."""
    return {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "r": 4,
        "lora_type": "ssb",
        "image_shape": INTERNAL_MODEL_RESOLUTION_HW,
        "pretrained_path": str(pretrained_path),
        "residual_block_indexes": [],
        "include_cls_token": True,
        "inv_sigmoid": False,
        "temporal_lora": bool(temporal_lora),
        "disable_conv_head": True,
        "out_sigmoid": False,
    }


def validate_official_repository(repository: Path) -> Dict[str, str]:
    repository = repository.resolve()
    required = {
        "model": repository / "models/endodav/endodav.py",
        "model_export": repository / "models/endodav/__init__.py",
        "depth_conversion": repository / "utils/layers.py",
        "official_evaluator": repository / "evaluate_depth_video_pose.py",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise AdapterError("Official EndoDAV checkout is incomplete: {}".format(missing))
    return {name: str(path) for name, path in required.items()}


@contextmanager
def _repository_import_path(repository: Path):
    value = str(repository.resolve())
    sys.path.insert(0, value)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        if sys.path and sys.path[0] == value:
            sys.path.pop(0)
        elif value in sys.path:
            sys.path.remove(value)


def _state_dict(value: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(value, Mapping) and isinstance(value.get("state_dict"), Mapping):
        value = value["state_dict"]
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AdapterError("EndoDAV depth checkpoint is not a state dict")
    return value


def _checkpoint_uses_temporal_lora(state: Mapping[str, Any]) -> bool:
    """Detect the optional temporal-LoRA modules encoded by official weights."""
    return any(
        key.startswith(TEMPORAL_LORA_KEY_PREFIX)
        and (key.endswith(".lora_A") or key.endswith(".lora_B"))
        for key in state
    )


def _weights_for_model(
    state: Mapping[str, Any], model_keys: Sequence[str]
) -> Tuple[Dict[str, torch.Tensor], Sequence[str]]:
    """Match official weights while ignoring only documented scalar metadata."""
    expected = set(model_keys)
    weights: Dict[str, torch.Tensor] = {}
    unexpected = []
    for key, value in state.items():
        if key in OFFICIAL_CHECKPOINT_METADATA_KEYS:
            continue
        if key not in expected or not isinstance(value, torch.Tensor):
            unexpected.append(key)
            continue
        weights[key] = value
    return weights, unexpected


def load_official_model(
    repository: Path,
    checkpoint: Path,
    pretrained_path: Path,
    device: str,
):
    """Instantiate and strict-check weights without modifying official source."""
    validate_official_repository(repository)
    checkpoint = checkpoint.resolve()
    pretrained_path = pretrained_path.resolve()
    base_checkpoint = pretrained_path / "video_depth_anything_vits.pth"
    if not checkpoint.is_file():
        raise AdapterError("EndoDAV depth_model.pth is missing: {}".format(checkpoint))
    if not base_checkpoint.is_file():
        raise AdapterError("VDA-S checkpoint is missing: {}".format(base_checkpoint))
    state = _state_dict(torch.load(str(checkpoint), map_location="cpu"))
    temporal_lora = _checkpoint_uses_temporal_lora(state)
    with _repository_import_path(repository):
        module = importlib.import_module("models.endodav")
    module_path = Path(str(module.__file__)).resolve()
    try:
        module_path.relative_to(repository.resolve())
    except ValueError as error:
        raise AdapterError(
            "Python imported models.endodav from another checkout: {}".format(module_path)
        ) from error
    constructor = getattr(module, "endodav", None)
    if constructor is None:
        raise AdapterError("Official models.endodav.endodav is unavailable")
    model = constructor(
        **official_constructor_kwargs(
            pretrained_path, temporal_lora=temporal_lora
        )
    )
    weights, unexpected = _weights_for_model(state, model.state_dict().keys())
    incompatible = model.load_state_dict(weights, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(unexpected) + list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise AdapterError(
            "EndoDAV checkpoint mismatch; missing keys: {}; unexpected keys: {}".format(
                missing[:20], unexpected[:20]
            )
        )
    model = model.to(torch.device(device)).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def disp_to_depth(
    output_disp: np.ndarray, min_depth: float = 0.1, max_depth: float = 150.0
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy equivalent of official ``utils.layers.disp_to_depth``."""
    disparity = np.asarray(output_disp, dtype=np.float32)
    if not np.isfinite(disparity).all():
        raise AdapterError("Official EndoDAV output contains NaN or Inf")
    min_disp = 1.0 / float(max_depth)
    max_disp = 1.0 / float(min_depth)
    scaled_disp = min_disp + (max_disp - min_disp) * disparity
    predicted_depth = 1.0 / scaled_disp
    return scaled_disp.astype(np.float32), predicted_depth.astype(np.float32)


def official_output_to_depth(output_disp: np.ndarray) -> np.ndarray:
    """Apply the official 0.1..150 m disparity-to-depth conversion."""
    _, depth = disp_to_depth(output_disp, *OFFICIAL_DEPTH_RANGE)
    return depth


def load_raw_rgb(paths: Sequence[Path]) -> np.ndarray:
    frames = []
    shape = None
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise AdapterError("Could not read RGB frame: {}".format(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if shape is None:
            shape = rgb.shape
        elif rgb.shape != shape:
            raise AdapterError("EndoDAV full-video inference requires one RGB shape")
        frames.append(rgb)
    if not frames:
        raise AdapterError("Cannot infer an empty sequence")
    return np.stack(frames)


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


class ModelForwardTimer:
    """Temporarily wrap ``model.forward``; official source remains untouched."""

    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self.seconds = 0.0
        self.call_count = 0
        self._original = None

    def __enter__(self):
        self._original = self.model.forward

        def timed_forward(*args, **kwargs):
            _synchronize(self.device)
            started = time.perf_counter()
            try:
                return self._original(*args, **kwargs)
            finally:
                _synchronize(self.device)
                self.seconds += time.perf_counter() - started
                self.call_count += 1

        self.model.forward = timed_forward
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.forward = self._original


def infer_official_full_video(model, frames: np.ndarray, device: str):
    """Call official full-video inference once; no GT or external windowing."""
    if getattr(model, "training", False):
        raise AdapterError("EndoDAV inference requires model.eval()")
    started = time.perf_counter()
    with ModelForwardTimer(model, device) as timer:
        output_disp = model.infer_video_depth(frames)
    pipeline_seconds = time.perf_counter() - started
    output_disp = np.asarray(output_disp, dtype=np.float32)
    if output_disp.shape != frames.shape[:3]:
        raise AdapterError(
            "Official output must be upsampled to input-frame shape {}; found {}".format(
                frames.shape[:3], output_disp.shape
            )
        )
    depth = official_output_to_depth(output_disp)
    return depth, {
        "model_forward_seconds": float(timer.seconds),
        "model_forward_call_count": int(timer.call_count),
        "sequence_pipeline_seconds": float(pipeline_seconds),
        "model_forward_timing_scope": (
            "CUDA-synchronized model.forward calls inside official infer_video_depth; "
            "excludes RGB disk decode, official preprocessing/stitching outside forward, "
            "evaluation resize, GT, metrics, TAE and JSON export"
        ),
        "sequence_pipeline_timing_scope": (
            "one complete official model.infer_video_depth(frames) call; excludes RGB "
            "disk decode, evaluation resize, GT, metrics, TAE and JSON export"
        ),
    }
