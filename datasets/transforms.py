"""Deterministic RGB preprocessing utilities for temporal SCARED clips."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


_NUMPY_TO_TORCH_DTYPE = {
    np.dtype(np.bool_): torch.bool,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
}


def tensor_from_numpy_buffer(array: np.ndarray) -> torch.Tensor:
    """Copy a NumPy array through the buffer protocol, bypassing its C-API type bridge."""
    contiguous = np.ascontiguousarray(array)
    dtype = _NUMPY_TO_TORCH_DTYPE.get(contiguous.dtype)
    if dtype is None:
        raise TypeError("Unsupported NumPy dtype for Torch conversion: {}".format(contiguous.dtype))
    # bytearray owns a writable copy; clone detaches the returned tensor from it.
    return torch.frombuffer(
        bytearray(contiguous.tobytes(order="C")), dtype=dtype
    ).reshape(contiguous.shape).clone()


def _rgb_image_to_tensor(image: Image.Image) -> torch.Tensor:
    width, height = image.size
    return (
        torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        .reshape(height, width, 3)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .div_(255.0)
    )


def _resample_bicubic() -> int:
    return getattr(Image, "Resampling", Image).BICUBIC


def _resize_image(image: Image.Image, height: int, width: int, mode: str) -> Image.Image:
    if mode == "resize":
        return image.resize((width, height), _resample_bicubic())
    source_width, source_height = image.size
    if mode == "letterbox":
        scale = min(width / source_width, height / source_height)
        resized = image.resize((max(1, round(source_width * scale)), max(1, round(source_height * scale))), _resample_bicubic())
        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
        return canvas
    if mode == "center_crop":
        scale = max(width / source_width, height / source_height)
        resized = image.resize((max(1, round(source_width * scale)), max(1, round(source_height * scale))), _resample_bicubic())
        left, top = (resized.width - width) // 2, (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))
    raise ValueError("resize_mode must be resize, letterbox, or center_crop; received {!r}".format(mode))


def normalize_image(image: torch.Tensor, mode: str) -> torch.Tensor:
    """Normalize RGB tensor [3,H,W] using one supported deterministic mode."""
    if mode in ("none", "zero_one"):
        return image
    if mode == "minus_one_one":
        return image.mul(2.0).sub(1.0)
    if mode == "imagenet":
        mean, std = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1), image.new_tensor(IMAGENET_STD).view(3, 1, 1)
        return (image - mean) / std
    raise ValueError("normalize_mode must be imagenet, zero_one, minus_one_one, or none; received {!r}".format(mode))


def unnormalize_image(image: torch.Tensor, mode: str = "imagenet") -> torch.Tensor:
    """Convert normalized [3,H,W] RGB back to a clipped [0,1] preview tensor."""
    if mode in ("none", "zero_one"):
        result = image
    elif mode == "minus_one_one":
        result = image.add(1.0).div(2.0)
    elif mode == "imagenet":
        mean, std = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1), image.new_tensor(IMAGENET_STD).view(3, 1, 1)
        result = image * std + mean
    else:
        raise ValueError("Unsupported normalize mode {!r}".format(mode))
    return result.clamp(0.0, 1.0)


def load_rgb_tensor(path: Union[str, Path], image_height: int, image_width: int, resize_mode: str = "resize", normalize_mode: str = "imagenet") -> torch.Tensor:
    """Load one RGB image with deterministic spatial preprocessing."""
    try:
        with Image.open(path) as image:
            processed = _resize_image(image.convert("RGB"), image_height, image_width, resize_mode)
    except (OSError, ValueError) as error:
        raise RuntimeError("Failed to decode SCARED RGB image {}: {}".format(path, error)) from error
    return normalize_image(_rgb_image_to_tensor(processed), normalize_mode)


def _decode_rgb(path: Union[str, Path], expected_size: tuple[int, int], label: str) -> torch.Tensor:
    """Decode exact-size RGB without permitting an implicit FOV-changing resize."""
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if rgb.size != expected_size:
                raise RuntimeError(
                    "{} image {} has size {} but requires {}".format(
                        label, path, rgb.size, expected_size
                    )
                )
            tensor = _rgb_image_to_tensor(rgb)
    except (OSError, ValueError) as error:
        raise RuntimeError("Failed to decode {} RGB image {}: {}".format(label, path, error)) from error
    return tensor


def load_precomputed_student_rgb_tensor(path: Union[str, Path], normalize_mode: str = "minus_one_one") -> torch.Tensor:
    """Canonical student input: decode 560x448 PNG, with no spatial resize."""
    return normalize_image(_decode_rgb(path, (560, 448), "precomputed student"), normalize_mode)


def load_teacher_rgb_tensor(path: Union[str, Path]) -> torch.Tensor:
    """Canonical teacher input: decode the required 1280x1024 RGB directly."""
    return _decode_rgb(path, (1280, 1024), "teacher")
