"""VGGT-Omega boundary for the shared formal VDA video pipeline."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from inference.student_video import infer_vda_video
from models.teacher.output_adapter import adapt_teacher_depth_outputs


class VGGTOmegaSequenceFrames:
    """Lazy raw-RGB paths decoded by VGGT-Omega's released preprocessing."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        image_resolution: int = 512,
        mode: str = "balanced",
        patch_size: int = 16,
    ) -> None:
        self.paths = [str(path) for path in paths]
        self.image_resolution = int(image_resolution)
        self.mode = str(mode)
        self.patch_size = int(patch_size)
        if not self.paths:
            raise ValueError("Cannot preprocess an empty VGGT-Omega sequence")
        module = importlib.import_module("vggt_omega.utils.load_fn")
        self._load_and_preprocess_images = module.load_and_preprocess_images
        self.last_shape = None

    def __len__(self) -> int:
        return len(self.paths)

    def load_indices(self, indices: Sequence[int]) -> torch.Tensor:
        images = self._load_and_preprocess_images(
            [self.paths[index] for index in indices],
            mode=self.mode,
            image_resolution=self.image_resolution,
            patch_size=self.patch_size,
        )
        shape = tuple(int(value) for value in images.shape[-2:])
        if self.last_shape is not None and shape != self.last_shape:
            raise RuntimeError(
                "VGGT-Omega preprocessing changed shape within one sequence: "
                "{} -> {}".format(self.last_shape, shape)
            )
        self.last_shape = shape
        return images

    def metadata(self) -> Dict[str, Any]:
        return {
            "implementation": "vggt_omega.utils.load_fn.load_and_preprocess_images",
            "mode": self.mode,
            "image_resolution": self.image_resolution,
            "patch_size": self.patch_size,
            "observed_input_shape_hw": list(self.last_shape) if self.last_shape else None,
            "normalization": "zero_one_then_vggt_omega_internal_resnet_normalization",
        }


def _forward_vggt_omega(model, images: torch.Tensor) -> Dict[str, torch.Tensor]:
    raw = model(images)
    with torch.autocast(device_type=images.device.type, enabled=False):
        return adapt_teacher_depth_outputs(raw, tuple(images.shape[-2:]))


def infer_vggt_omega_video(
    model,
    frames: VGGTOmegaSequenceFrames,
    emit,
    *,
    device,
    amp=True,
    max_windows=None,
    inspect_window=None,
) -> dict:
    """Run dense VGGT-Omega with the student's exact VDA window/stitching code."""
    return infer_vda_video(
        model,
        frames,
        emit,
        device=device,
        amp=amp,
        max_windows=max_windows,
        forward_model=_forward_vggt_omega,
        inspect_window=inspect_window,
        prediction_label="VGGT-Omega",
    )


__all__ = ["VGGTOmegaSequenceFrames", "infer_vggt_omega_video"]
