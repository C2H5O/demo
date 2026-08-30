"""PC-Depth-inspired specular highlight detection and local inpainting."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

from datasets.transforms import tensor_from_numpy_buffer


def _cv2():
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required when dataset.highlight.enabled=true. "
            "Install the declared opencv-python dependency."
        ) from error


@dataclass(frozen=True)
class HighlightDetectionConfig:
    enabled: bool = True
    absolute_threshold: float = 250.0
    candidate_threshold: float = 230.0
    relative_threshold: float = 0.95
    minimum_component_size: int = 5000
    keep_components: str = "smaller"
    dilation_radius: int = 2
    median_kernel: int = 31
    inpaint_blur_sigma: float = 8.0
    decay_window_size: int = 10
    decay_coefficient: float = 20.0


class SpecularHighlightProcessor:
    """Return a binary highlight mask and an inpainted RGB tensor.

    The detector follows PC-Depth's absolute/relative brightness tests while
    making empty-component, uint8-input, and division-by-zero cases explicit.
    """

    def __init__(self, config: HighlightDetectionConfig | Dict[str, Any]) -> None:
        self.config = (
            HighlightDetectionConfig(**config) if isinstance(config, dict) else config
        )
        if self.config.keep_components not in {"smaller", "larger", "all"}:
            raise ValueError("keep_components must be smaller, larger, or all")
        if self.config.minimum_component_size < 0:
            raise ValueError("minimum_component_size cannot be negative")

    @staticmethod
    def _to_uint8_rgb(image: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(image):
            value = image.detach().cpu().float()
            if value.ndim != 3 or value.shape[0] != 3:
                raise ValueError("Highlight input must have shape [3,H,W]")
            maximum = float(torch.nan_to_num(value).max()) if value.numel() else 0.0
            if maximum <= 1.0 + 1e-6:
                value = value * 255.0
            value_uint8 = (
                torch.nan_to_num(value)
                .clamp(0.0, 255.0)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
            )
            value = np.frombuffer(
                bytes(value_uint8.untyped_storage()), dtype=np.uint8
            ).reshape(tuple(value_uint8.shape)).copy()
        else:
            value = np.asarray(image)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("Highlight input must have shape [H,W,3]")
        if np.issubdtype(value.dtype, np.floating):
            maximum = float(np.nanmax(value)) if value.size else 0.0
            if maximum <= 1.0 + 1e-6:
                value = value * 255.0
        return np.clip(value, 0.0, 255.0).round().astype(np.uint8)

    @staticmethod
    def _ellipse(mask: np.ndarray, radius: int, operation: str) -> np.ndarray:
        if radius <= 0:
            return mask
        cv2 = _cv2()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        if operation == "dilate":
            return cv2.dilate(mask, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _module1(
        luminance: np.ndarray,
        green: np.ndarray,
        blue: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        eps = 1e-6
        p95_green = float(np.percentile(green, 95))
        p95_blue = float(np.percentile(blue, 95))
        p95_luminance = max(float(np.percentile(luminance, 95)), eps)
        green_ratio = p95_green / p95_luminance
        blue_ratio = p95_blue / p95_luminance
        return (
            (green > green_ratio * threshold)
            | (blue > blue_ratio * threshold)
            | (luminance > threshold)
        )

    def _fill_components(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        cv2 = _cv2()
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            return image.copy()
        filled = image.copy()
        count, labels, _, _ = cv2.connectedComponentsWithStats(binary, 8)
        outer = self._ellipse(binary * 255, 4, "dilate") > 0
        inner = self._ellipse(binary * 255, 2, "dilate") > 0
        ring = outer & ~inner
        global_fallback = np.median(image[~binary.astype(bool)], axis=0) if np.any(~binary.astype(bool)) else np.zeros(3)
        for label in range(1, count):
            component = labels == label
            local_ring = self._ellipse(component.astype(np.uint8) * 255, 4, "dilate") > 0
            local_ring &= ~self._ellipse(component.astype(np.uint8) * 255, 2, "dilate").astype(bool)
            samples = image[local_ring & ring]
            color = np.median(samples, axis=0) if samples.size else global_fallback
            filled[component] = color
        return filled

    def _relative_mask(
        self,
        filled: np.ndarray,
        red: np.ndarray,
        green: np.ndarray,
        blue: np.ndarray,
    ) -> np.ndarray:
        cv2 = _cv2()
        kernel = max(int(self.config.median_kernel), 3)
        if kernel % 2 == 0:
            kernel += 1
        channels = [
            cv2.medianBlur(filled[..., index], kernel).astype(np.float32)
            for index in range(3)
        ]
        ratios = []
        for original, filtered in zip((red, green, blue), channels):
            mean = float(filtered.mean())
            contrast = 1.0 / max((mean + float(filtered.std())) / max(mean, 1e-6), 1e-6)
            ratios.append(contrast * original / np.maximum(filtered, 1.0))
        return np.max(np.stack(ratios, axis=-1), axis=-1) > self.config.relative_threshold

    def _classify(self, mask: np.ndarray) -> np.ndarray:
        cv2 = _cv2()
        count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
        result = np.zeros(mask.shape, dtype=np.float32)
        for label in range(1, count):
            size = int(np.count_nonzero(labels == label))
            keep = (
                self.config.keep_components == "all"
                or (
                    self.config.keep_components == "smaller"
                    and size < self.config.minimum_component_size
                )
                or (
                    self.config.keep_components == "larger"
                    and size >= self.config.minimum_component_size
                )
            )
            if keep:
                result[labels == label] = 1.0
        return result

    def _inpaint(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        cv2 = _cv2()
        if not np.any(mask):
            return image.astype(np.float32)
        filled = self._fill_components(mask, image)
        blurred = cv2.GaussianBlur(
            filled,
            (0, 0),
            float(self.config.inpaint_blur_sigma),
        )
        size = max(int(self.config.decay_window_size), 1)
        kernel = np.ones((size, size), dtype=np.float32) / max(
            float(self.config.decay_coefficient), 1e-6
        )
        blend = cv2.filter2D(mask.astype(np.float32), -1, kernel) + mask
        blend = np.clip(blend, 0.0, 1.0)[..., None]
        output = blend * blurred + (1.0 - blend) * image
        return cv2.medianBlur(np.clip(output, 0, 255).astype(np.uint8), 3).astype(
            np.float32
        )

    def process_numpy(
        self, image: torch.Tensor | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return binary float32 mask and float32 [0,1] inpainted RGB."""
        rgb = self._to_uint8_rgb(image)
        red, green, blue = (rgb[..., index].astype(np.float32) for index in range(3))
        luminance = 0.2989 * red + 0.5870 * green + 0.1140 * blue
        absolute = self._module1(
            luminance, green, blue, self.config.absolute_threshold
        )
        candidate = self._module1(
            luminance, green, blue, self.config.candidate_threshold
        )
        filled = self._fill_components(candidate, rgb)
        relative = self._relative_mask(filled, red, green, blue)
        combined = (absolute | relative) & candidate
        dilated = self._ellipse(
            combined.astype(np.uint8) * 255,
            self.config.dilation_radius,
            "dilate",
        )
        mask = self._classify(dilated)
        inpainted = self._inpaint(mask, rgb) / 255.0
        return mask, inpainted

    def __call__(self, image: torch.Tensor | np.ndarray) -> Dict[str, torch.Tensor]:
        mask, inpainted = self.process_numpy(image)
        return {
            "highlight_mask": tensor_from_numpy_buffer(mask).unsqueeze(0).float(),
            "inpainted_image": tensor_from_numpy_buffer(inpainted)
            .permute(2, 0, 1)
            .contiguous()
            .float(),
        }
