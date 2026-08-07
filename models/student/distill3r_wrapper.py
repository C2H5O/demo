"""Project adapter around the official Distill3R student implementation."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISTILL3R_ROOT = PROJECT_ROOT / "external" / "Distill3R"
DISTILL3R_DUNE_ROOT = DISTILL3R_ROOT / "external" / "dune"
DISTILL3R_FAST3R_ROOT = DISTILL3R_ROOT / "external" / "fast3r"


@dataclass(frozen=True)
class Distill3RStudentConfig:
    """Configuration accepted by the official ``CompressedFast3R`` student."""

    image_height: int = 448
    image_width: int = 560
    img_size: int = 560
    patch_size: int = 14
    embed_dim: int = 384
    encoder_depth: int = 12
    encoder_heads: int = 6
    decoder_depth: int = 6
    decoder_heads: int = 6
    decoder_attention_implementation: str = "flash_attention"
    encoder_type: str = "dune"
    max_views: int = 16
    max_parallel_views_for_head: int = 8
    landscape_only: bool = True
    with_local_head: bool = True
    load_pretrained: bool = True
    freeze_encoder: bool = False
    pretrained_checkpoint: str = "./checkpoints/dune/dune_vitsmall14_448.pth"
    use_local_dune_submodule: bool = True

    def validate(self) -> None:
        if (self.image_height, self.image_width) != (448, 560):
            raise ValueError(
                "This project requires student image_height/image_width=448/560"
            )
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError(
                "Distill3R input resolution must be divisible by patch_size"
            )
        if self.encoder_type != "dune":
            raise ValueError("The configured Distill3R student must use encoder_type=dune")
        if (self.patch_size, self.embed_dim, self.encoder_depth, self.encoder_heads) != (
            14,
            384,
            12,
            6,
        ):
            raise ValueError(
                "Distill3R's DUNE student requires patch_size=14, embed_dim=384, "
                "encoder_depth=12, and encoder_heads=6"
            )
        if not self.with_local_head:
            raise ValueError("Local point maps are required by the existing loss/evaluation pipeline")
        if not self.load_pretrained:
            raise ValueError(
                "The upstream Distill3R DUNE constructor requires load_pretrained=true"
            )
        if not str(self.pretrained_checkpoint).strip():
            raise ValueError("student.pretrained_checkpoint must be configured")
        if not self.use_local_dune_submodule:
            raise ValueError(
                "student.use_local_dune_submodule must remain true so the configured "
                "local DUNE checkpoint is used"
            )
        if self.max_views <= 0 or self.max_parallel_views_for_head <= 0:
            raise ValueError("max_views and max_parallel_views_for_head must be positive")
        if self.decoder_attention_implementation not in {
            "flash_attention",
            "pytorch_auto",
            "pytorch_naive",
        }:
            raise ValueError(
                "decoder_attention_implementation must be flash_attention, "
                "pytorch_auto, or pytorch_naive"
            )


def _ensure_official_sources_importable() -> None:
    missing = [
        path
        for path in (DISTILL3R_ROOT, DISTILL3R_DUNE_ROOT, DISTILL3R_FAST3R_ROOT)
        if not path.is_dir()
    ]
    if missing:
        raise RuntimeError(
            "Distill3R sources are incomplete: {}. Run `git submodule update "
            "--init --recursive` from the project root.".format(
                ", ".join(str(path) for path in missing)
            )
        )
    for path in (DISTILL3R_DUNE_ROOT, DISTILL3R_FAST3R_ROOT, DISTILL3R_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


@contextmanager
def _pinned_dune_hub(enabled: bool, checkpoint_path: Path) -> Iterator[None]:
    """Redirect Distill3R's hub call to the configured local DUNE checkpoint."""

    if not enabled:
        yield
        return
    original_load = torch.hub.load

    def local_load(repo_or_dir: Any, model: str, *args: Any, **kwargs: Any) -> Any:
        if str(repo_or_dir).lower() == "naver/dune":
            if model != "dune_vitsmall_14_448_encoder":
                raise ValueError("Unexpected Distill3R DUNE hub model: {}".format(model))
            from model.dune import load_dune_encoder_from_checkpoint

            encoder, _ = load_dune_encoder_from_checkpoint(str(checkpoint_path))
            return encoder
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = local_load
    try:
        yield
    finally:
        torch.hub.load = original_load


def _official_model_factory(**kwargs: Any) -> nn.Module:
    _ensure_official_sources_importable()
    try:
        from distill3r.student.model import CompressedFast3R
    except ImportError as error:
        raise RuntimeError(
            "Failed to import the official Distill3R student. Create/activate the "
            "documented Conda environment and initialize all Git submodules."
        ) from error
    return CompressedFast3R(**kwargs)


class Distill3RStudent(nn.Module):
    """Expose official Distill3R predictions in the existing cache-loss format."""

    def __init__(
        self,
        config: Mapping[str, Any] | Distill3RStudentConfig,
        model_factory: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.config = (
            Distill3RStudentConfig(**dict(config))
            if isinstance(config, Mapping)
            else config
        )
        self.config.validate()
        factory = model_factory or _official_model_factory
        model_kwargs = asdict(self.config)
        model_kwargs.pop("image_height")
        model_kwargs.pop("image_width")
        freeze_encoder = bool(model_kwargs.pop("freeze_encoder"))
        use_local_dune = bool(model_kwargs.pop("use_local_dune_submodule"))
        checkpoint_value = str(model_kwargs.pop("pretrained_checkpoint"))
        checkpoint_path = Path(checkpoint_value).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        if model_factory is None and not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Configured DUNE checkpoint does not exist: {}".format(
                    checkpoint_path
                )
            )
        attention_implementation = str(
            model_kwargs.pop("decoder_attention_implementation")
        )
        with _pinned_dune_hub(
            use_local_dune and model_factory is None,
            checkpoint_path,
        ):
            self.student = factory(**model_kwargs)
        self._set_decoder_attention_implementation(attention_implementation)
        if freeze_encoder:
            encoder = getattr(self.student, "encoder", None)
            if encoder is None:
                raise RuntimeError("The official Distill3R student has no encoder to freeze")
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

    def _set_decoder_attention_implementation(self, implementation: str) -> None:
        """Select a pinned Fast3R attention backend without editing the submodule."""

        decoder = getattr(self.student, "decoder", None)
        blocks = getattr(decoder, "dec_blocks", ())
        for block in blocks:
            attention = getattr(block, "attn", None)
            if attention is None or not hasattr(attention, "attn_implementation"):
                raise RuntimeError(
                    "The pinned Fast3R decoder attention interface has changed"
                )
            attention.attn_implementation = implementation

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _views(self, images: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        batch, frames, _, height, width = images.shape
        true_shape = torch.tensor(
            [height, width], dtype=torch.long
        ).unsqueeze(0).expand(batch, -1)
        return [
            {"img": images[:, frame], "true_shape": true_shape}
            for frame in range(frames)
        ]

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(
                "images must have shape [B,T,3,H,W], got {}".format(
                    tuple(images.shape)
                )
            )
        batch, frames, channels, height, width = images.shape
        if channels != 3:
            raise ValueError("Distill3R expects three-channel RGB images")
        if (height, width) != (
            self.config.image_height,
            self.config.image_width,
        ):
            raise ValueError(
                "Distill3R expects 448x560 input, got {}x{}".format(height, width)
            )
        if frames > self.config.max_views:
            raise ValueError(
                "Input has {} frames, but max_views={}".format(
                    frames, self.config.max_views
                )
            )
        raw_outputs = self.student(self._views(images))
        if not isinstance(raw_outputs, (list, tuple)) or len(raw_outputs) != frames:
            raise RuntimeError(
                "Official Distill3R must return one output dictionary per input view"
            )
        required = ("pts3d_in_other_view", "pts3d_local", "conf", "conf_local")
        for index, output in enumerate(raw_outputs):
            missing = [name for name in required if name not in output]
            if missing:
                raise KeyError(
                    "Distill3R view {} output is missing {}".format(index, missing)
                )
        adapted = {
            "xyz_global": torch.stack(
                [output["pts3d_in_other_view"] for output in raw_outputs], dim=1
            ),
            "xyz_local": torch.stack(
                [output["pts3d_local"] for output in raw_outputs], dim=1
            ),
            "conf_global": torch.stack(
                [output["conf"] for output in raw_outputs], dim=1
            ),
            "conf_local": torch.stack(
                [output["conf_local"] for output in raw_outputs], dim=1
            ),
        }
        expected = {
            "xyz_global": (batch, frames, height, width, 3),
            "xyz_local": (batch, frames, height, width, 3),
            "conf_global": (batch, frames, height, width),
            "conf_local": (batch, frames, height, width),
        }
        wrong = {
            name: (tuple(adapted[name].shape), shape)
            for name, shape in expected.items()
            if tuple(adapted[name].shape) != shape
        }
        if wrong:
            raise RuntimeError("Distill3R output resolution contract failed: {}".format(wrong))
        return adapted


def build_distill3r_student(config: Mapping[str, Any]) -> Distill3RStudent:
    return Distill3RStudent(config)


__all__ = [
    "Distill3RStudent",
    "Distill3RStudentConfig",
    "build_distill3r_student",
]
