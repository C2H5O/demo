"""Frozen DUNE ViT-S/14 intermediate features with an encoder-only Fast3R DPT head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.student.upstream_sources import (
    PROJECT_ROOT,
    ensure_fast3r_source_importable,
    load_pinned_dune_encoder,
)


EncoderFactory = Callable[[], nn.Module]
HeadFactory = Callable[[int, int], nn.Module]


@dataclass(frozen=True)
class DuneFast3RHeadConfig:
    architecture: str = "dune_fast3r_dpt"
    image_height: int = 448
    image_width: int = 560
    encoder_variant: str = "dune_vitsmall14_448"
    encoder_layers: Tuple[int, int, int, int] = (2, 5, 8, 11)
    encoder_dim: int = 384
    encoder_blocks: int = 12
    patch_size: int = 14
    dune_checkpoint: str = "./checkpoints/dune/dune_vitsmall14_448.pth"
    freeze_encoder: bool = True
    use_fast3r_decoder: bool = False
    normalize_mode: str = "minus_one_one"
    head_feature_dim: int = 256
    head_last_dim: int = 128
    depth_mode: Tuple[Any, Any, Any] = ("exp", -float("inf"), float("inf"))

    def validate(self) -> None:
        if self.architecture != "dune_fast3r_dpt":
            raise ValueError("student.architecture must be dune_fast3r_dpt")
        if self.encoder_variant not in {
            "dune_vitsmall14_336",
            "dune_vitsmall14_448",
        }:
            raise ValueError("Unsupported DUNE ViT-S variant {!r}".format(self.encoder_variant))
        if tuple(self.encoder_layers) != (2, 5, 8, 11):
            raise ValueError("encoder_layers must be the 0-based indices [2,5,8,11]")
        if self.encoder_dim != 384 or self.encoder_blocks != 12:
            raise ValueError("DUNE ViT-S requires encoder_dim=384 and encoder_blocks=12")
        if self.patch_size != 14:
            raise ValueError("DUNE ViT-S requires patch_size=14")
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("Input resolution must be divisible by patch_size=14")
        if not self.freeze_encoder:
            raise ValueError("The DUNE encoder must remain frozen")
        if self.use_fast3r_decoder:
            raise ValueError("Fast3R decoder is forbidden in this experiment")
        if self.normalize_mode != "minus_one_one":
            raise ValueError("Dataset RGB must use minus_one_one normalization")


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _build_fast3r_dpt_head(config: DuneFast3RHeadConfig) -> nn.Module:
    """Construct only Fast3R's DPT point head, with four 384-D encoder inputs."""
    ensure_fast3r_source_importable()
    from fast3r.dust3r.heads.dpt_head import PixelwiseTaskWithDPT
    from fast3r.dust3r.heads.postprocess import postprocess

    return PixelwiseTaskWithDPT(
        num_channels=3,
        feature_dim=config.head_feature_dim,
        last_dim=config.head_last_dim,
        hooks_idx=[0, 1, 2, 3],
        dim_tokens=[config.encoder_dim] * 4,
        postprocess=postprocess,
        depth_mode=tuple(config.depth_mode),
        conf_mode=None,
        head_type="regression",
        patch_size=config.patch_size,
    )


def _transformer_block_count(encoder: nn.Module) -> int:
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise RuntimeError("DUNE encoder has no Transformer blocks")
    if bool(getattr(encoder, "chunked_blocks", False)):
        return len(blocks[-1])
    return len(blocks)


class DuneFast3RHeadStudent(nn.Module):
    """Predict one camera-local point map independently for each of 16 frames.

    ``[B,T,3,H,W] -> DUNE blocks [2,5,8,11] -> Fast3R DPT ->
    [B,T,H,W,3]``. No cross-view, Fast3R, MASt3R, or LLaMA decoder is built.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | DuneFast3RHeadConfig,
        encoder_factory: Optional[EncoderFactory] = None,
        head_factory: Optional[HeadFactory] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if isinstance(config, Mapping):
            values = dict(config)
            if "encoder_layers" in values:
                values["encoder_layers"] = tuple(values["encoder_layers"])
            if "depth_mode" in values:
                values["depth_mode"] = tuple(values["depth_mode"])
            config = DuneFast3RHeadConfig(**values)
        self.config = config
        self.config.validate()
        target_device = device or torch.device("cpu")
        self.encoder = (
            encoder_factory().to(target_device)
            if encoder_factory is not None
            else load_pinned_dune_encoder(
                _project_path(self.config.dune_checkpoint), target_device
            )
        )
        self.head = (
            head_factory(self.config.encoder_dim, self.config.patch_size).to(target_device)
            if head_factory is not None
            else _build_fast3r_dpt_head(self.config).to(target_device)
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )
        # Encoder and head are constructed on the requested device above, but
        # normalization buffers are registered afterward. Move the complete
        # module once so parameters and every buffer share the same device.
        self.to(target_device)
        self._validate_encoder_contract()
        self._freeze_encoder()

    def _validate_encoder_contract(self) -> None:
        patch = getattr(self.encoder, "patch_size", None)
        if isinstance(patch, Sequence):
            patch = tuple(patch)
            correct_patch = patch == (self.config.patch_size, self.config.patch_size)
        else:
            correct_patch = int(patch) == self.config.patch_size if patch is not None else False
        if not correct_patch:
            raise RuntimeError("DUNE patch size {} != {}".format(patch, self.config.patch_size))
        dim = int(
            getattr(
                self.encoder,
                "embed_dim",
                getattr(self.encoder, "num_features", -1),
            )
        )
        if dim != self.config.encoder_dim:
            raise RuntimeError("DUNE token dim {} != {}".format(dim, self.config.encoder_dim))
        blocks = _transformer_block_count(self.encoder)
        if blocks != self.config.encoder_blocks:
            raise RuntimeError("DUNE block count {} != {}".format(blocks, self.config.encoder_blocks))
        if not hasattr(self.encoder, "get_intermediate_layers"):
            raise RuntimeError("DUNE encoder lacks get_intermediate_layers")

    def _freeze_encoder(self) -> None:
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        if not any(parameter.requires_grad for parameter in self.head.parameters()):
            raise RuntimeError("Fast3R DPT head has no trainable parameters")

    def assert_freeze_contract(self) -> None:
        if any(parameter.requires_grad for parameter in self.encoder.parameters()):
            raise RuntimeError("DUNE encoder unexpectedly has trainable parameters")
        if not any(parameter.requires_grad for parameter in self.head.parameters()):
            raise RuntimeError("Fast3R DPT head unexpectedly has no trainable parameters")

    def train(self, mode: bool = True) -> "DuneFast3RHeadStudent":
        super().train(mode)
        self.encoder.eval()
        return self

    def parameter_statistics(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        encoder = sum(parameter.numel() for parameter in self.encoder.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "dune_frozen": encoder,
        }

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,T,3,H,W]")
        batch, frames, _, height, width = images.shape
        if frames != 16:
            raise ValueError("Cross-clip student requires exactly 16 frames")
        if (height, width) != (self.config.image_height, self.config.image_width):
            raise ValueError(
                "Student input {}x{} != configured {}x{}".format(
                    height, width, self.config.image_height, self.config.image_width
                )
            )
        flat = images.reshape(batch * frames, 3, height, width)
        # Dataset uses MASt3R-style [-1,1]; DUNE expects ImageNet-normalized RGB.
        dune_input = (flat.mul(0.5).add(0.5) - self.imagenet_mean) / self.imagenet_std
        with torch.no_grad():
            features = self.encoder.get_intermediate_layers(
                dune_input,
                n=list(self.config.encoder_layers),
                reshape=False,
                return_class_token=False,
                norm=True,
            )
        if len(features) != 4:
            raise RuntimeError("DUNE returned {} intermediate features".format(len(features)))
        expected_tokens = (height // self.config.patch_size) * (
            width // self.config.patch_size
        )
        wrong = [
            tuple(value.shape)
            for value in features
            if tuple(value.shape) != (
                batch * frames,
                expected_tokens,
                self.config.encoder_dim,
            )
        ]
        if wrong:
            raise RuntimeError(
                "DUNE intermediate feature contract failed: {}".format(wrong)
            )
        with torch.cuda.amp.autocast(enabled=False):
            result = self.head([value.float() for value in features], (height, width))
        if "pts3d" not in result:
            raise KeyError("Fast3R DPT head did not return pts3d")
        points = result["pts3d"]
        expected = (batch * frames, height, width, 3)
        if tuple(points.shape) != expected:
            raise RuntimeError(
                "Fast3R DPT output {} != {}".format(tuple(points.shape), expected)
            )
        return {"pts3d_local": points.reshape(batch, frames, height, width, 3)}


__all__ = ["DuneFast3RHeadConfig", "DuneFast3RHeadStudent"]
