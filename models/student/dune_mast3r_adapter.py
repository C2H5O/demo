"""Project adapter for the official DUNE-S/14 + binocular MASt3R model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import torch
import torch.nn as nn

from models.student.official_mast3r import PROJECT_ROOT, load_pinned_dune_mast3r


@dataclass(frozen=True)
class DuneMast3RConfig:
    architecture: str = "dune_mast3r"
    image_height: int = 448
    image_width: int = 560
    patch_size: int = 14
    checkpoint: str = "./checkpoints/mast3r/dunemast3r_cvpr25_vitsmall.pth"
    dune_checkpoint: str = "./checkpoints/dune/dune_vitsmall14_448.pth"
    freeze_encoder: bool = True
    normalize_mode: str = "minus_one_one"

    def validate(self) -> None:
        if self.architecture != "dune_mast3r":
            raise ValueError("student.architecture must be dune_mast3r")
        if (self.image_height, self.image_width) != (448, 560):
            raise ValueError("V1 requires 448x560 inputs")
        if self.patch_size != 14:
            raise ValueError("DUNE-S V1 requires patch_size=14")
        if self.image_height % 14 or self.image_width % 14:
            raise ValueError("Input resolution must be divisible by patch size 14")
        if not self.freeze_encoder:
            raise ValueError("V1 requires a frozen DUNE encoder")
        if self.normalize_mode != "minus_one_one":
            raise ValueError("Official MASt3R views require minus_one_one RGB")


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


class DuneMast3RStudent(nn.Module):
    """Expose exactly two reference-frame point maps at full SCARED resolution."""

    def __init__(
        self,
        config: Mapping[str, Any] | DuneMast3RConfig,
        model_factory: Optional[Callable[[], nn.Module]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = DuneMast3RConfig(**dict(config)) if isinstance(config, Mapping) else config
        self.config.validate()
        target_device = device or torch.device("cpu")
        if model_factory is None:
            self.model = load_pinned_dune_mast3r(
                _project_path(self.config.checkpoint),
                _project_path(self.config.dune_checkpoint),
                target_device,
            )
        else:
            self.model = model_factory().to(target_device)
        self._configure_v1_trainable_parameters()
        self.assert_freeze_contract()

    @property
    def dune_encoder(self) -> nn.Module:
        module = getattr(self.model, "dune_backbone", None)
        if not isinstance(module, nn.Module):
            raise RuntimeError("Official DUNE-MASt3R model has no dune_backbone")
        return module

    @property
    def mast3r(self) -> nn.Module:
        module = getattr(self.model, "mast3r", None)
        if not isinstance(module, nn.Module):
            raise RuntimeError("Official DUNE-MASt3R model has no mast3r module")
        return module

    def _configure_v1_trainable_parameters(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        trainable_modules = []
        for name in (
            "decoder_embed", "dec_blocks", "dec_blocks2", "dec_norm",
            "downstream_head1", "downstream_head2",
        ):
            module = getattr(self.mast3r, name, None)
            if isinstance(module, nn.Module):
                trainable_modules.append(module)
        if not trainable_modules:
            raise RuntimeError("Could not locate official MASt3R decoder/head modules")
        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def assert_freeze_contract(self) -> None:
        if any(parameter.requires_grad for parameter in self.dune_encoder.parameters()):
            raise RuntimeError("DUNE encoder has trainable parameters")
        trainable = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("MASt3R decoder/head have no trainable parameters")
        allowed = (
            "mast3r.decoder_embed", "mast3r.dec_blocks", "mast3r.dec_blocks2",
            "mast3r.dec_norm", "mast3r.downstream_head1", "mast3r.downstream_head2",
            "mast3r.head1", "mast3r.head2",
        )
        invalid = [name for name in trainable if not name.startswith(allowed)]
        if invalid:
            raise RuntimeError("Unexpected trainable parameters: {}".format(invalid[:20]))

    def parameter_statistics(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        dune = sum(parameter.numel() for parameter in self.dune_encoder.parameters())
        return {"total": total, "trainable": trainable, "frozen": total - trainable, "dune_frozen": dune}

    def train(self, mode: bool = True) -> "DuneMast3RStudent":
        super().train(mode)
        self.dune_encoder.eval()
        return self

    @staticmethod
    def _view(image: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, _, height, width = image.shape
        true_shape = torch.tensor([height, width], device=image.device, dtype=torch.long)
        return {"img": image, "true_shape": true_shape.unsqueeze(0).expand(batch, -1)}

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 5 or images.shape[1] != 2 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,2,3,H,W], got {}".format(tuple(images.shape)))
        batch, _, _, height, width = images.shape
        if (height, width) != (self.config.image_height, self.config.image_width):
            raise ValueError("DUNE-MASt3R V1 expects 448x560, got {}x{}".format(height, width))
        pred1, pred2 = self.model(self._view(images[:, 0]), self._view(images[:, 1]))
        if "pts3d" not in pred1 or "pts3d_in_other_view" not in pred2:
            raise KeyError("Official DUNE-MASt3R output is missing point maps")
        output = {
            "pts3d_ref": pred1["pts3d"],
            "pts3d_other_in_ref": pred2["pts3d_in_other_view"],
        }
        expected = (batch, height, width, 3)
        wrong = {key: tuple(value.shape) for key, value in output.items() if tuple(value.shape) != expected}
        if wrong:
            raise RuntimeError("DUNE-MASt3R output contract failed: {} expected {}".format(wrong, expected))
        return output

    def reference_depth(self, images: torch.Tensor) -> torch.Tensor:
        """Depth is Z only for the first (reference-camera) output."""
        return self(images)["pts3d_ref"][..., 2]


__all__ = ["DuneMast3RConfig", "DuneMast3RStudent"]
