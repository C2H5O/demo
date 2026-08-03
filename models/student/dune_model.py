"""DUNE ViT-Small/14 encoder with efficient multi-view point-map heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DUNEStudentConfig:
    image_size: int = 336
    patch_size: int = 14
    embed_dim: int = 384
    encoder_depth: int = 12
    encoder_heads: int = 6
    fusion_layers: int = 6
    fusion_heads: int = 6
    scene_layers: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_views: int = 16
    encoder_checkpoint: Optional[str] = None
    freeze_encoder: bool = False


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images).flatten(2).transpose(1, 2)


class LayerScale(nn.Module):
    def __init__(self, dimension: int, initial_value: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dimension,), initial_value))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.gamma


class Attention(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dimension % heads != 0:
            raise ValueError("embed_dim must be divisible by attention heads")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.dropout = dropout
        self.qkv = nn.Linear(dimension, 3 * dimension, bias=True)
        self.proj = nn.Linear(dimension, dimension, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, dimension = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, count, 3, self.heads, self.head_dimension)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, count, dimension)
        return self.proj(attended)


class MLP(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(tokens)))))


class DUNEBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, eps=1e-6)
        self.attn = Attention(dimension, heads, dropout)
        self.ls1 = LayerScale(dimension)
        self.norm2 = nn.LayerNorm(dimension, eps=1e-6)
        self.mlp = MLP(dimension, int(dimension * mlp_ratio), dropout)
        self.ls2 = LayerScale(dimension)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.ls1(self.attn(self.norm1(tokens)))
        return tokens + self.ls2(self.mlp(self.norm2(tokens)))


class DUNEVisionTransformer(nn.Module):
    """Architecture-compatible DUNE ViT-Small encoder for local checkpoints."""

    def __init__(self, config: DUNEStudentConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("DUNE image_size must be divisible by patch_size")
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        self.num_register_tokens = 4
        grid = config.image_size // config.patch_size
        self.patch_embed = PatchEmbed(config.patch_size, config.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, config.embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, config.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, config.embed_dim))
        self.blocks = nn.ModuleList([
            DUNEBlock(config.embed_dim, config.encoder_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.encoder_depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self._initialize()

    def _initialize(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _position_embedding(self, grid_height: int, grid_width: int) -> torch.Tensor:
        patch_positions = self.pos_embed[:, 1:]
        source_side = int(math.sqrt(patch_positions.shape[1]))
        if source_side * source_side != patch_positions.shape[1]:
            raise RuntimeError("DUNE positional embedding does not form a square grid")
        if (source_side, source_side) != (grid_height, grid_width):
            patch_positions = patch_positions.reshape(1, source_side, source_side, self.embed_dim)
            patch_positions = patch_positions.permute(0, 3, 1, 2)
            patch_positions = F.interpolate(
                patch_positions,
                size=(grid_height, grid_width),
                mode="bicubic",
                align_corners=False,
            )
            patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(1, grid_height * grid_width, self.embed_dim)
        return torch.cat((self.pos_embed[:, :1], patch_positions), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Input resolution {}x{} must be divisible by patch size {}".format(height, width, self.patch_size))
        patches = self.patch_embed(images)
        positions = self._position_embedding(height // self.patch_size, width // self.patch_size)
        cls = self.cls_token.expand(images.shape[0], -1, -1) + positions[:, :1]
        registers = self.register_tokens.expand(images.shape[0], -1, -1)
        patches = patches + positions[:, 1:]
        tokens = torch.cat((cls, registers, patches), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 1 + self.num_register_tokens :]

    def load_dune_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Strictly load all encoder tensors from a DUNE training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError("DUNE encoder checkpoint not found: {}".format(checkpoint_path))
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        encoder_state: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if not key.startswith("encoder."):
                continue
            clean_key = key[len("encoder.") :]
            if clean_key.startswith("blocks.0."):
                clean_key = "blocks." + clean_key[len("blocks.0.") :]
            encoder_state[clean_key] = value
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "DUNE checkpoint does not match the ViT-Small encoder. Missing: {}; unexpected: {}".format(
                    missing, unexpected
                )
            )


class PointMapHead(nn.Module):
    """Resize-convolution decoder producing dense XYZ and confidence maps.

    Bilinear resize followed by a regular convolution avoids the spatial-phase
    imbalance of transposed convolution. That imbalance is especially visible
    when the ViT patch grid is expanded to a dense depth map and can otherwise
    leave a checkerboard pattern at the 14-pixel patch period.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(dimension, 256, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refine_half = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refine_quarter = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1),
        )

    def forward(self, tokens: torch.Tensor, grid: Tuple[int, int], output_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, dimension = tokens.shape
        features = tokens.transpose(1, 2).reshape(batch, dimension, grid[0], grid[1])
        features = self.input_projection(features)
        features = F.interpolate(features, scale_factor=2.0, mode="bilinear", align_corners=False)
        features = self.refine_half(features)
        features = F.interpolate(features, scale_factor=2.0, mode="bilinear", align_corners=False)
        features = self.refine_quarter(features)
        output = self.output_projection(features)
        output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=False)
        xyz = output[:, :3].permute(0, 2, 3, 1).contiguous()
        confidence = torch.sigmoid(output[:, 3])
        return xyz, confidence


class DUNEViTSmallPointMapStudent(nn.Module):
    """DUNE encoder + efficient temporal fusion + global/local point heads."""

    def __init__(self, config: Union[DUNEStudentConfig, Dict[str, Any]]) -> None:
        super().__init__()
        self.config = DUNEStudentConfig(**config) if isinstance(config, dict) else config
        if self.config.embed_dim != 384 or self.config.encoder_depth != 12 or self.config.encoder_heads != 6:
            raise ValueError("dune_vitsmall14 requires embed_dim=384, encoder_depth=12, encoder_heads=6")
        if self.config.patch_size != 14:
            raise ValueError("dune_vitsmall14 requires patch_size=14")
        self.encoder = DUNEVisionTransformer(self.config)
        if self.config.encoder_checkpoint:
            self.encoder.load_dune_checkpoint(self.config.encoder_checkpoint)
        if self.config.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

        self.view_embedding = nn.Parameter(torch.zeros(1, self.config.max_views, 1, self.config.embed_dim))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.config.embed_dim,
            nhead=self.config.fusion_heads,
            dim_feedforward=int(self.config.embed_dim * self.config.mlp_ratio),
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_fusion = nn.TransformerEncoder(temporal_layer, num_layers=self.config.fusion_layers)
        scene_layer = nn.TransformerEncoderLayer(
            d_model=self.config.embed_dim,
            nhead=self.config.fusion_heads,
            dim_feedforward=int(self.config.embed_dim * self.config.mlp_ratio),
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scene_fusion = nn.TransformerEncoder(scene_layer, num_layers=self.config.scene_layers)
        self.global_head = PointMapHead(self.config.embed_dim)
        self.local_head = PointMapHead(self.config.embed_dim)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("images must have shape [B,T,3,H,W], got {}".format(tuple(images.shape)))
        batch, frames, channels, height, width = images.shape
        if frames > self.config.max_views:
            raise ValueError("Input has {} frames, but max_views={}".format(frames, self.config.max_views))
        flattened = images.reshape(batch * frames, channels, height, width)
        patch_tokens = self.encoder(flattened)
        patches = patch_tokens.shape[1]
        tokens = patch_tokens.reshape(batch, frames, patches, self.config.embed_dim)
        tokens = tokens + self.view_embedding[:, :frames]

        # Patch-wise temporal attention is O(P*T^2), avoiding O((P*T)^2)
        # memory while preserving cross-frame communication at every location.
        temporal = tokens.permute(0, 2, 1, 3).reshape(batch * patches, frames, self.config.embed_dim)
        temporal = self.temporal_fusion(temporal)
        temporal = temporal.reshape(batch, patches, frames, self.config.embed_dim).permute(0, 2, 1, 3)
        scene = self.scene_fusion(tokens.mean(dim=2))
        fused = temporal + scene.unsqueeze(2)

        grid = (height // self.config.patch_size, width // self.config.patch_size)
        fused_flat = fused.reshape(batch * frames, patches, self.config.embed_dim)
        global_xyz, global_conf = self.global_head(fused_flat, grid, (height, width))
        local_xyz, local_conf = self.local_head(fused_flat, grid, (height, width))
        return {
            "xyz_global": global_xyz.reshape(batch, frames, height, width, 3),
            "conf_global": global_conf.reshape(batch, frames, height, width),
            "xyz_local": local_xyz.reshape(batch, frames, height, width, 3),
            "conf_local": local_conf.reshape(batch, frames, height, width),
        }
