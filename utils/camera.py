"""Camera conventions used by the migrated VGGT-Omega cache pipeline."""

from __future__ import annotations

import importlib
from typing import Tuple

import torch


def decode_vggt_pose(
    pose_encoding: torch.Tensor,
    image_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    module = importlib.import_module("vggt_omega.utils.pose_enc")
    return module.encoding_to_camera(pose_encoding, image_shape)
