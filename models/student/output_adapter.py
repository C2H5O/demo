"""Normalize DUNE output keys and expose depth as local point-map Z."""

from __future__ import annotations

from typing import Dict

import torch


def adapt_student_outputs(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    required = ("xyz_global", "xyz_local", "conf_global", "conf_local")
    missing = [name for name in required if name not in outputs]
    if missing:
        raise KeyError("DUNE outputs are missing {}".format(missing))
    adapted = dict(outputs)
    adapted["depth"] = outputs["xyz_local"][..., 2]
    return adapted
