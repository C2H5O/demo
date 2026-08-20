"""Verify pinned sources, checkpoints, CUDA, and 448x560 DUNE-MASt3R forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.student.dune_mast3r_adapter import DuneMast3RStudent
from models.student.official_mast3r import (
    DUNE_ROOT,
    MAST3R_ROOT,
    ensure_official_sources_importable,
)
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vggtomast3r_v1.yaml")
    parser.add_argument("--imports-only", action="store_true")
    args = parser.parse_args()
    ensure_official_sources_importable()
    from mast3r.model import AsymmetricMASt3RWithDUNEBackbone  # noqa: F401
    from model.dune import load_dune_encoder_from_checkpoint  # noqa: F401

    config = load_config(args.config)
    result = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mast3r_source": str(MAST3R_ROOT),
        "dune_source": str(DUNE_ROOT),
    }
    if not args.imports_only:
        if not torch.cuda.is_available():
            raise RuntimeError("Full verification requires CUDA")
        model = DuneMast3RStudent(config["student"], device=torch.device("cuda")).eval()
        actual_patch_size = model.dune_encoder.patch_size
        if isinstance(actual_patch_size, tuple):
            actual_patch_size = actual_patch_size[0]
        if int(actual_patch_size) != 14:
            raise RuntimeError("Loaded DUNE encoder patch size is {}, expected 14".format(actual_patch_size))
        images = torch.zeros(1, 2, 3, 448, 560, device="cuda")
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            output = model(images)
        result["output_shapes"] = {key: list(value.shape) for key, value in output.items()}
        if not all(torch.isfinite(value).all() for value in output.values()):
            raise RuntimeError("Official 448x560 forward produced NaN or Inf")
        result["dune_patch_size"] = int(actual_patch_size)
        result["parameter_statistics"] = model.parameter_statistics()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
