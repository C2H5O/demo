"""Fail fast unless the environment and current experiment config agree."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml


EXPECTED_PYTHON = (3, 10, 20)
EXPECTED_TORCH = "2.3.1"
EXPECTED_CUDA = "12.1"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_RESOLUTION = (448, 560)


def main() -> None:
    versions = {
        "python": sys.version_info[:3],
        "torch": torch.__version__.split("+")[0],
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
    }
    expected = {
        "python": EXPECTED_PYTHON,
        "torch": EXPECTED_TORCH,
        "cuda": EXPECTED_CUDA,
        "numpy": EXPECTED_NUMPY,
    }
    if versions != expected:
        raise RuntimeError("Environment mismatch: expected {}; got {}".format(expected, versions))

    root = Path(__file__).resolve().parents[1]
    with (root / "configs" / "crossclip_teacher_projection.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    dataset = config["dataset"]
    student = config["student"]
    teacher = config["teacher"]
    resolutions = {
        (int(dataset["image_height"]), int(dataset["image_width"])),
        (int(student["image_height"]), int(student["image_width"])),
    }
    if resolutions != {EXPECTED_RESOLUTION}:
        raise RuntimeError("Dataset/student resolutions must both be 448x560")
    if dataset["clip_length"] != 16 or dataset["window_stride"] != 1:
        raise RuntimeError("Cross-clip data must use 16 frames at stride one")
    if list(student["encoder_layers"]) != [2, 5, 8, 11]:
        raise RuntimeError("DUNE encoder layers must be [2,5,8,11]")
    if student["freeze_encoder"] or student["use_fast3r_decoder"]:
        raise RuntimeError(
            "Current config must jointly train DUNE and keep the Fast3R decoder disabled"
        )
    if teacher["variant"] != "base" or not teacher["frozen"]:
        raise RuntimeError("Teacher must be the frozen base model")
    print("environment and cross-clip config OK")


if __name__ == "__main__":
    main()
