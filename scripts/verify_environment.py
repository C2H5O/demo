"""Fail fast unless the environment and current experiment config agree."""

from __future__ import annotations

import importlib.metadata
import os
import site
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
        try:
            numpy_distribution = importlib.metadata.distribution("numpy")
            distribution_version = numpy_distribution.version
            distribution_location = str(numpy_distribution.locate_file(""))
        except importlib.metadata.PackageNotFoundError:
            distribution_version = "not-found"
            distribution_location = "not-found"
        diagnostics = {
            "python_executable": sys.executable,
            "sys_prefix": sys.prefix,
            "numpy_import_file": np.__file__,
            "numpy_distribution_version": distribution_version,
            "numpy_distribution_location": distribution_location,
            "user_site": site.getusersitepackages(),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        }
        raise RuntimeError(
            "Environment mismatch: expected {}; got {}; diagnostics={}".format(
                expected, versions, diagnostics
            )
        )

    root = Path(__file__).resolve().parents[1]
    with (root / "configs" / "vggtoda3.yaml").open(
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
    if (dataset["clip_length"], dataset["sample_stride"], dataset["window_stride"]) != (16, 1, 8):
        raise RuntimeError("Cross-clip data must use length=16 sample_stride=1 window_stride=8")
    if student["architecture"] != "da3_small" or student["use_ray"] or student["use_ray_pose"]:
        raise RuntimeError("Current config must use DA3-Small depth+camera without ray")
    if student["patch_size"] != 14:
        raise RuntimeError("DA3-Small patch size must be 14")
    try:
        import depth_anything_3  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Run bash scripts/setup_da3.sh") from error
    if teacher["variant"] != "base" or not teacher["frozen"]:
        raise RuntimeError("Teacher must be the frozen base model")
    print("environment and VGGT-DA3 config OK")


if __name__ == "__main__":
    main()
