"""Fail fast unless the active environment matches the training contract."""

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
EXPECTED_DUNE_CHECKPOINT = "./checkpoints/dune/dune_vitsmall14_448.pth"
EXPECTED_GT_DIRECTORIES = ["data/depth", "data/scene_points"]


def main() -> None:
    python_version = sys.version_info[:3]
    torch_version = torch.__version__.split("+")[0]
    cuda_version = torch.version.cuda
    if python_version != EXPECTED_PYTHON:
        raise RuntimeError(
            "Expected Python {}, got {}".format(EXPECTED_PYTHON, python_version)
        )
    if torch_version != EXPECTED_TORCH:
        raise RuntimeError(
            "Expected PyTorch {}, got {}".format(EXPECTED_TORCH, torch.__version__)
        )
    if cuda_version != EXPECTED_CUDA:
        raise RuntimeError(
            "Expected PyTorch CUDA runtime {}, got {}".format(
                EXPECTED_CUDA, cuda_version
            )
        )
    if np.__version__ != EXPECTED_NUMPY:
        raise RuntimeError(
            "Expected NumPy {}, got {}".format(EXPECTED_NUMPY, np.__version__)
        )

    root = Path(__file__).resolve().parents[1]
    with (root / "configs" / "student_distillation.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    dataset_resolution = (
        int(config["dataset"]["image_height"]),
        int(config["dataset"]["image_width"]),
    )
    student_resolution = (
        int(config["student"]["image_height"]),
        int(config["student"]["image_width"]),
    )
    teacher_resolution = (
        int(config["teacher"]["image_height"]),
        int(config["teacher"]["image_width"]),
    )
    if not (
        dataset_resolution
        == student_resolution
        == teacher_resolution
        == EXPECTED_RESOLUTION
    ):
        raise RuntimeError(
            "Dataset/student/teacher resolutions must all be 448x560: {}".format(
                (dataset_resolution, student_resolution, teacher_resolution)
            )
        )
    student_checkpoint = str(config["student"].get("pretrained_checkpoint", ""))
    if student_checkpoint != EXPECTED_DUNE_CHECKPOINT:
        raise RuntimeError(
            "student.pretrained_checkpoint must be {}, got {}".format(
                EXPECTED_DUNE_CHECKPOINT, student_checkpoint
            )
        )
    gt_directories = list(
        config["dataset"].get("ground_truth", {}).get("relative_directories", [])
    )
    if gt_directories != EXPECTED_GT_DIRECTORIES:
        raise RuntimeError(
            "Ground-truth relative directories must be {}, got {}".format(
                EXPECTED_GT_DIRECTORIES, gt_directories
            )
        )
    evaluation_gt = str(config.get("evaluation", {}).get("gt_relative_directory", ""))
    if evaluation_gt != "data/depth":
        raise RuntimeError(
            "evaluation.gt_relative_directory must be data/depth, got {}".format(
                evaluation_gt
            )
        )
    print(
        "environment OK: Python {} PyTorch {} CUDA {} NumPy {} resolution {}x{}".format(
            ".".join(str(value) for value in python_version),
            torch.__version__,
            cuda_version,
            np.__version__,
            *EXPECTED_RESOLUTION,
        )
    )


if __name__ == "__main__":
    main()
