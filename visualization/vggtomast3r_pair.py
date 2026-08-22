"""Fixed-range V1 panel with separate camera-local point-cloud exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from datasets.scared_pair_dataset import ScaredPairDistillDataset, make_scared_pair_rgb_dataset
from models.student.dune_mast3r_adapter import DuneMast3RStudent
from utils.checkpoint import require_student_cache_protocol
from utils.config import ensure_dir, load_config
from visualization.scared_student import depth_to_magma, write_binary_ply


def _rgb(image: torch.Tensor) -> np.ndarray:
    return np.round(((image.float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy()).astype(np.uint8)


def export_pair_visualization(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    pair_index: int,
    output_dir: Path,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    point_stride: int = 4,
) -> Path:
    config = load_config(config_path)
    rgb_dataset = make_scared_pair_rgb_dataset(config["dataset"], split)
    dataset = ScaredPairDistillDataset(
        rgb_dataset, Path(config["teacher"]["cache_root"]) / split,
        config["dataset"].get("ground_truth"),
        expected_base_checkpoint=str(config["teacher"]["pretrained_checkpoint"]),
    )
    sample = dataset[pair_index]
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    require_student_cache_protocol(checkpoint)
    device = torch.device(str(config.get("device", "cuda")))
    model = DuneMast3RStudent(checkpoint.get("config", {}).get("student", config["student"]), device=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    with torch.inference_mode():
        prediction = model(sample["images"].unsqueeze(0).to(device))
    ref_points = prediction["pts3d_ref"][0].float().cpu()
    other_points = prediction["pts3d_other_local"][0].float().cpu()
    student_depth = ref_points[..., 2].numpy()
    teacher_depth = sample["target"]["pts3d_ref"][..., 2].numpy()
    gt_depth = sample["ground_truth_depth_ref"].numpy()
    gt_valid = sample["ground_truth_valid_mask_ref"].numpy()
    rgb_a, rgb_b = _rgb(sample["images"][0]), _rgb(sample["images"][1])
    valid_student = np.isfinite(student_depth) & (student_depth > min_depth) & (student_depth < max_depth)
    valid_teacher = np.isfinite(teacher_depth) & (teacher_depth > min_depth) & (teacher_depth < max_depth)
    valid_gt = gt_valid & np.isfinite(gt_depth) & (gt_depth > min_depth) & (gt_depth < max_depth)
    error = np.abs(student_depth - gt_depth)
    error_valid = valid_student & valid_gt
    teacher_color = depth_to_magma(teacher_depth, valid_teacher, min_depth, max_depth)
    student_color = depth_to_magma(student_depth, valid_student, min_depth, max_depth)
    gt_color = depth_to_magma(gt_depth, valid_gt, min_depth, max_depth)
    error_color = depth_to_magma(error, error_valid, 0.0, max_depth - min_depth)
    panels = [rgb_a, rgb_b, teacher_color, student_color, gt_color, error_color]
    output = ensure_dir(output_dir)
    for name, image in (
        ("rgb_reference", rgb_a), ("rgb_second", rgb_b),
        ("teacher_depth_reference", teacher_color),
        ("student_depth_reference", student_color),
        ("gt_depth_reference", gt_color), ("absolute_depth_error", error_color),
    ):
        Image.fromarray(image).save(output / "{}.png".format(name))
    Image.fromarray(np.concatenate(panels, axis=1)).save(output / "pair_panel.png")

    for name, points, colors in (
        ("reference_camera_local", ref_points.numpy(), rgb_a),
        ("other_camera_local", other_points.numpy(), rgb_b),
    ):
        valid = np.isfinite(points).all(axis=-1)
        sampled = np.zeros(valid.shape, dtype=bool)
        sampled[::point_stride, ::point_stride] = True
        valid &= sampled
        write_binary_ply(output / "{}.ply".format(name), points[valid], colors[valid])
    np.savez_compressed(
        output / "pair_frame_local.npz",
        pts3d_ref=ref_points.numpy(),
        pts3d_other_local=other_points.numpy(),
        coordinate_system=np.asarray(
            ["reference camera-local", "other camera-local"]
        ),
        frame_names=np.asarray(sample["frame_names"]),
    )
    metadata: Dict[str, Any] = {
        "coordinate_system": "two independent camera-local coordinate systems",
        "depth_color_range": [min_depth, max_depth],
        "error_color_range": [0.0, max_depth - min_depth],
        "frame_names": sample["frame_names"],
        "checkpoint": str(checkpoint_path),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


__all__ = ["export_pair_visualization"]
