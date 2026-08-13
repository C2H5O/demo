"""Split DPT branches into input, 1x1 projection, and resize diagnostics.

The unchanged local DPT head is observed with temporary hooks during exactly
one complete student forward. No layer, parameter, random-image-ID behavior,
checkpoint, or training code is modified.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.scared_clip_dataset import make_scared_rgb_dataset
from diagnostics.trace_distill3r_artifacts import (
    TOKEN_GRID,
    _feature_maps,
    _require_import_below,
    _select_clip,
    save_nearest_png,
)
from utils.config import ensure_dir, load_config


STAGE_DIRECTORIES = OrderedDict(
    [
        ("branch0_input", "00_branch0_input"),
        ("branch0_projected", "01_branch0_projected"),
        ("branch0_resized", "02_branch0_resized"),
        ("branch1_input", "10_branch1_input"),
        ("branch1_projected", "11_branch1_projected"),
        ("branch1_resized", "12_branch1_resized"),
        ("branch2_input", "20_branch2_input"),
        ("branch2_projected", "21_branch2_projected"),
        ("branch2_resized", "22_branch2_resized"),
    ]
)
BRANCH_PERIODS = {0: 4, 1: 2, 2: 1}


def gradient_statistics(spatial_map: np.ndarray) -> Dict[str, float]:
    """Raw spatial statistics used to compare projection input and output."""

    value = np.asarray(spatial_map, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("spatial_map must be two-dimensional")
    gradient_x = np.abs(np.diff(value, axis=1))
    gradient_y = np.abs(np.diff(value, axis=0))
    magnitude_mean = float(np.abs(value).mean())
    magnitude_std = float(value.std())
    return {
        "mean_gx": float(gradient_x.mean()),
        "mean_gy": float(gradient_y.mean()),
        "spatial_cv": magnitude_std / (magnitude_mean + 1e-12),
        "normalized_neighbor_difference": (
            float(gradient_x.mean()) + float(gradient_y.mean())
        ) / (2.0 * magnitude_mean + 1e-12),
    }


def phase_statistics(spatial_map: np.ndarray, period: int) -> Dict[str, Any]:
    """Compute gradient imbalance between spatial phases of a resize period."""

    if period <= 1:
        return {
            "period_x": 1,
            "period_y": 1,
            "mod_x": None,
            "mod_y": None,
            "phase_ratio_x": math.nan,
            "phase_ratio_y": math.nan,
            "phase_cv_x": math.nan,
            "phase_cv_y": math.nan,
        }
    value = np.asarray(spatial_map, dtype=np.float64)
    height, width = value.shape
    gradient_x = np.abs(np.diff(value, axis=1))
    gradient_y = np.abs(np.diff(value, axis=0))
    x_positions = np.arange(1, width)
    y_positions = np.arange(1, height)
    mod_x = np.asarray(
        [gradient_x[:, x_positions % period == phase].mean() for phase in range(period)],
        dtype=np.float64,
    )
    mod_y = np.asarray(
        [gradient_y[y_positions % period == phase, :].mean() for phase in range(period)],
        dtype=np.float64,
    )

    def summarize(values: np.ndarray) -> Tuple[float, float]:
        ratio = float(values.max() / (values.min() + 1e-12))
        coefficient = float(values.std() / (values.mean() + 1e-12))
        return ratio, coefficient

    ratio_x, cv_x = summarize(mod_x)
    ratio_y, cv_y = summarize(mod_y)
    return {
        "period_x": period,
        "period_y": period,
        "mod_x": [float(item) for item in mod_x],
        "mod_y": [float(item) for item in mod_y],
        "phase_ratio_x": ratio_x,
        "phase_ratio_y": ratio_y,
        "phase_cv_x": cv_x,
        "phase_cv_y": cv_y,
    }


def mean_phase_template(spatial_map: np.ndarray, period: int) -> np.ndarray:
    """Average corresponding pixels in all non-overlapping period x period cells."""

    value = np.asarray(spatial_map, dtype=np.float64)
    height, width = value.shape
    if period <= 0 or height % period or width % period:
        raise ValueError("Map {} is not divisible by period {}".format(value.shape, period))
    return value.reshape(height // period, period, width // period, period).mean(axis=(0, 2))


def deconv_kernel_energy(module: nn.ConvTranspose2d) -> np.ndarray:
    """Mean absolute trained weight per spatial kernel phase."""

    weight = module.weight.detach().float().cpu().numpy()
    return np.abs(weight).mean(axis=(0, 1))


class BranchStage2Trace:
    """Capture DPT branch internals without changing their execution."""

    def __init__(self, model: nn.Module, frame_index: int, frame_count: int) -> None:
        self.model = model
        self.frame_index = frame_index
        self.frame_count = frame_count
        self.handles: List[Any] = []
        self.calls: MutableMapping[str, List[torch.Tensor]] = OrderedDict()
        self.sequence_calls: MutableMapping[int, List[torch.Tensor]] = OrderedDict()
        self.last_layer_calls: MutableMapping[int, List[torch.Tensor]] = OrderedDict()

    def _capture_pre(self, name: str):
        def callback(_module: nn.Module, inputs: Tuple[Any, ...]) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                raise TypeError("{} pre-hook expected a tensor input".format(name))
            self.calls.setdefault(name, []).append(inputs[0].detach().float().cpu())

        return callback

    def _capture_output(self, name: str):
        def callback(_module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
            if not torch.is_tensor(output):
                raise TypeError("{} hook expected a tensor output".format(name))
            self.calls.setdefault(name, []).append(output.detach().float().cpu())

        return callback

    def _capture_branch_output(self, branch: int, target: MutableMapping[int, List[torch.Tensor]]):
        def callback(_module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
            if not torch.is_tensor(output):
                raise TypeError("branch{} output hook expected a tensor".format(branch))
            target.setdefault(branch, []).append(output.detach().float().cpu())

        return callback

    def register(self) -> None:
        dpt = self.model.student.downstream_head_local.dpt
        if len(dpt.act_postprocess) < 3:
            raise RuntimeError("Local DPT exposes fewer than three act_postprocess branches")
        for branch in range(3):
            sequence = dpt.act_postprocess[branch]
            if not isinstance(sequence, nn.Sequential) or not sequence:
                raise RuntimeError("act_postprocess[{}] is not a non-empty Sequential".format(branch))
            projection = sequence[0]
            if not isinstance(projection, nn.Conv2d) or projection.kernel_size != (1, 1):
                raise RuntimeError("branch{} does not start with the expected 1x1 Conv2d".format(branch))
            if branch in (0, 1):
                resize = sequence[1]
                expected = BRANCH_PERIODS[branch]
                if not isinstance(resize, nn.ConvTranspose2d):
                    raise RuntimeError("branch{} resize is not ConvTranspose2d".format(branch))
                if resize.kernel_size != (expected, expected) or resize.stride != (expected, expected):
                    raise RuntimeError(
                        "branch{} expected deconv k=s={}, got kernel={} stride={}".format(
                            branch, expected, resize.kernel_size, resize.stride
                        )
                    )
            self.handles.append(sequence.register_forward_pre_hook(self._capture_pre("branch{}_input".format(branch))))
            self.handles.append(projection.register_forward_hook(self._capture_output("branch{}_projected".format(branch))))
            self.handles.append(
                sequence.register_forward_hook(
                    self._capture_branch_output(branch, self.sequence_calls)
                )
            )
            self.handles.append(
                sequence[-1].register_forward_hook(
                    self._capture_branch_output(branch, self.last_layer_calls)
                )
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _select(self, tensors: List[torch.Tensor], label: str) -> torch.Tensor:
        if not tensors:
            raise RuntimeError("{} hook did not run".format(label))
        combined = torch.cat(tensors, dim=0)
        if combined.shape[0] != self.frame_count:
            raise RuntimeError(
                "{} captured {} views, expected {}".format(label, combined.shape[0], self.frame_count)
            )
        return combined[self.frame_index]

    def selected(self) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, float]]:
        features: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        consistency: Dict[str, float] = {}
        for branch in range(3):
            features["branch{}_input".format(branch)] = self._select(
                self.calls.get("branch{}_input".format(branch), []), "branch{}_input".format(branch)
            )
            features["branch{}_projected".format(branch)] = self._select(
                self.calls.get("branch{}_projected".format(branch), []), "branch{}_projected".format(branch)
            )
            sequence_output = self._select(self.sequence_calls.get(branch, []), "branch{}_sequence".format(branch))
            last_output = self._select(self.last_layer_calls.get(branch, []), "branch{}_last_layer".format(branch))
            features["branch{}_resized".format(branch)] = sequence_output
            consistency["branch{}_sequence_vs_last_max_abs_difference".format(branch)] = float(
                (sequence_output - last_output).abs().max().item()
            )
            if branch == 2:
                consistency["branch2_projected_vs_resized_max_abs_difference"] = float(
                    (features["branch2_projected"] - sequence_output).abs().max().item()
                )
        return features, consistency


def _projection_change(input_row: Mapping[str, Any], projected_row: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "spatial_cv_ratio": float(projected_row["spatial_cv"]) / (float(input_row["spatial_cv"]) + 1e-12),
        "normalized_neighbor_difference_ratio": float(projected_row["normalized_neighbor_difference"])
        / (float(input_row["normalized_neighbor_difference"]) + 1e-12),
    }


def projection_comparison(
    input_map: np.ndarray,
    projected_map: np.ndarray,
    input_row: Mapping[str, Any],
    projected_row: Mapping[str, Any],
) -> Dict[str, float]:
    """Compare same-lattice norm structure without subtracting unlike channels."""

    if input_map.shape != projected_map.shape:
        raise ValueError("Projection comparison requires the same H/W lattice")
    input_flat = np.asarray(input_map, dtype=np.float64).reshape(-1)
    projected_flat = np.asarray(projected_map, dtype=np.float64).reshape(-1)
    input_centered = input_flat - input_flat.mean()
    projected_centered = projected_flat - projected_flat.mean()
    denominator = float(
        np.sqrt(np.dot(input_centered, input_centered))
        * np.sqrt(np.dot(projected_centered, projected_centered))
    )
    correlation = (
        float(np.dot(input_centered, projected_centered) / denominator)
        if denominator > 0.0
        else math.nan
    )
    return {"norm_map_pearson_correlation": correlation, **_projection_change(input_row, projected_row)}


def _is_phase_imbalanced(row: Mapping[str, Any], ratio_threshold: float, cv_threshold: float) -> bool:
    ratios = (float(row["phase_ratio_x"]), float(row["phase_ratio_y"]))
    cvs = (float(row["phase_cv_x"]), float(row["phase_cv_y"]))
    return max(ratios) >= ratio_threshold and max(cvs) >= cv_threshold


def diagnose(
    rows_by_stage: Mapping[str, Mapping[str, Any]],
    ratio_threshold: float,
    cv_threshold: float,
    projection_change_threshold: float,
) -> Dict[str, Any]:
    """Conservative automatic interpretation of branch0 versus controls."""

    branch0 = rows_by_stage["branch0_resized"]
    branch1 = rows_by_stage["branch1_resized"]
    branch0_strong = _is_phase_imbalanced(branch0, ratio_threshold, cv_threshold)
    branch1_strong = _is_phase_imbalanced(branch1, ratio_threshold, cv_threshold)
    projection_change = _projection_change(
        rows_by_stage["branch0_input"], rows_by_stage["branch0_projected"]
    )
    projection_changed = max(projection_change.values()) >= projection_change_threshold

    if branch0_strong and branch1_strong:
        case = 3
        message = (
            "The issue is likely generic to DPT transposed-convolution resizing rather "
            "than specific to the x4 branch."
        )
    elif branch0_strong and not branch1_strong and not projection_changed:
        case = 1
        message = "ConvTranspose4 is the first confirmed source of periodic spatial artifact."
    elif branch0_strong and not branch1_strong:
        case = 4
        message = (
            "The x4 transposed-convolution branch is substantially more phase-imbalanced "
            "than the x2 control branch; projection statistics also changed, so inspect "
            "branch0_projected before assigning sole causality."
        )
    elif projection_changed:
        case = 2
        message = (
            "Artifact may emerge during branch0 channel projection or was latent in F0; "
            "ConvTranspose4 does not cross the configured phase-imbalance threshold."
        )
    else:
        case = 0
        message = "No stage crosses the configured automatic thresholds; inspect the raw maps."
    return {
        "case": case,
        "diagnosis": message,
        "branch0_phase_imbalanced": branch0_strong,
        "branch1_phase_imbalanced": branch1_strong,
        "branch0_projection_change": projection_change,
        "thresholds": {
            "phase_ratio": ratio_threshold,
            "phase_cv": cv_threshold,
            "projection_change": projection_change_threshold,
        },
        "interpretation_note": (
            "A 1x1 convolution cannot create a new sub-token spatial lattice. Its evidence "
            "is limited to changes in native 32x40 norm structure; repeated mod-4 phases "
            "can only be confirmed after the x4 spatial expansion."
        ),
    }


def _save_overview(
    output_dir: Path,
    norm_maps: Mapping[str, np.ndarray],
    rows_by_stage: Mapping[str, Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 13))
    for axis, stage in zip(axes.flat, STAGE_DIRECTORIES):
        row = rows_by_stage[stage]
        title = "{} {}\ngx={:.3g} gy={:.3g}".format(
            stage, tuple(norm_maps[stage].shape), row["mean_gx"], row["mean_gy"]
        )
        if stage.endswith("resized") and int(row["period_x"]) > 1:
            title += "\nphase ratio x={:.3f} y={:.3f}".format(
                row["phase_ratio_x"], row["phase_ratio_y"]
            )
        axis.imshow(norm_maps[stage], cmap="viridis", interpolation="nearest")
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.suptitle("Distill3R DPT branch projection vs resize (one forward)")
    figure.tight_layout()
    figure.savefig(output_dir / "artifact_trace_stage2_overview.png", dpi=160)
    plt.close(figure)


def _weight_diagnostics(dpt: nn.Module, output_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for branch in (0, 1):
        module = dpt.act_postprocess[branch][1]
        if not isinstance(module, nn.ConvTranspose2d):
            raise RuntimeError("branch{} second module is not ConvTranspose2d".format(branch))
        weight = module.weight.detach().float().cpu()
        energy = deconv_kernel_energy(module)
        np.save(output_dir / "branch{}_deconv_kernel_energy.npy".format(branch), energy.astype(np.float32))
        save_nearest_png(
            output_dir / "branch{}_deconv_kernel_energy.png".format(branch), energy, "magma"
        )
        result["branch{}".format(branch)] = {
            "module": repr(module),
            "weight_shape": list(weight.shape),
            "weight_mean": float(weight.mean().item()),
            "weight_std": float(weight.std().item()),
            "weight_abs_mean": float(weight.abs().mean().item()),
            "kernel_energy_min": float(energy.min()),
            "kernel_energy_max": float(energy.max()),
            "kernel_energy_max_min_ratio": float(energy.max() / (energy.min() + 1e-12)),
            "kernel_energy_cv": float(energy.std() / (energy.mean() + 1e-12)),
        }
    return result


def run_stage2(args: argparse.Namespace) -> Path:
    from models.student.distill3r_wrapper import DISTILL3R_FAST3R_ROOT, DISTILL3R_ROOT
    from visualization.scared_student import load_student

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config)
    device = torch.device(args.device or str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostics requested but CUDA is unavailable")
    dataset = make_scared_rgb_dataset(config["dataset"], args.split)
    dataset_index, record = _select_clip(
        dataset, args.sequence_id, args.clip_offset, args.clip_index
    )
    sample = dataset[dataset_index]
    images = sample["images"].unsqueeze(0).to(device)
    _, frame_count, _, height, width = images.shape
    if (height, width) != (448, 560):
        raise RuntimeError("Stage 2 requires 448x560 input, got {}x{}".format(height, width))
    if not 0 <= args.frame_index < frame_count:
        raise IndexError("frame_index={} is outside [0,{})".format(args.frame_index, frame_count))

    model = load_student(args.checkpoint, config, device)
    model.eval()
    if (height // int(model.student.patch_size), width // int(model.student.patch_size)) != TOKEN_GRID:
        raise RuntimeError("The live DUNE token grid is not 32x40")
    dpt = model.student.downstream_head_local.dpt

    dpt_module = inspect.getmodule(dpt.__class__)
    student_module = inspect.getmodule(model.student.__class__)
    if dpt_module is None or student_module is None:
        raise RuntimeError("Could not resolve live Distill3R/Fast3R modules")
    dpt_import = Path(inspect.getfile(dpt_module)).resolve()
    student_import = Path(inspect.getfile(student_module)).resolve()
    _require_import_below(dpt_import, DISTILL3R_FAST3R_ROOT.resolve(), "Fast3R DPT")
    _require_import_below(student_import, DISTILL3R_ROOT.resolve(), "Distill3R student")

    trace = BranchStage2Trace(model, args.frame_index, frame_count)
    trace.register()
    amp_enabled = (
        bool(config.get("evaluation", {}).get("amp", True))
        and device.type == "cuda"
        and not args.no_amp
    )
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            model(images)  # Exactly one unchanged model forward.
    finally:
        trace.remove()
    features, consistency = trace.selected()

    expected_shapes = {
        "branch0_input": (384, 32, 40),
        "branch0_projected": (96, 32, 40),
        "branch0_resized": (96, 128, 160),
        "branch1_input": (384, 32, 40),
        "branch1_projected": (192, 32, 40),
        "branch1_resized": (192, 64, 80),
        "branch2_input": (384, 32, 40),
        "branch2_projected": (384, 32, 40),
        "branch2_resized": (384, 32, 40),
    }
    wrong = {
        name: (tuple(features[name].shape), expected)
        for name, expected in expected_shapes.items()
        if tuple(features[name].shape) != expected
    }
    if wrong:
        raise RuntimeError("Live DPT branch shapes differ from the pinned contract: {}".format(wrong))

    output_dir = ensure_dir(args.output_dir)
    norm_maps: "OrderedDict[str, np.ndarray]" = OrderedDict()
    rows: List[Dict[str, Any]] = []
    periodicity: Dict[str, Any] = {}
    for stage, directory_name in STAGE_DIRECTORIES.items():
        stage_dir = ensure_dir(output_dir / directory_name)
        feature = features[stage]
        maps = _feature_maps(feature)
        np.save(stage_dir / "feature.npy", feature.numpy().astype(np.float32))
        for map_name, spatial_map in maps.items():
            np.save(stage_dir / "{}.npy".format(map_name), spatial_map.astype(np.float32))
            save_nearest_png(stage_dir / "{}.png".format(map_name), spatial_map)
        norm = maps["norm"]
        norm_maps[stage] = norm
        branch = int(stage[len("branch")])
        period = BRANCH_PERIODS[branch] if stage.endswith("resized") else 1
        gradients = gradient_statistics(norm)
        phase = phase_statistics(norm, period)
        row = {
            "stage": stage,
            "channels": int(feature.shape[0]),
            "H": int(feature.shape[1]),
            "W": int(feature.shape[2]),
            **gradients,
            "period_x": phase["period_x"],
            "period_y": phase["period_y"],
            "phase_ratio_x": phase["phase_ratio_x"],
            "phase_ratio_y": phase["phase_ratio_y"],
            "phase_cv_x": phase["phase_cv_x"],
            "phase_cv_y": phase["phase_cv_y"],
        }
        rows.append(row)
        if period > 1:
            periodicity[stage] = phase
            template = mean_phase_template(norm, period)
            prefix = "branch{}_mean_phase_template".format(branch)
            np.save(output_dir / "{}.npy".format(prefix), template.astype(np.float32))
            save_nearest_png(output_dir / "{}.png".format(prefix), template, "magma")
        print("{:<22} {}".format(stage, list(feature.shape)))

    fields = list(rows[0].keys())
    with (output_dir / "artifact_stage2_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "artifact_stage2_periodicity.json").write_text(
        json.dumps(periodicity, indent=2, allow_nan=True), encoding="utf-8"
    )
    rows_by_stage = {row["stage"]: row for row in rows}
    projection_metrics = {
        "branch{}".format(branch): projection_comparison(
            norm_maps["branch{}_input".format(branch)],
            norm_maps["branch{}_projected".format(branch)],
            rows_by_stage["branch{}_input".format(branch)],
            rows_by_stage["branch{}_projected".format(branch)],
        )
        for branch in range(3)
    }
    (output_dir / "projection_comparison.json").write_text(
        json.dumps(projection_metrics, indent=2), encoding="utf-8"
    )
    diagnosis = diagnose(
        rows_by_stage,
        args.phase_ratio_threshold,
        args.phase_cv_threshold,
        args.projection_change_threshold,
    )
    (output_dir / "artifact_stage2_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2), encoding="utf-8"
    )
    weight_metrics = _weight_diagnostics(dpt, output_dir)
    (output_dir / "deconv_weight_metrics.json").write_text(
        json.dumps(weight_metrics, indent=2), encoding="utf-8"
    )
    _save_overview(output_dir, norm_maps, rows_by_stage)

    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "sequence_id": str(sample["sequence_id"]),
        "dataset_index": dataset_index,
        "clip_start": int(record.clip_start),
        "frame_index": args.frame_index,
        "frame_name": sample["frame_names"][args.frame_index],
        "seed": args.seed,
        "input_shape": list(images.shape),
        "forward_count": 1,
        "network_unchanged": True,
        "amp_enabled": amp_enabled,
        "student_import_path": str(student_import),
        "dpt_import_path": str(dpt_import),
        "branch_output_consistency": consistency,
        "projection_comparison": projection_metrics,
        "actual_modules": {
            "branch0": repr(dpt.act_postprocess[0]),
            "branch1": repr(dpt.act_postprocess[1]),
            "branch2": repr(dpt.act_postprocess[2]),
        },
    }
    (output_dir / "artifact_stage2_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Imported Distill3R student from {}".format(student_import))
    print("Imported Fast3R DPT from {}".format(dpt_import))
    for name, difference in consistency.items():
        print("{}={:.9g}".format(name, difference))
    print("Diagnosis: {}".format(diagnosis["diagnosis"]))
    print("Wrote Stage 2 trace to {}".format(output_dir))
    return output_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/student_distillation.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--sequence-id", "--sequence", dest="sequence_id", default=None)
    parser.add_argument("--clip-offset", type=int, default=0)
    parser.add_argument("--clip-index", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("diagnostics/artifact_trace_stage2")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--phase-ratio-threshold", type=float, default=1.25)
    parser.add_argument("--phase-cv-threshold", type=float, default=0.10)
    parser.add_argument("--projection-change-threshold", type=float, default=1.50)
    args = parser.parse_args(argv)
    if args.clip_index is not None and args.sequence_id is not None:
        parser.error("--clip-index and --sequence-id are mutually exclusive")
    return args


def main() -> None:
    run_stage2(parse_args())


if __name__ == "__main__":
    main()
