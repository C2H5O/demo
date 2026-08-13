"""Zero-shot branch0 reconstruction ablation using one cached decoder pass.

One unchanged student forward supplies the local DPT token inputs. The selected
frame is then replayed through the original local DPT with only branch0 spatial
reconstruction varied: trained ConvTranspose4, phase-tied ConvTranspose4, or
parameter-free bilinear x4. No module parameter is mutated or saved.
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
import torch.nn.functional as F
from PIL import Image


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
from diagnostics.trace_distill3r_artifacts_stage2 import phase_statistics
from utils.config import ensure_dir, load_config


MODES = ("baseline", "phase_tied", "bilinear")
STAGE_PERIODS = OrderedDict(
    [
        ("branch0_projected", 1),
        ("branch0_resized", 4),
        ("scratch0", 4),
        ("path2", 4),
        ("path1", 8),
        ("head_conv1", 8),
        ("depth", 14),
    ]
)
OVERVIEW_STAGES = ("branch0_resized", "scratch0", "path2", "path1", "depth")


class LocalDPTInputCapture:
    """Capture the exact decoder token list entering the unchanged local DPT."""

    def __init__(self, dpt: nn.Module, frame_count: int, frame_index: int) -> None:
        self.dpt = dpt
        self.frame_count = frame_count
        self.frame_index = frame_index
        self.calls: List[List[torch.Tensor]] = []
        self.handle: Optional[Any] = None

    def _callback(self, _module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], (list, tuple)):
            raise TypeError("Local DPT hook expected a decoder-token list")
        tokens = inputs[0]
        if not tokens or not all(torch.is_tensor(item) for item in tokens):
            raise TypeError("Local DPT token list contains a non-tensor value")
        self.calls.append([item.detach() for item in tokens])

    def register(self) -> None:
        self.handle = self.dpt.register_forward_pre_hook(self._callback)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def selected_tokens(self) -> List[torch.Tensor]:
        if not self.calls:
            raise RuntimeError("The local DPT input hook did not run")
        layer_count = len(self.calls[0])
        if any(len(call) != layer_count for call in self.calls):
            raise RuntimeError("Local DPT calls returned different decoder layer counts")
        combined = [
            torch.cat([call[layer] for call in self.calls], dim=0)
            for layer in range(layer_count)
        ]
        if any(item.shape[0] != self.frame_count for item in combined):
            raise RuntimeError(
                "Captured local DPT batch does not contain {} frames".format(self.frame_count)
            )
        return [item[self.frame_index : self.frame_index + 1] for item in combined]


def phase_tied_weight(module: nn.ConvTranspose2d) -> torch.Tensor:
    """Construct a temporary weight whose spatial phases share one mapping."""

    weight = module.weight
    return weight.mean(dim=(-2, -1), keepdim=True).expand_as(weight).contiguous()


def reconstruct_branch0(
    projected: torch.Tensor,
    module: nn.ConvTranspose2d,
    mode: str,
) -> torch.Tensor:
    if mode == "baseline":
        return module(projected)
    if mode == "phase_tied":
        tied = phase_tied_weight(module)
        return F.conv_transpose2d(
            projected,
            tied,
            bias=module.bias,
            stride=module.stride,
            padding=module.padding,
            output_padding=module.output_padding,
            groups=module.groups,
            dilation=module.dilation,
        )
    if mode == "bilinear":
        return F.interpolate(
            projected,
            size=(projected.shape[-2] * 4, projected.shape[-1] * 4),
            mode="bilinear",
            align_corners=True,
        )
    raise ValueError("Unknown branch0 mode: {}".format(mode))


def _spatial_layers(
    dpt: nn.Module,
    decoder_tokens: Sequence[torch.Tensor],
    image_size: Tuple[int, int],
) -> List[torch.Tensor]:
    height, width = image_size
    grid_h = height // (dpt.stride_level * dpt.P_H)
    grid_w = width // (dpt.stride_level * dpt.P_W)
    layers = [dpt.adapt_tokens(decoder_tokens[hook]) for hook in dpt.hooks]
    spatial: List[torch.Tensor] = []
    for layer in layers:
        batch, patches, channels = layer.shape
        if patches != grid_h * grid_w:
            raise RuntimeError(
                "DPT layer has {} tokens, expected {}x{}".format(patches, grid_h, grid_w)
            )
        spatial.append(
            layer.reshape(batch, grid_h, grid_w, channels).permute(0, 3, 1, 2)
        )
    return spatial


def replay_local_dpt(
    head: nn.Module,
    decoder_tokens: Sequence[torch.Tensor],
    image_size: Tuple[int, int],
    modes: Sequence[str],
) -> Tuple["OrderedDict[str, Dict[str, torch.Tensor]]", Dict[str, Any]]:
    """Replay only Local DPT, sharing every tensor except branch0 reconstruction."""

    dpt = head.dpt
    spatial = _spatial_layers(dpt, decoder_tokens, image_size)
    branch0_projected = dpt.act_postprocess[0][0](spatial[0])
    deconv = dpt.act_postprocess[0][1]
    if not isinstance(deconv, nn.ConvTranspose2d):
        raise RuntimeError("Local DPT branch0 second module is not ConvTranspose2d")
    if deconv.kernel_size != (4, 4) or deconv.stride != (4, 4):
        raise RuntimeError("Local DPT branch0 is not the expected k=4,s=4 deconvolution")

    unchanged_resized = [dpt.act_postprocess[index](spatial[index]) for index in (1, 2, 3)]
    unchanged_scratch = [
        dpt.scratch.layer_rn[index](unchanged_resized[index - 1])
        for index in (1, 2, 3)
    ]
    path4 = dpt.scratch.refinenet4(unchanged_scratch[2])[
        :, :, : unchanged_scratch[1].shape[2], : unchanged_scratch[1].shape[3]
    ]
    path3 = dpt.scratch.refinenet3(path4, unchanged_scratch[1])
    path2 = dpt.scratch.refinenet2(path3, unchanged_scratch[0])

    results: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()
    branch0_outputs: Dict[str, torch.Tensor] = {}
    for mode in modes:
        branch0_resized = reconstruct_branch0(branch0_projected, deconv, mode)
        if tuple(branch0_resized.shape[-3:]) != (96, 128, 160):
            raise RuntimeError(
                "{} branch0 output has shape {}, expected [B,96,128,160]".format(
                    mode, tuple(branch0_resized.shape)
                )
            )
        scratch0 = dpt.scratch.layer_rn[0](branch0_resized)
        path1 = dpt.scratch.refinenet1(path2, scratch0)
        head_conv1 = dpt.head[0](path1)
        head_resized = dpt.head[1](head_conv1)
        head_conv2 = dpt.head[2](head_resized)
        activated = dpt.head[3](head_conv2)
        raw_output = dpt.head[4](activated)
        processed = head.postprocess(raw_output, head.depth_mode, head.conf_mode)
        depth = processed["pts3d"][..., 2]
        results[mode] = {
            "branch0_projected": branch0_projected,
            "branch0_resized": branch0_resized,
            "scratch0": scratch0,
            "path2": path2,
            "path1": path1,
            "head_conv1": head_conv1,
            "depth": depth,
            "raw_output": raw_output,
        }
        branch0_outputs[mode] = branch0_resized

    projected_differences = {
        "baseline_vs_{}_max_abs_difference".format(mode): float(
            (
                results["baseline"]["branch0_projected"]
                - results[mode]["branch0_projected"]
            )
            .detach()
            .float()
            .abs()
            .max()
            .item()
        )
        for mode in modes
        if mode != "baseline"
    }
    path2_differences = {
        "baseline_vs_{}_max_abs_difference".format(mode): float(
            (results["baseline"]["path2"] - results[mode]["path2"])
            .detach()
            .float()
            .abs()
            .max()
            .item()
        )
        for mode in modes
        if mode != "baseline"
    }
    tied = phase_tied_weight(deconv)
    diagnostics = {
        "branch0_projected_differences": projected_differences,
        "path2_differences": path2_differences,
        "branch0_output_shapes": {
            mode: list(branch0_outputs[mode].shape) for mode in modes
        },
        "original_weight": {
            "shape": list(deconv.weight.shape),
            "mean": float(deconv.weight.detach().float().mean().item()),
            "std": float(deconv.weight.detach().float().std().item()),
        },
        "phase_tied_weight": {
            "shape": list(tied.shape),
            "mean": float(tied.detach().float().mean().item()),
            "std": float(tied.detach().float().std().item()),
            "phase_tied_spatial_max_abs_difference": float(
                (tied - tied[:, :, :1, :1]).detach().float().abs().max().item()
            ),
        },
    }
    return results, diagnostics


def _analysis_map(stage: str, tensor: torch.Tensor) -> np.ndarray:
    selected = tensor[0].detach().float().cpu()
    if stage == "depth":
        return selected.numpy()
    return _feature_maps(selected)["norm"]


def relative_reduction(variant: float, baseline: float, excess_ratio: bool = False) -> float:
    if not np.isfinite(variant) or not np.isfinite(baseline):
        return math.nan
    if excess_ratio:
        return 1.0 - (variant - 1.0) / (baseline - 1.0 + 1e-12)
    return 1.0 - variant / (baseline + 1e-12)


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Pearson inputs must have the same shape")
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else math.nan


def lowpass_correlation(first: np.ndarray, second: np.ndarray, kernel_size: int = 8) -> float:
    tensors = []
    for value in (first, second):
        tensor = torch.from_numpy(np.asarray(value, dtype=np.float32))[None, None]
        tensors.append(
            F.avg_pool2d(tensor, kernel_size=kernel_size, stride=kernel_size)[0, 0].numpy()
        )
    return pearson_correlation(tensors[0], tensors[1])


def _finite_percentiles(values: Sequence[np.ndarray], low: float, high: float) -> Tuple[float, float]:
    finite = np.concatenate([value[np.isfinite(value)].reshape(-1) for value in values])
    if not finite.size:
        raise RuntimeError("No finite values are available for visualization")
    lower, upper = np.percentile(finite, (low, high))
    return float(lower), float(upper)


def save_colored_map(
    path: Path,
    value: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_name: str,
) -> None:
    normalized = np.zeros(value.shape, dtype=np.float32)
    finite = np.isfinite(value)
    normalized[finite] = np.clip(
        (value[finite] - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0
    )
    rgba = np.round(plt.get_cmap(cmap_name)(normalized) * 255.0).astype(np.uint8)
    rgba[~finite] = 0
    Image.fromarray(rgba, mode="RGBA").save(path)


def _save_overview(
    output_dir: Path,
    modes: Sequence[str],
    maps: Mapping[str, Mapping[str, np.ndarray]],
    rows: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(
        len(OVERVIEW_STAGES), len(modes), figsize=(5 * len(modes), 4 * len(OVERVIEW_STAGES))
    )
    axes = np.asarray(axes).reshape(len(OVERVIEW_STAGES), len(modes))
    for row_index, stage in enumerate(OVERVIEW_STAGES):
        stage_values = [maps[mode][stage] for mode in modes]
        vmin, vmax = _finite_percentiles(stage_values, 2.0, 98.0)
        for column, mode in enumerate(modes):
            metric = rows[(mode, stage)]
            axis = axes[row_index, column]
            axis.imshow(
                maps[mode][stage], cmap="magma" if stage == "depth" else "viridis",
                interpolation="nearest", vmin=vmin, vmax=vmax
            )
            axis.set_title(
                "{} / {} {}\nratio x={:.3f} y={:.3f}\nCV x={:.3f} y={:.3f}".format(
                    mode,
                    stage,
                    tuple(maps[mode][stage].shape),
                    metric["phase_ratio_x"],
                    metric["phase_ratio_y"],
                    metric["phase_cv_x"],
                    metric["phase_cv_y"],
                ),
                fontsize=8,
            )
            axis.axis("off")
    figure.suptitle("Branch0 reconstruction ablation (shared scale within each row)")
    figure.tight_layout()
    figure.savefig(output_dir / "branch0_replacement_overview.png", dpi=160)
    plt.close(figure)


def _save_profile_plot(
    path: Path,
    title: str,
    modes: Sequence[str],
    profiles: Mapping[str, Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode in modes:
        profile = profiles[mode]
        axes[0].plot(profile["mod_x"], marker="o", label=mode)
        axes[1].plot(profile["mod_y"], marker="o", label=mode)
    for axis, direction in zip(axes, ("x", "y")):
        axis.set_title("{} modulo phase".format(direction))
        axis.set_xlabel("phase (0 = cell boundary)")
        axis.set_ylabel("mean |gradient|")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _mean_reduction(row: Mapping[str, Any]) -> float:
    keys = (
        "reduction_excess_ratio_x",
        "reduction_excess_ratio_y",
        "reduction_phase_cv_x",
        "reduction_phase_cv_y",
    )
    values = [float(row[key]) for key in keys if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else math.nan


def diagnose_ablation(
    rows: Mapping[Tuple[str, str], Mapping[str, Any]],
    correlations: Mapping[str, Mapping[str, float]],
    modes: Sequence[str],
    reduction_threshold: float,
) -> Dict[str, Any]:
    if not all(mode in modes for mode in MODES):
        return {
            "case": "partial",
            "diagnosis": "Run --branch0-mode all for a causal three-way diagnosis.",
        }
    tied_branch = _mean_reduction(rows[("phase_tied", "branch0_resized")])
    bilinear_branch = _mean_reduction(rows[("bilinear", "branch0_resized")])
    tied_depth = _mean_reduction(rows[("phase_tied", "depth")])
    bilinear_depth = _mean_reduction(rows[("bilinear", "depth")])
    tied_path1 = _mean_reduction(rows[("phase_tied", "path1")])
    bilinear_path1 = _mean_reduction(rows[("bilinear", "path1")])
    branch_both = min(tied_branch, bilinear_branch) >= reduction_threshold
    depth_both = min(tied_depth, bilinear_depth) >= reduction_threshold

    if bilinear_branch >= reduction_threshold and tied_branch < reduction_threshold * 0.5:
        case = "D"
        message = (
            "Phase tying does not substantially improve branch0 while bilinear does; the "
            "non-overlapping transposed-convolution expansion is implicated beyond phase variation."
        )
    elif branch_both and max(tied_depth, bilinear_depth) < reduction_threshold * 0.5:
        case = "C"
        message = (
            "The branch0 periodicity is removed, but path1/depth remain largely unchanged; "
            "branch0 is real but not dominant in the final depth grid."
        )
    elif branch_both and depth_both:
        tied_structure = float(correlations["phase_tied"]["mean_correlation"])
        bilinear_structure = float(correlations["bilinear"]["mean_correlation"])
        if tied_structure >= bilinear_structure and tied_structure >= 0.8:
            case = "E"
            message = (
                "Independent subpixel phase mappings are the principal mechanism: phase tying "
                "and bilinear both reduce the artifact, while phase tying better preserves "
                "baseline large-scale structure."
            )
        else:
            case = "A"
            message = (
                "The periodic depth artifact is causally linked to phase-dependent x4 branch "
                "reconstruction."
            )
    elif branch_both and max(tied_path1, bilinear_path1) >= reduction_threshold * 0.5:
        case = "B"
        message = (
            "ConvTranspose4 is a major source, but downstream scratch/RefineNet retains or "
            "regenerates part of the artifact."
        )
    else:
        case = "inconclusive"
        message = "The configured reductions do not match a predefined causal case."
    return {
        "case": case,
        "diagnosis": message,
        "reduction_threshold": reduction_threshold,
        "mean_reductions": {
            "phase_tied_branch0": tied_branch,
            "bilinear_branch0": bilinear_branch,
            "phase_tied_path1": tied_path1,
            "bilinear_path1": bilinear_path1,
            "phase_tied_depth": tied_depth,
            "bilinear_depth": bilinear_depth,
        },
        "large_scale_correlations": correlations,
        "warning": (
            "This is an untrained zero-shot mechanism ablation, not an accuracy or architecture ranking."
        ),
    }


def _requested_modes(value: str) -> Tuple[str, ...]:
    if value == "all":
        return MODES
    if value == "baseline":
        return ("baseline",)
    return ("baseline", value)


def run_ablation(args: argparse.Namespace) -> Path:
    from models.student.distill3r_wrapper import DISTILL3R_FAST3R_ROOT, DISTILL3R_ROOT
    from visualization.scared_student import load_student

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    modes = _requested_modes(args.branch0_mode)
    config = load_config(args.config)
    device = torch.device(args.device or str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA ablation requested but CUDA is unavailable")
    dataset = make_scared_rgb_dataset(config["dataset"], args.split)
    dataset_index, record = _select_clip(
        dataset, args.sequence_id, args.clip_offset, args.clip_index
    )
    sample = dataset[dataset_index]
    images = sample["images"].unsqueeze(0).to(device)
    _, frame_count, _, height, width = images.shape
    if (height, width) != (448, 560):
        raise RuntimeError("Ablation requires 448x560 input, got {}x{}".format(height, width))
    if not 0 <= args.frame_index < frame_count:
        raise IndexError("frame_index={} is outside [0,{})".format(args.frame_index, frame_count))

    model = load_student(args.checkpoint, config, device)
    model.eval()
    local_head = model.student.downstream_head_local
    dpt = local_head.dpt
    dpt_module = inspect.getmodule(dpt.__class__)
    student_module = inspect.getmodule(model.student.__class__)
    if dpt_module is None or student_module is None:
        raise RuntimeError("Could not resolve live Distill3R/Fast3R modules")
    dpt_import = Path(inspect.getfile(dpt_module)).resolve()
    student_import = Path(inspect.getfile(student_module)).resolve()
    _require_import_below(dpt_import, DISTILL3R_FAST3R_ROOT.resolve(), "Fast3R DPT")
    _require_import_below(student_import, DISTILL3R_ROOT.resolve(), "Distill3R student")

    deconv = dpt.act_postprocess[0][1]
    if not isinstance(deconv, nn.ConvTranspose2d):
        raise RuntimeError("Branch0 resize is not ConvTranspose2d")
    original_weight = deconv.weight.detach().clone()
    capture = LocalDPTInputCapture(dpt, frame_count, args.frame_index)
    capture.register()
    amp_enabled = (
        bool(config.get("evaluation", {}).get("amp", True))
        and device.type == "cuda"
        and not args.no_amp
    )
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            original_prediction = model(images)  # The only encoder/decoder/model forward.
    finally:
        capture.remove()
    decoder_tokens = capture.selected_tokens()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
        results, replacement_checks = replay_local_dpt(
            local_head, decoder_tokens, (height, width), modes
        )
    model_weight_max_abs_difference = float(
        (deconv.weight.detach() - original_weight).float().abs().max().item()
    )
    if model_weight_max_abs_difference != 0.0:
        raise RuntimeError("Diagnostic replay modified the original branch0 weight")

    original_depth = original_prediction["xyz_local"][0, args.frame_index, ..., 2].float()
    replay_depth = results["baseline"]["depth"][0].float()
    replay_difference = (original_depth - replay_depth).abs()
    baseline_replay = {
        "max_abs_difference": float(replay_difference.max().item()),
        "mean_abs_difference": float(replay_difference.mean().item()),
        "allclose_rtol_5e-3_atol_5e-3": bool(
            torch.allclose(original_depth, replay_depth, rtol=5e-3, atol=5e-3)
        ),
    }
    if not baseline_replay["allclose_rtol_5e-3_atol_5e-3"]:
        raise RuntimeError("Manual baseline DPT replay does not match the original forward")

    output_dir = ensure_dir(args.output_dir)
    maps: "OrderedDict[str, OrderedDict[str, np.ndarray]]" = OrderedDict()
    metric_rows: List[Dict[str, Any]] = []
    profile_data: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        maps[mode] = OrderedDict()
        for stage, period in STAGE_PERIODS.items():
            value = _analysis_map(stage, results[mode][stage])
            maps[mode][stage] = value
            stage_dir = ensure_dir(output_dir / mode / stage)
            np.save(stage_dir / "norm.npy", value.astype(np.float32))
            save_nearest_png(
                stage_dir / "norm.png", value, "magma" if stage == "depth" else "viridis"
            )
            phase = phase_statistics(value, period)
            profile_data["{}:{}".format(mode, stage)] = phase
            metric_rows.append(
                {
                    "mode": mode,
                    "stage": stage,
                    "H": int(value.shape[0]),
                    "W": int(value.shape[1]),
                    "period_x": phase["period_x"],
                    "period_y": phase["period_y"],
                    "phase_ratio_x": phase["phase_ratio_x"],
                    "phase_ratio_y": phase["phase_ratio_y"],
                    "phase_cv_x": phase["phase_cv_x"],
                    "phase_cv_y": phase["phase_cv_y"],
                }
            )

    initial_rows = {(row["mode"], row["stage"]): row for row in metric_rows}
    for row in metric_rows:
        baseline = initial_rows[("baseline", row["stage"])]
        row["reduction_excess_ratio_x"] = relative_reduction(
            float(row["phase_ratio_x"]), float(baseline["phase_ratio_x"]), True
        )
        row["reduction_excess_ratio_y"] = relative_reduction(
            float(row["phase_ratio_y"]), float(baseline["phase_ratio_y"]), True
        )
        row["reduction_phase_cv_x"] = relative_reduction(
            float(row["phase_cv_x"]), float(baseline["phase_cv_x"])
        )
        row["reduction_phase_cv_y"] = relative_reduction(
            float(row["phase_cv_y"]), float(baseline["phase_cv_y"])
        )
    rows = {(row["mode"], row["stage"]): row for row in metric_rows}
    with (output_dir / "replacement_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    (output_dir / "replacement_periodicity.json").write_text(
        json.dumps(profile_data, indent=2, allow_nan=True), encoding="utf-8"
    )

    correlations: Dict[str, Dict[str, float]] = {}
    correlation_rows: List[Dict[str, Any]] = []
    for mode in modes:
        if mode == "baseline":
            continue
        values = {
            stage: lowpass_correlation(maps["baseline"][stage], maps[mode][stage])
            for stage in ("branch0_resized", "path1", "depth")
        }
        values["mean_correlation"] = float(np.mean(list(values.values())))
        correlations[mode] = values
        correlation_rows.append({"mode": mode, **values})
    if correlation_rows:
        with (output_dir / "large_scale_similarity.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(correlation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(correlation_rows)

    depth_values = [maps[mode]["depth"] for mode in modes]
    depth_vmin, depth_vmax = _finite_percentiles(depth_values, 2.0, 98.0)
    for mode in modes:
        depth = maps[mode]["depth"]
        np.save(output_dir / "depth_{}.npy".format(mode), depth.astype(np.float32))
        save_colored_map(
            output_dir / "depth_{}.png".format(mode), depth, depth_vmin, depth_vmax, "magma"
        )
    difference_values = []
    for mode in modes:
        if mode == "baseline":
            continue
        difference = np.abs(maps[mode]["depth"] - maps["baseline"]["depth"])
        difference_values.append((mode, difference))
    if difference_values:
        _, difference_max = _finite_percentiles(
            [value for _, value in difference_values], 0.0, 98.0
        )
        for mode, difference in difference_values:
            np.save(
                output_dir / "depth_diff_{}_vs_baseline.npy".format(mode),
                difference.astype(np.float32),
            )
            save_colored_map(
                output_dir / "depth_diff_{}_vs_baseline.png".format(mode),
                difference, 0.0, difference_max, "inferno"
            )

    _save_overview(output_dir, modes, maps, rows)
    _save_profile_plot(
        output_dir / "branch0_mod4_profile.png",
        "branch0 resized mod-4 gradient profile",
        modes,
        {mode: profile_data["{}:branch0_resized".format(mode)] for mode in modes},
    )
    _save_profile_plot(
        output_dir / "depth_mod14_profile.png",
        "final depth mod-14 gradient profile",
        modes,
        {mode: profile_data["{}:depth".format(mode)] for mode in modes},
    )

    diagnosis = diagnose_ablation(
        rows, correlations, modes, args.reduction_threshold
    )
    (output_dir / "replacement_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, allow_nan=True), encoding="utf-8"
    )
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "sequence_id": str(sample["sequence_id"]),
        "dataset_index": dataset_index,
        "clip_start": int(record.clip_start),
        "frame_index": args.frame_index,
        "frame_name": sample["frame_names"][args.frame_index],
        "seed": args.seed,
        "modes": list(modes),
        "model_forward_count": 1,
        "decoder_feature_sets": 1,
        "network_parameters_mutated": False,
        "checkpoint_saved": False,
        "amp_enabled": amp_enabled,
        "student_import_path": str(student_import),
        "dpt_import_path": str(dpt_import),
        "replacement_checks": replacement_checks,
        "model_weight_max_abs_difference_after_ablation": model_weight_max_abs_difference,
        "baseline_replay_vs_original_forward": baseline_replay,
        "shared_depth_display_range": [depth_vmin, depth_vmax],
        "real_fusion_note": (
            "path2 is independent of branch0 in the live DPT; branch0 scratch0 enters only "
            "at refinenet1(path2, scratch0), so path2 is an identical control across modes."
        ),
    }
    (output_dir / "replacement_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Original branch0 weight: shape={} mean={:.9g} std={:.9g}".format(
        replacement_checks["original_weight"]["shape"],
        replacement_checks["original_weight"]["mean"],
        replacement_checks["original_weight"]["std"],
    ))
    print("Phase-tied weight: shape={} mean={:.9g} std={:.9g}".format(
        replacement_checks["phase_tied_weight"]["shape"],
        replacement_checks["phase_tied_weight"]["mean"],
        replacement_checks["phase_tied_weight"]["std"],
    ))
    print("phase_tied_spatial_max_abs_difference={:.9g}".format(
        replacement_checks["phase_tied_weight"]["phase_tied_spatial_max_abs_difference"]
    ))
    for mode, shape in replacement_checks["branch0_output_shapes"].items():
        print("{}_branch0_resized shape={}".format(mode, shape))
    print("baseline_replay_max_abs_difference={:.9g}".format(
        baseline_replay["max_abs_difference"]
    ))
    print("model_weight_max_abs_difference_after_ablation={:.9g}".format(
        model_weight_max_abs_difference
    ))
    print("Diagnosis: {}".format(diagnosis["diagnosis"]))
    print("Wrote branch0 replacement ablation to {}".format(output_dir))
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
        "--branch0-mode",
        choices=("baseline", "phase_tied", "bilinear", "all"),
        default="baseline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/branch0_replacement_ablation"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--reduction-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.clip_index is not None and args.sequence_id is not None:
        parser.error("--clip-index and --sequence-id are mutually exclusive")
    return args


def main() -> None:
    run_ablation(parse_args())


if __name__ == "__main__":
    main()
