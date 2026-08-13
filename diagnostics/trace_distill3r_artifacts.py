"""Trace where token-grid artifacts first appear in one Distill3R forward.

This script is intentionally inference-only.  It registers temporary forward
hooks, calls the unchanged student exactly once, removes every hook, and then
writes native-resolution feature summaries and periodicity measurements.
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.scared_clip_dataset import make_scared_rgb_dataset
from utils.config import ensure_dir, load_config


TOKEN_GRID = (32, 40)
STAGE_DIRECTORIES = OrderedDict(
    [
        ("f0_encoder", "00_f0_encoder"),
        ("f3_decoder", "01_f3_decoder"),
        ("f4_decoder", "02_f4_decoder"),
        ("f6_decoder", "03_f6_decoder"),
        ("dpt_act_0", "10_dpt_act_0"),
        ("dpt_act_1", "11_dpt_act_1"),
        ("dpt_act_2", "12_dpt_act_2"),
        ("dpt_act_3", "13_dpt_act_3"),
        ("scratch_0", "20_scratch_0"),
        ("scratch_1", "21_scratch_1"),
        ("scratch_2", "22_scratch_2"),
        ("scratch_3", "23_scratch_3"),
        ("path_4", "30_path_4"),
        ("path_3", "31_path_3"),
        ("path_2", "32_path_2"),
        ("path_1", "33_path_1"),
        ("head_conv1", "40_head_conv1"),
        ("head_after_resize", "41_head_after_resize"),
        ("head_conv2", "42_head_conv2"),
        ("raw_output", "43_raw_output"),
        ("depth", "50_depth"),
    ]
)
OVERVIEW_STAGES = (
    "f0_encoder",
    "f3_decoder",
    "f6_decoder",
    "dpt_act_0",
    "scratch_0",
    "path_4",
    "path_3",
    "path_2",
    "path_1",
    "head_conv1",
    "depth",
)


def _cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def _require_import_below(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            "{} was imported from {}, outside the pinned source {}".format(label, path, root)
        ) from error


def _feature_maps(feature: torch.Tensor) -> Dict[str, np.ndarray]:
    """Return raw spatial summaries for a native-resolution [C,H,W] tensor."""

    if feature.ndim != 3:
        raise ValueError("Expected feature [C,H,W], got {}".format(tuple(feature.shape)))
    value = feature.float()
    return {
        "mean": value.mean(dim=0).numpy(),
        "absmean": value.abs().mean(dim=0).numpy(),
        "norm": torch.linalg.vector_norm(value, dim=0).numpy(),
    }


def boundary_metrics(
    spatial_map: np.ndarray,
    token_grid: Tuple[int, int] = TOKEN_GRID,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compare raw gradients at token-cell boundaries with interior gradients."""

    value = np.asarray(spatial_map, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("spatial_map must be two-dimensional")
    height, width = value.shape
    grid_h, grid_w = token_grid
    if height % grid_h or width % grid_w:
        return {
            "cell_h": math.nan,
            "cell_w": math.nan,
            "boundary_x_mean": math.nan,
            "interior_x_mean": math.nan,
            "boundary_y_mean": math.nan,
            "interior_y_mean": math.nan,
            "boundary_ratio_x": math.nan,
            "boundary_ratio_y": math.nan,
        }

    cell_h, cell_w = height // grid_h, width // grid_w
    gradient_x = np.abs(np.diff(value, axis=1))
    gradient_y = np.abs(np.diff(value, axis=0))
    boundary_x = np.arange(1, width) % cell_w == 0
    boundary_y = np.arange(1, height) % cell_h == 0

    def means(gradient: np.ndarray, mask: np.ndarray, axis: int) -> Tuple[float, float, float]:
        boundary_values = np.compress(mask, gradient, axis=axis).reshape(-1)
        interior_values = np.compress(~mask, gradient, axis=axis).reshape(-1)
        boundary_mean = float(boundary_values.mean()) if boundary_values.size else math.nan
        interior_mean = float(interior_values.mean()) if interior_values.size else math.nan
        ratio = (
            boundary_mean / (interior_mean + eps)
            if np.isfinite(boundary_mean) and np.isfinite(interior_mean)
            else math.nan
        )
        return boundary_mean, interior_mean, ratio

    boundary_x_mean, interior_x_mean, ratio_x = means(gradient_x, boundary_x, axis=1)
    boundary_y_mean, interior_y_mean, ratio_y = means(gradient_y, boundary_y, axis=0)
    return {
        "cell_h": int(cell_h),
        "cell_w": int(cell_w),
        "boundary_x_mean": boundary_x_mean,
        "interior_x_mean": interior_x_mean,
        "boundary_y_mean": boundary_y_mean,
        "interior_y_mean": interior_y_mean,
        "boundary_ratio_x": ratio_x,
        "boundary_ratio_y": ratio_y,
    }


def modulo_gradient_profile(spatial_map: np.ndarray, cell_h: int, cell_w: int) -> Dict[str, Any]:
    """Average raw x/y gradients by their phase inside one token-grid cell."""

    value = np.asarray(spatial_map, dtype=np.float64)
    height, width = value.shape
    gradient_x = np.abs(np.diff(value, axis=1))
    gradient_y = np.abs(np.diff(value, axis=0))
    x_coordinates = np.arange(1, width)
    y_coordinates = np.arange(1, height)
    x_profile = [
        float(gradient_x[:, x_coordinates % cell_w == phase].mean())
        if np.any(x_coordinates % cell_w == phase)
        else math.nan
        for phase in range(cell_w)
    ]
    y_profile = [
        float(gradient_y[y_coordinates % cell_h == phase, :].mean())
        if np.any(y_coordinates % cell_h == phase)
        else math.nan
        for phase in range(cell_h)
    ]
    return {
        "height": int(height),
        "width": int(width),
        "modulo_x": int(cell_w),
        "modulo_y": int(cell_h),
        "mod_x": {"phase{}".format(index): value for index, value in enumerate(x_profile)},
        "mod_y": {"phase{}".format(index): value for index, value in enumerate(y_profile)},
    }


def _normalized_rgba(spatial_map: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    finite = np.isfinite(spatial_map)
    normalized = np.zeros(spatial_map.shape, dtype=np.float32)
    if np.any(finite):
        low, high = np.percentile(spatial_map[finite], (2.0, 98.0))
        normalized[finite] = np.clip(
            (spatial_map[finite] - low) / max(float(high - low), 1e-12), 0.0, 1.0
        )
    return np.round(plt.get_cmap(cmap_name)(normalized) * 255.0).astype(np.uint8)


def save_nearest_png(path: Path, spatial_map: np.ndarray, cmap_name: str = "viridis") -> None:
    """Save an enlarged PNG using only nearest-neighbour pixel replication."""

    rgba = _normalized_rgba(spatial_map, cmap_name)
    height, width = spatial_map.shape
    scale = max(1, min(16, int(math.ceil(640.0 / max(height, width)))))
    image = Image.fromarray(rgba, mode="RGBA")
    if scale > 1:
        image = image.resize((width * scale, height * scale), resample=Image.Resampling.NEAREST)
    image.save(path)


def _select_clip(dataset: Any, sequence_id: Optional[str], clip_offset: int, clip_index: Optional[int]) -> Tuple[int, Any]:
    if clip_index is not None:
        if not 0 <= clip_index < len(dataset):
            raise IndexError("clip_index={} is outside [0,{})".format(clip_index, len(dataset)))
        return clip_index, dataset.clips[clip_index]
    candidates = [
        index
        for index, record in enumerate(dataset.clips)
        if sequence_id is None or str(record.sequence["sequence_id"]) == sequence_id
    ]
    if not candidates:
        available = sorted({str(record.sequence["sequence_id"]) for record in dataset.clips})
        raise ValueError("No clips for sequence {!r}. Available: {}".format(sequence_id, available))
    if not 0 <= clip_offset < len(candidates):
        raise IndexError("clip_offset={} is outside [0,{})".format(clip_offset, len(candidates)))
    index = candidates[clip_offset]
    return index, dataset.clips[index]


class SingleForwardTrace:
    """Temporary hook set for the unchanged local Distill3R inference path."""

    def __init__(self, model: torch.nn.Module, frame_index: int, frame_count: int, grid: Tuple[int, int]) -> None:
        self.model = model
        self.frame_index = frame_index
        self.frame_count = frame_count
        self.grid = grid
        self.handles: List[Any] = []
        self.calls: MutableMapping[str, List[torch.Tensor]] = OrderedDict()
        self.decoder_features: MutableMapping[str, torch.Tensor] = OrderedDict()

    def _capture_batch(self, name: str):
        def callback(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
            if not torch.is_tensor(output):
                raise TypeError("{} hook expected a tensor".format(name))
            self.calls.setdefault(name, []).append(_cpu_float(output))

        return callback

    def _capture_decoder(self, _module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Sequence[torch.Tensor]) -> None:
        if len(output) <= 6:
            raise RuntimeError("Fast3R decoder returned only {} layers".format(len(output)))
        grid_h, grid_w = self.grid
        patches = grid_h * grid_w
        start, end = self.frame_index * patches, (self.frame_index + 1) * patches
        for index, name in ((0, "f0_encoder"), (3, "f3_decoder"), (4, "f4_decoder"), (6, "f6_decoder")):
            layer = output[index]
            if layer.ndim != 3 or layer.shape[0] != 1 or layer.shape[1] != self.frame_count * patches:
                raise RuntimeError(
                    "Decoder output {} has unexpected shape {} for {} frames and grid {}".format(
                        index, tuple(layer.shape), self.frame_count, self.grid
                    )
                )
            selected = layer[0, start:end].reshape(grid_h, grid_w, layer.shape[-1])
            self.decoder_features[name] = _cpu_float(selected.permute(2, 0, 1))

    def register(self) -> None:
        student = self.model.student
        dpt = student.downstream_head_local.dpt
        self.handles.append(student.decoder.register_forward_hook(self._capture_decoder))
        modules = OrderedDict()
        for index, module in enumerate(dpt.act_postprocess):
            modules["dpt_act_{}".format(index)] = module
        for index, module in enumerate(dpt.scratch.layer_rn):
            modules["scratch_{}".format(index)] = module
        modules.update(
            [
                ("path_4", dpt.scratch.refinenet4),
                ("path_3", dpt.scratch.refinenet3),
                ("path_2", dpt.scratch.refinenet2),
                ("path_1", dpt.scratch.refinenet1),
                ("head_conv1", dpt.head[0]),
                ("head_after_resize", dpt.head[1]),
                ("head_conv2", dpt.head[2]),
                ("raw_output", dpt.head[4]),
            ]
        )
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(self._capture_batch(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def selected_features(self) -> "OrderedDict[str, torch.Tensor]":
        selected: "OrderedDict[str, torch.Tensor]" = OrderedDict(self.decoder_features)
        for name in STAGE_DIRECTORIES:
            if name in selected or name == "depth":
                continue
            calls = self.calls.get(name, [])
            if not calls:
                raise RuntimeError("Hook {} did not run".format(name))
            batch = torch.cat(calls, dim=0)
            if batch.shape[0] != self.frame_count:
                raise RuntimeError(
                    "Hook {} captured {} views, expected {}".format(name, batch.shape[0], self.frame_count)
                )
            selected[name] = batch[self.frame_index]
        return selected


def _save_periodicity_plot(path: Path, stage: str, profile: Mapping[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, key, title in ((axes[0], "mod_x", "x-gradient phase"), (axes[1], "mod_y", "y-gradient phase")):
        values = list(profile[key].values())
        axis.plot(range(len(values)), values, marker="o")
        axis.set_xticks(range(len(values)))
        axis.set_title(title)
        axis.set_xlabel("phase (0 = token boundary)")
        axis.set_ylabel("mean |gradient|")
        axis.grid(alpha=0.25)
    figure.suptitle(stage)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_overview(
    output_dir: Path,
    norm_maps: Mapping[str, np.ndarray],
    rows_by_stage: Mapping[str, Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(18, 12))
    for axis, stage in zip(axes.flat, OVERVIEW_STAGES):
        value = norm_maps[stage]
        row = rows_by_stage[stage]
        axis.imshow(value, cmap="viridis", interpolation="nearest")
        axis.set_title(
            "{} {}\nratio x={} y={}".format(
                stage,
                tuple(value.shape),
                _format_ratio(row["boundary_ratio_x"]),
                _format_ratio(row["boundary_ratio_y"]),
            ),
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes.flat[len(OVERVIEW_STAGES):]:
        axis.axis("off")
    figure.suptitle("Distill3R single-forward local-head artifact trace")
    figure.tight_layout()
    figure.savefig(output_dir / "artifact_trace_overview.png", dpi=160)
    plt.close(figure)


def _format_ratio(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "{:.3f}".format(numeric) if np.isfinite(numeric) else "n/a"


def _diagnosis(rows: Sequence[Mapping[str, Any]], threshold: float) -> Dict[str, Any]:
    first = None
    for row in rows:
        ratios = [float(row["boundary_ratio_x"]), float(row["boundary_ratio_y"])]
        finite = [value for value in ratios if np.isfinite(value)]
        if finite and max(finite) >= threshold:
            first = str(row["stage"])
            break
    if first is None:
        case = "inconclusive"
        explanation = "No eligible stage crossed the configured boundary-ratio threshold."
    elif first.startswith("f"):
        case = "A"
        explanation = "The first strong grid signal is in encoder/Fast3R decoder features."
    elif first.startswith("dpt_act") or first.startswith("scratch"):
        case = "B"
        explanation = "The first strong grid signal is in the DPT artificial pyramid or scratch projection."
    elif first.startswith("path"):
        case = "C"
        explanation = "The first strong grid signal appears during RefineNet coarse-to-fine fusion."
    else:
        case = "D"
        explanation = "The first strong grid signal appears in the regression/postprocess output path."
    return {
        "threshold": threshold,
        "first_stage_above_threshold": first,
        "case": case,
        "explanation": explanation,
        "native_token_grid_note": (
            "F0/F3/F4/F6 are 32x40 (cell size 1), so every adjacent gradient is a token "
            "boundary and there is no interior baseline; their ratios are intentionally NaN. "
            "Inspect their nearest-neighbour norm maps and compare F0 versus F3/F6."
        ),
    }


def run_trace(args: argparse.Namespace) -> Path:
    # Delay the visualization/checkpoint dependency (and its OpenCV import)
    # until an actual trace is requested. Metric helpers remain lightweight.
    from visualization.scared_student import load_student
    from models.student.distill3r_wrapper import DISTILL3R_FAST3R_ROOT, DISTILL3R_ROOT

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
    dataset_index, record = _select_clip(dataset, args.sequence_id, args.clip_offset, args.clip_index)
    sample = dataset[dataset_index]
    images = sample["images"].unsqueeze(0).to(device)
    _, frame_count, _, height, width = images.shape
    if (height, width) != (448, 560):
        raise RuntimeError("Artifact trace requires the actual 448x560 input, got {}x{}".format(height, width))
    if not 0 <= args.frame_index < frame_count:
        raise IndexError("frame_index={} is outside [0,{})".format(args.frame_index, frame_count))

    model = load_student(args.checkpoint, config, device)
    model.eval()
    patch_size = int(model.student.patch_size)
    grid = (height // patch_size, width // patch_size)
    if grid != TOKEN_GRID:
        raise RuntimeError("Expected the DUNE token grid {}, got {}".format(TOKEN_GRID, grid))

    trace = SingleForwardTrace(model, args.frame_index, frame_count, grid)
    trace.register()
    amp_enabled = bool(config.get("evaluation", {}).get("amp", True)) and device.type == "cuda" and not args.no_amp
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            prediction = model(images)  # The only model forward in this program.
    finally:
        trace.remove()

    features = trace.selected_features()
    pts3d_local = _cpu_float(prediction["xyz_local"][0, args.frame_index])
    depth = pts3d_local[..., 2].numpy()
    features["depth"] = torch.from_numpy(depth)

    output_dir = ensure_dir(args.output_dir)
    norm_maps: "OrderedDict[str, np.ndarray]" = OrderedDict()
    metric_rows: List[Dict[str, Any]] = []
    periodicity: "OrderedDict[str, Any]" = OrderedDict()

    for stage, directory_name in STAGE_DIRECTORIES.items():
        stage_dir = ensure_dir(output_dir / directory_name)
        feature = features[stage]
        if feature.ndim == 2:
            maps = {"mean": feature.numpy(), "absmean": feature.abs().numpy(), "norm": feature.abs().numpy()}
        else:
            maps = _feature_maps(feature)
        for map_name, spatial_map in maps.items():
            np.save(stage_dir / "{}.npy".format(map_name), spatial_map.astype(np.float32))
            save_nearest_png(stage_dir / "{}.png".format(map_name), spatial_map, "magma" if stage == "depth" else "viridis")
        if args.save_full_features and feature.ndim == 3:
            np.save(stage_dir / "feature.npy", feature.numpy().astype(np.float32))
        if stage == "depth":
            np.save(stage_dir / "depth.npy", depth.astype(np.float32))
            np.save(stage_dir / "pts3d_local.npy", pts3d_local.numpy().astype(np.float32))
            save_nearest_png(stage_dir / "depth.png", depth, "magma")

        norm_map = maps["norm"]
        norm_maps[stage] = norm_map
        metrics = boundary_metrics(norm_map)
        row = {
            "stage": stage,
            "channels": int(feature.shape[0]) if feature.ndim == 3 else 1,
            "H": int(norm_map.shape[0]),
            "W": int(norm_map.shape[1]),
            **metrics,
        }
        metric_rows.append(row)
        if np.isfinite(float(metrics["cell_h"])) and np.isfinite(float(metrics["cell_w"])):
            cell_h, cell_w = int(metrics["cell_h"]), int(metrics["cell_w"])
            profile = modulo_gradient_profile(norm_map, cell_h, cell_w)
            periodicity[stage] = profile
            _save_periodicity_plot(stage_dir / "periodicity.png", stage, profile)

        print("{:<20} {}".format(stage, list(feature.shape)))

    fieldnames = list(metric_rows[0].keys())
    with (output_dir / "artifact_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    (output_dir / "periodicity_metrics.json").write_text(
        json.dumps(periodicity, indent=2, allow_nan=True), encoding="utf-8"
    )
    rows_by_stage = {str(row["stage"]): row for row in metric_rows}
    diagnosis = _diagnosis(metric_rows, args.artifact_ratio_threshold)
    (output_dir / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    _write_overview(output_dir, norm_maps, rows_by_stage)

    student_module = inspect.getmodule(model.student.__class__)
    dpt_module = inspect.getmodule(model.student.downstream_head_local.dpt.__class__)
    if student_module is None or dpt_module is None:
        raise RuntimeError("Could not resolve the live Distill3R/Fast3R import modules")
    resolved_student_import = Path(inspect.getfile(student_module)).resolve()
    resolved_dpt_import = Path(inspect.getfile(dpt_module)).resolve()
    _require_import_below(resolved_student_import, DISTILL3R_ROOT.resolve(), "Distill3R student")
    _require_import_below(resolved_dpt_import, DISTILL3R_FAST3R_ROOT.resolve(), "Fast3R DPT")
    student_import_path = str(resolved_student_import)
    dpt_import_path = str(resolved_dpt_import)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "split": args.split,
        "sequence_id": str(sample["sequence_id"]),
        "dataset_index": dataset_index,
        "clip_offset": args.clip_offset,
        "clip_start": int(record.clip_start),
        "frame_index": args.frame_index,
        "frame_name": sample["frame_names"][args.frame_index],
        "seed": args.seed,
        "input_shape": list(images.shape),
        "token_grid": list(grid),
        "forward_count": 1,
        "model_training": model.training,
        "amp_enabled": amp_enabled,
        "save_full_features": args.save_full_features,
        "student_import_path": student_import_path,
        "dpt_import_path": dpt_import_path,
        "network_unchanged": True,
        "actual_output_resolution": list(depth.shape),
        "note": "Hooks were temporary and all tensors came from one unchanged local-head forward.",
    }
    (output_dir / "trace_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Imported Distill3R student from {}".format(student_import_path))
    print("Imported Fast3R DPT from {}".format(dpt_import_path))
    print("Wrote single-forward artifact trace to {}".format(output_dir))
    print("Automatic diagnosis: case={} first_stage={}".format(diagnosis["case"], diagnosis["first_stage_above_threshold"]))
    return output_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/student_distillation.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--sequence-id", "--sequence", dest="sequence_id", default=None)
    parser.add_argument("--clip-offset", type=int, default=0, help="Offset within the selected sequence")
    parser.add_argument("--clip-index", type=int, default=None, help="Absolute dataset clip index")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/artifact_trace"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Defaults to config.device")
    parser.add_argument("--no-amp", action="store_true", help="Disable evaluation autocast")
    parser.add_argument("--save-full-features", action="store_true", help="Also save large native [C,H,W] tensors")
    parser.add_argument("--artifact-ratio-threshold", type=float, default=1.25)
    args = parser.parse_args(argv)
    if args.clip_index is not None and args.sequence_id is not None:
        parser.error("--clip-index and --sequence-id are mutually exclusive")
    return args


def main() -> None:
    run_trace(parse_args())


if __name__ == "__main__":
    main()
