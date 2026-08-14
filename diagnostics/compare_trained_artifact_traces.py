"""Compare baseline and retrained-head traces with shared depth rendering limits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.trace_distill3r_artifacts_stage2 import phase_statistics
from utils.config import ensure_dir


STAGES = {
    "branch0_output": ("10_dpt_act_0/norm.npy", 4),
    "scratch0": ("20_scratch_0/norm.npy", 4),
    "path1": ("33_path_1/norm.npy", 8),
    "depth": ("50_depth/depth.npy", 14),
}


def _metadata(trace_dir: Path) -> Mapping[str, Any]:
    path = trace_dir / "trace_metadata.json"
    if not path.is_file():
        raise FileNotFoundError("Missing trace metadata: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def validate_matching_sample(baseline_dir: Path, experiment_dir: Path) -> None:
    baseline = _metadata(baseline_dir)
    experiment = _metadata(experiment_dir)
    fields = ("split", "sequence_id", "clip_start", "frame_index", "frame_name", "seed")
    differences = {
        field: (baseline.get(field), experiment.get(field))
        for field in fields
        if baseline.get(field) != experiment.get(field)
    }
    if differences:
        raise RuntimeError("Artifact traces are not from the same sample: {}".format(differences))


def _shared_depth_images(
    baseline: np.ndarray, experiment: np.ndarray, output_dir: Path
) -> Dict[str, Any]:
    finite = np.concatenate((baseline[np.isfinite(baseline)], experiment[np.isfinite(experiment)]))
    if not finite.size:
        raise RuntimeError("Both depth maps contain no finite values")
    vmin, vmax = np.percentile(finite, (2.0, 98.0))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    for name, depth in (("original_dpt", baseline), ("bilinear_head", experiment)):
        np.save(output_dir / "depth_{}.npy".format(name), depth.astype(np.float32))
        plt.imsave(
            output_dir / "depth_{}.png".format(name),
            depth,
            cmap="magma",
            vmin=float(vmin),
            vmax=float(vmax),
        )
    return {"vmin": float(vmin), "vmax": float(vmax), "percentiles": [2.0, 98.0]}


def compare_traces(baseline_dir: Path, experiment_dir: Path, output_dir: Path) -> Path:
    baseline_dir = baseline_dir.resolve()
    experiment_dir = experiment_dir.resolve()
    validate_matching_sample(baseline_dir, experiment_dir)
    output_dir = ensure_dir(output_dir)
    rows = []
    loaded: Dict[str, Dict[str, np.ndarray]] = {"original_dpt": {}, "bilinear_head": {}}
    for stage, (relative_path, period) in STAGES.items():
        for variant, root in (("original_dpt", baseline_dir), ("bilinear_head", experiment_dir)):
            path = root / relative_path
            if not path.is_file():
                raise FileNotFoundError("Missing artifact map: {}".format(path))
            spatial_map = np.load(path)
            loaded[variant][stage] = spatial_map
            stats = phase_statistics(spatial_map, period)
            rows.append(
                {
                    "variant": variant,
                    "stage": stage,
                    "H": int(spatial_map.shape[0]),
                    "W": int(spatial_map.shape[1]),
                    "period": period,
                    "phase_ratio_x": stats["phase_ratio_x"],
                    "phase_ratio_y": stats["phase_ratio_y"],
                    "phase_cv_x": stats["phase_cv_x"],
                    "phase_cv_y": stats["phase_cv_y"],
                }
            )

    with (output_dir / "trained_artifact_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    display = _shared_depth_images(
        loaded["original_dpt"]["depth"], loaded["bilinear_head"]["depth"], output_dir
    )
    metadata = {
        "baseline_trace": str(baseline_dir),
        "bilinear_head_trace": str(experiment_dir),
        "same_sample_verified": True,
        "depth_display": {"colormap": "magma", **display},
    }
    (output_dir / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Wrote trained-head artifact comparison to {}".format(output_dir))
    print("Shared depth display range: vmin={} vmax={}".format(display["vmin"], display["vmax"]))
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--experiment-trace", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("diagnostics/trained_head_artifact_comparison")
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    compare_traces(args.baseline_trace, args.experiment_trace, args.output_dir)


if __name__ == "__main__":
    main()
