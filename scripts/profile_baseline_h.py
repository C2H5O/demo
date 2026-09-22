"""Reproducible CUDA profiling: official dense DA3 versus Baseline-H sparse A."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_crossclip_projection import TRAINED_STUDENT_SOURCE, _evaluation_model
from inference.da3_kv_attention import DA3KVAttention
from inference.da3_runtime_profiler import DA3RuntimeProfiler
from inference.kv_sampling import KVSamplingConfig, WindowFrameMetadata
from inference.student_video import WINDOW
from utils.config import load_config
from utils.merge_student_checkpoint import ensure_merged_student_checkpoint


def metadata():
    slots = tuple(range(WINDOW))
    return WindowFrameMetadata(
        window_id=0, frame_positions=slots, absolute_frame_ids=slots,
        frame_roles=("new",) * WINDOW, is_padding=(False,) * WINDOW,
        first_window=True, absolute_id_source="fixed_profile_tensor",
    )


def event_time(operation):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    end.record()
    torch.cuda.synchronize()
    return output, start.elapsed_time(end)


def output_close(left, right):
    report = {}
    for key in ("depth", "intrinsics", "extrinsics"):
        torch.testing.assert_close(left[key], right[key], rtol=2e-3, atol=2e-3)
        report[key] = {
            "allclose": True,
            "max_abs_diff": float((left[key] - right[key]).abs().max().item()),
        }
    return report


def top_level_stats(samples):
    result = {}
    for name in sorted({key for sample in samples for key in sample}):
        values = [sample[name] for sample in samples if name in sample]
        result[name] = {
            "calls": len(values), "total_seconds": sum(values) / 1000.0,
            "mean_ms": statistics.fmean(values),
        }
    return result


def timed_run(model, images, config, warmup, iterations):
    adapter = DA3KVAttention(model, config, WINDOW)
    times, logical = [], []
    model.enable_cuda_timing(True)
    with adapter:
        def forward():
            adapter.begin_window(metadata(), images)
            with torch.autocast("cuda", enabled=True):
                return model(images, include_global_points=False)

        for _ in range(warmup):
            output = forward()
            torch.cuda.synchronize()
            adapter.finish_window()
            del output
        for _ in range(iterations):
            output, elapsed = event_time(forward)
            logical.append(model.forward_cuda_timings_ms())
            times.append(elapsed)
            adapter.finish_window()
            del output
    model.enable_cuda_timing(False)
    return {
        "window_ms": times,
        "mean_window_ms": statistics.fmean(times),
        "median_window_ms": statistics.median(times),
        "top_level": top_level_stats(logical),
        "adapter_audit": adapter.summary(),
    }


def correctness_run(model, images, config, *, dense):
    def once(profile):
        run_config = (
            replace(config, profile_attention=True, debug=False)
            if profile and not dense else config
        )
        adapter = DA3KVAttention(model, run_config, WINDOW)
        profiler = DA3RuntimeProfiler(model, profile_dense_attention=dense) if profile else None
        with adapter:
            if profiler:
                profiler.__enter__()
            try:
                adapter.begin_window(metadata(), images)
                with torch.autocast("cuda", enabled=True):
                    output = model(images, include_global_points=False)
                torch.cuda.synchronize()
                adapter.finish_window()
                if profiler:
                    profiler.finish_iteration()
                return {key: output[key] for key in ("depth", "intrinsics", "extrinsics")}
            finally:
                if profiler:
                    profiler.__exit__(None, None, None)
    return output_close(once(False), once(True))


def profile_run(model, images, config, *, dense, iterations):
    profiled_config = config if dense else replace(config, profile_attention=True, debug=False)
    adapter = DA3KVAttention(model, profiled_config, WINDOW)
    with DA3RuntimeProfiler(model, profile_dense_attention=dense) as profiler, adapter:
        for _ in range(iterations):
            adapter.begin_window(metadata(), images)
            with torch.autocast("cuda", enabled=True):
                output = model(images, include_global_points=False)
            profiler.finish_iteration()
            adapter.finish_window()
            del output
    return {"runtime": profiler.summary(), "attention": adapter.summary()}


def bottlenecks(profile, limit=10):
    regions = profile["runtime"]["regions"]
    leaves = {
        name: value for name, value in regions.items()
        if not name.endswith((".total", ".attention_total"))
    }
    return [
        {"region": name, **value}
        for name, value in sorted(
            leaves.items(), key=lambda item: item[1]["total_seconds"], reverse=True
        )[:limit]
    ]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-config", type=Path, default=Path("configs/baselines/H_dense_profile.yaml"))
    parser.add_argument("--sparse-config", type=Path, default=Path("configs/baselines/H_A_profile.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-iterations", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_H/profile"))
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.profile_iterations) < 1:
        raise ValueError("Warmup and iteration counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline-H profiling requires CUDA")

    dense_raw, sparse_raw = load_config(args.dense_config), load_config(args.sparse_config)
    dense_cfg = KVSamplingConfig.from_mapping(dense_raw["kv_sampling"])
    sparse_cfg = KVSamplingConfig.from_mapping(sparse_raw["kv_sampling"])
    if dense_cfg.enabled or dense_cfg.profile_attention:
        raise ValueError("Dense config must keep KV sampling and legacy profiling disabled")
    if (not sparse_cfg.enabled or sparse_cfg.method != "layer_stride_kv"
            or not sparse_cfg.profile_attention):
        raise ValueError("Sparse profile config must select profiled layer_stride_kv A")
    formal_sparse = replace(sparse_cfg, profile_attention=False, debug=False)
    checkpoint = args.checkpoint or Path(sparse_raw["vda_evaluation"]["checkpoint"])
    merged = ensure_merged_student_checkpoint(checkpoint, sparse_raw)
    model = _evaluation_model(merged, sparse_raw, torch.device("cuda"), TRAINED_STUDENT_SOURCE)
    torch.manual_seed(20260922)
    images = torch.rand(1, WINDOW, 3, 448, 560, device="cuda")

    correctness = {
        "dense_profiler_off_vs_on": correctness_run(model, images, dense_cfg, dense=True),
        "sparse_A_profiler_off_vs_on": correctness_run(model, images, formal_sparse, dense=False),
    }
    dense_timing = timed_run(model, images, dense_cfg, args.warmup, args.iterations)
    sparse_timing = timed_run(model, images, formal_sparse, args.warmup, args.iterations)
    dense_profile = profile_run(model, images, dense_cfg, dense=True, iterations=args.profile_iterations)
    sparse_profile = profile_run(model, images, formal_sparse, dense=False, iterations=args.profile_iterations)

    expected = [{"layer": layer, "q_tokens": 40992, "kv_tokens": 40992,
                 "calls": args.profile_iterations} for layer in (5, 7, 9, 11)]
    if dense_profile["runtime"]["dense_attention_shapes"] != expected:
        raise RuntimeError("Dense attention token audit did not match four 40992-token blocks")
    common = {
        "checkpoint": str(checkpoint), "inference_checkpoint": str(merged),
        "input_shape": list(images.shape), "amp": True,
        "warmup_windows_excluded": args.warmup, "timed_windows": args.iterations,
        "profiled_windows": args.profile_iterations,
        "correctness": correctness,
        "timing_scope": "fixed GPU tensor; excludes RGB decode, stitching, scoring, and adapter audit",
    }
    comparison = {
        **common,
        "dense_mean_window_ms": dense_timing["mean_window_ms"],
        "sparse_A_mean_window_ms": sparse_timing["mean_window_ms"],
        "speedup": dense_timing["mean_window_ms"] / sparse_timing["mean_window_ms"],
        "dense_top_bottlenecks_no_parent_double_count": bottlenecks(dense_profile),
        "sparse_A_top_bottlenecks_no_parent_double_count": bottlenecks(sparse_profile),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "dense_profile.json": {**common, "timing": dense_timing, "profile": dense_profile},
        "sparse_A_profile.json": {**common, "timing": sparse_timing, "profile": sparse_profile},
        "comparison.json": comparison,
    }
    for name, payload in payloads.items():
        (args.output_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("path       mean_ms    speedup")
    print(f"dense      {dense_timing['mean_window_ms']:8.3f}      1.000x")
    print(f"sparse_A   {sparse_timing['mean_window_ms']:8.3f}      {comparison['speedup']:.3f}x")
    for label in ("dense", "sparse_A"):
        print(f"{label} top leaf bottlenecks (parents excluded):")
        for item in comparison[f"{label}_top_bottlenecks_no_parent_double_count"]:
            print(f"  {item['mean_ms']:8.3f} ms  {item['region']}")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
