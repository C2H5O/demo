"""One-window CUDA benchmark: dense DA3-Small versus optimized Baseline-H A."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.evaluate_crossclip_projection import (
    TRAINED_STUDENT_SOURCE,
    _evaluation_model,
)
from inference.da3_kv_attention import DA3KVAttention
from inference.kv_sampling import KVSamplingConfig, WindowFrameMetadata
from inference.student_video import WINDOW
from utils.config import load_config
from utils.merge_student_checkpoint import ensure_merged_student_checkpoint


def _metadata() -> WindowFrameMetadata:
    slots = tuple(range(WINDOW))
    return WindowFrameMetadata(
        window_id=0,
        frame_positions=slots,
        absolute_frame_ids=slots,
        frame_roles=("new",) * WINDOW,
        is_padding=(False,) * WINDOW,
        first_window=True,
        absolute_id_source="fixed_benchmark_tensor",
    )


def _cuda_elapsed(operation) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    end.record()
    torch.cuda.synchronize()
    milliseconds = start.elapsed_time(end)
    del output
    return milliseconds


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/H_A.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("Warmup and timed iteration counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    config = load_config(args.config)
    kv_config = KVSamplingConfig.from_mapping(config["kv_sampling"])
    if kv_config.method != "layer_stride_kv" or kv_config.layer_policy != {
        "block5": "spatial", "block7": "spatial",
        "block9": "temporal", "block11": "temporal",
    }:
        raise ValueError("Benchmark is restricted to Baseline-H experiment A")
    if kv_config.debug or kv_config.profile_attention:
        raise ValueError("Formal H_A config must disable debug and attention profiling")
    checkpoint = args.checkpoint or Path(config["vda_evaluation"]["checkpoint"])
    merged = ensure_merged_student_checkpoint(checkpoint, config)
    device = torch.device("cuda")
    model = _evaluation_model(merged, config, device, TRAINED_STUDENT_SOURCE)
    images = torch.rand(1, WINDOW, 3, 448, 560, device=device)
    metadata = _metadata()

    def dense_forward():
        with torch.autocast(device_type="cuda", enabled=True):
            return model(images, include_global_points=False)

    for _ in range(args.warmup):
        dense_forward()
    torch.cuda.synchronize()
    dense_ms = [_cuda_elapsed(dense_forward) for _ in range(args.iterations)]

    optimized_ms = []
    with DA3KVAttention(model, kv_config, WINDOW) as adapter:
        def optimized_forward():
            adapter.begin_window(metadata, images)
            with torch.autocast(device_type="cuda", enabled=True):
                return model(images, include_global_points=False)

        for _ in range(args.warmup):
            output = optimized_forward()
            torch.cuda.synchronize()
            adapter.finish_window()
            del output
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = optimized_forward()
            end.record()
            torch.cuda.synchronize()
            optimized_ms.append(start.elapsed_time(end))
            adapter.finish_window()
            del output
        audit = adapter.summary()["kv_selection_examples"][0]["layer_stride_kv"]

    profile_config = replace(kv_config, profile_attention=True, debug=False)
    with DA3KVAttention(model, profile_config, WINDOW) as profile_adapter:
        profile_adapter.begin_window(metadata, images)
        with torch.autocast(device_type="cuda", enabled=True):
            profile_output = model(images, include_global_points=False)
        torch.cuda.synchronize()
        profile_adapter.finish_window()
        stage_profile = profile_adapter.summary()["layer_stride_profile_seconds"]
        del profile_output

    dense_mean = sum(dense_ms) / len(dense_ms)
    optimized_mean = sum(optimized_ms) / len(optimized_ms)
    print(json.dumps({
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "inference_checkpoint": str(merged),
        "input_shape": list(images.shape),
        "amp": True,
        "warmup_windows_excluded": args.warmup,
        "timed_windows": args.iterations,
        "dense_mean_window_ms": dense_mean,
        "optimized_a_mean_window_ms": optimized_mean,
        "speedup": dense_mean / optimized_mean,
        "projection_and_sdpa_tokens": audit,
        "single_window_stage_profile_seconds": stage_profile,
        "timing_scope": "fixed GPU tensor; begin_window plus model forward; excludes RGB decode, stitching, finish_window audit, and scoring",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
