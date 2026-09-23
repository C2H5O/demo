"""High-level Hamlyn method orchestration with lazy environment-specific imports."""

from __future__ import annotations

from pathlib import Path

from evaluation.hamlyn.common import evaluate_method
from evaluation.hamlyn.config import DEFAULT_CONFIG, load_runtime_config
from evaluation.hamlyn.constants import METHOD_ORDER
from evaluation.hamlyn.data import discover_sequences
from evaluation.hamlyn.preflight import run_preflight
from evaluation.hamlyn.summary import collect_summary


STAGES = ("infer", "evaluate", "all")


def preflight(config_path: Path = DEFAULT_CONFIG):
    runtime = load_runtime_config(config_path)
    records = discover_sequences(runtime.hamlyn_root)
    return run_preflight(runtime, records)


def run_method(
    method: str,
    stage: str = "all",
    force_inference: bool = False,
    config_path: Path = DEFAULT_CONFIG,
):
    if method not in METHOD_ORDER:
        raise ValueError("method must be one of {}".format(METHOD_ORDER))
    if stage not in STAGES:
        raise ValueError("stage must be one of {}".format(STAGES))
    runtime = load_runtime_config(config_path)
    records = discover_sequences(runtime.hamlyn_root)
    if stage in ("infer", "all"):
        if method in ("ours", "da3"):
            from evaluation.hamlyn.ours import infer_da3_sequences

            infer_da3_sequences(method, records, runtime, force=force_inference)
        elif method == "endodav":
            from evaluation.hamlyn.endodav import infer_endodav_sequences

            infer_endodav_sequences(records, runtime, force=force_inference)
        else:
            from evaluation.hamlyn.endo3r import infer_endo3r_sequences

            infer_endo3r_sequences(records, runtime, force=force_inference)
    if stage in ("evaluate", "all"):
        return evaluate_method(method, records, runtime.output_root)
    return None


def collect(config_path: Path = DEFAULT_CONFIG):
    runtime = load_runtime_config(config_path)
    return collect_summary(runtime.output_root)
