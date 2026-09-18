"""Compare the resolved Baseline C/D/J training contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config  # noqa: E402


def _shared_contract(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    return {
        "dataset": {
            key: config["dataset"][key]
            for key in ("clip_length", "sample_stride", "window_stride")
        },
        "teacher": {
            key: config["teacher"][key]
            for key in (
                "variant", "pretrained_checkpoint", "mode", "use_cache", "frozen",
                "freeze_backbone", "freeze_heads", "input_height", "input_width",
                "amp", "amp_dtype",
            )
        },
        "student": config["student"],
        "dataloader": config["dataloader"],
        "optimizer": {
            key: training[key]
            for key in (
                "learning_rate", "lora_learning_rate", "min_learning_rate",
                "weight_decay", "warmup_fraction", "gradient_accumulation_steps",
                "gradient_clip_norm", "amp", "amp_dtype",
            )
        },
    }


def audit(c_path: Path, d_path: Path, j_path: Path) -> dict[str, Any]:
    configs = {name: load_config(path) for name, path in (
        ("C", c_path), ("D", d_path), ("J", j_path)
    )}
    shared = {name: _shared_contract(config) for name, config in configs.items()}
    if shared["C"] != shared["J"] or shared["D"] != shared["J"]:
        raise RuntimeError("C/D shared training contract differs from J")

    c_attention = configs["C"]["attention_distill"]
    j_attention = configs["J"]["attention_distill"]
    mathematical_fields = (
        "teacher_source", "teacher_output_dtype", "teacher_layers", "student_layers",
        "attention_type", "spatial_alignment", "common_grid", "head_aggregation",
        "divergence", "temperature_teacher", "temperature_student", "weight",
        "frame_offsets", "query_chunk_size", "eps", "pair_chunk_size",
        "teacher_probability_outside_checkpoint",
    )
    if any(c_attention[key] != j_attention[key] for key in mathematical_fields):
        raise RuntimeError("C attention contract differs from J")

    expected = {
        "C": (True, "legacy"),
        "D": (False, "angular_soft_margin"),
        "J": (True, "angular_soft_margin"),
    }
    rows = {}
    for name, config in configs.items():
        enabled, highlight = expected[name]
        if bool(config["attention_distill"]["enabled"]) is not enabled:
            raise RuntimeError("{} attention ablation is incorrect".format(name))
        if config["loss"]["highlight_mode"] != highlight:
            raise RuntimeError("{} highlight ablation is incorrect".format(name))
        rows[name] = {
            "teacher_supervision": True,
            "attention_distillation": enabled,
            "angular_highlight": highlight == "angular_soft_margin",
            "teacher_input": "512x640",
            "student_input": "448x560",
            "temporal": "32/2/8",
            "attention_implementation": "J optimized" if enabled else "N/A",
        }
    return {"status": "passed", "baselines": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-config", type=Path, required=True)
    parser.add_argument("--d-config", type=Path, required=True)
    parser.add_argument("--j-config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.c_config, args.d_config, args.j_config), indent=2))


if __name__ == "__main__":
    main()
