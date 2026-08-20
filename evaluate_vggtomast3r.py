from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluate_vggtomast3r import evaluate as evaluate_endo3r
from evaluation.evaluate_vggtomast3r_vda import evaluate as evaluate_vda
from utils.config import load_config


def select_protocol(config: dict, override: str | None = None) -> str:
    value = override or str(config.get("evaluation", {}).get("protocol", "vda"))
    normalized = value.strip().lower()
    aliases = {"video-depth-anything-depth": "vda"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in ("vda", "endo3r"):
        raise ValueError("Evaluation protocol must be 'vda' or 'endo3r'")
    return normalized


def evaluate(
    config_path: Path,
    checkpoint: Path | None = None,
    split: str | None = None,
    output: Path | None = None,
    limit_pairs: int | None = None,
    protocol: str | None = None,
):
    selected = select_protocol(load_config(config_path), protocol)
    evaluator = evaluate_vda if selected == "vda" else evaluate_endo3r
    return evaluator(config_path, checkpoint, split, output, limit_pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vggtomast3r_v1.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "test"), default=None)
    parser.add_argument(
        "--protocol",
        choices=("vda", "endo3r"),
        default=None,
        help="Override evaluation.protocol (default in V1 config: vda)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit-pairs", type=int, default=None)
    args = parser.parse_args()
    evaluate(
        Path(args.config),
        args.checkpoint,
        args.split,
        args.output,
        args.limit_pairs,
        args.protocol,
    )


if __name__ == "__main__":
    main()
