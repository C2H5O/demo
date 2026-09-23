"""Hamlyn zero-shot inference/evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.hamlyn.config import DEFAULT_CONFIG
from evaluation.hamlyn.pipeline import collect, preflight, run_method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--method", choices=("ours", "da3", "endodav", "endo3r"))
    action.add_argument("--collect-summary", action="store_true")
    parser.add_argument("--stage", choices=("infer", "evaluate", "all"), default="all")
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.preflight:
        preflight(args.config)
    elif args.collect_summary:
        collect(args.config)
    else:
        run_method(
            args.method,
            stage=args.stage,
            force_inference=args.force_inference,
            config_path=args.config,
        )


if __name__ == "__main__":
    main()
