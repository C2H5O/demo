from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluate_crossclip_projection import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate complete student sequences with VDA depth metrics and TAE"
    )
    parser.add_argument("--config", default="configs/vggtoda3.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "test"), default=None)
    parser.add_argument("--protocol", choices=("vda",), default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit-windows", "--limit-clips", dest="limit_clips", type=int, default=None, help="Debug window budget; --limit-clips is a legacy alias. Omit for full evaluation.")
    args = parser.parse_args()
    evaluate(
        Path(args.config),
        args.checkpoint,
        args.split,
        args.output,
        args.limit_clips,
        args.protocol,
    )


if __name__ == "__main__":
    main()
