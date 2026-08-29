from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluate_crossclip_projection import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the 16-frame cross-clip student (VDA by default)"
    )
    parser.add_argument("--config", default="configs/vggtoda3.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "test"), default=None)
    parser.add_argument("--protocol", choices=("vda", "endo3r"), default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit-clips", type=int, default=None)
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
