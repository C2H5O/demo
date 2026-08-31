"""Evaluate untouched official DA3-Small on raw SCARED datasets 8 and 9."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluate_crossclip_projection import evaluate_official_da3_small


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vggtoda3.yaml")
    parser.add_argument("--protocol", choices=("vda", "endo3r"), default="vda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit-clips", type=int, default=None)
    args = parser.parse_args()
    evaluate_official_da3_small(
        Path(args.config),
        output=args.output,
        limit_clips=args.limit_clips,
        protocol=args.protocol,
    )


if __name__ == "__main__":
    main()
