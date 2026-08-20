from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluate_vggtomast3r import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vggtomast3r_v1.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "test"), default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit-pairs", type=int, default=None)
    args = parser.parse_args()
    evaluate(Path(args.config), args.checkpoint, args.split, args.output, args.limit_pairs)


if __name__ == "__main__":
    main()
