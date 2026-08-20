from __future__ import annotations

import argparse
from pathlib import Path

from visualization.vggtomast3r_pair import export_pair_visualization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vggtomast3r_v1.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vggtomast3r_v1/visualization"))
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--point-stride", type=int, default=4)
    args = parser.parse_args()
    print(export_pair_visualization(
        Path(args.config), args.checkpoint, args.split, args.pair_index,
        args.output_dir, args.min_depth, args.max_depth, args.point_stride,
    ))


if __name__ == "__main__":
    main()
