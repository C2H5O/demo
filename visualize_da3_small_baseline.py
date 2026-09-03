"""Visualize untouched official DA3-Small on raw SCARED datasets 8 and 9."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluate_crossclip_projection import OFFICIAL_DA3_SMALL_SOURCE
from utils.config import load_config
from visualization.crossclip_projection import export_crossclip_visualization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vggtoda3.yaml")
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-depth", type=float, default=None)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--point-stride", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    visual = dict(config.get("da3_small_baseline_visualization", {}))
    export_crossclip_visualization(
        Path(args.config),
        "test",
        args.clip_index,
        args.output_dir
        or Path(str(visual.get("output_dir", "outputs/da3_small_baseline/visualization"))),
        OFFICIAL_DA3_SMALL_SOURCE,
        None,
        args.min_depth if args.min_depth is not None else float(visual.get("min_depth", 0.1)),
        args.max_depth if args.max_depth is not None else float(visual.get("max_depth", 10.0)),
        args.point_stride
        if args.point_stride is not None
        else int(visual.get("point_stride", 4)),
    )


if __name__ == "__main__":
    main()
