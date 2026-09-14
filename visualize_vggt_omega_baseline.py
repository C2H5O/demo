"""Visualize one complete SCARED test sequence with online VGGT-Omega."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_config
from visualization.vggt_omega_video import export_vggt_omega_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines/T.yaml")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--min-depth", type=float, default=None)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--point-stride", type=int, default=None)
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=None,
        help="Debug window budget; omit to export the complete sequence.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    visual = dict(config.get("vggt_omega_baseline_visualization", {}))
    export_vggt_omega_video(
        Path(args.config),
        sequence_index=args.sequence_index,
        output_root=args.output_dir
        or Path(str(visual.get("output_dir", "outputs/baseline_T/visualization"))),
        checkpoint_path=args.checkpoint,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        point_stride=args.point_stride,
        max_windows=args.limit_windows,
    )


if __name__ == "__main__":
    main()
