from __future__ import annotations

import argparse
from pathlib import Path

from visualization.crossclip_projection import export_crossclip_visualization


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a 16-frame cross-clip student prediction or teacher cache"
    )
    parser.add_argument("--config", default="configs/crossclip_teacher_projection.yaml")
    parser.add_argument("--source", choices=("student", "teacher"), default="student")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-depth", type=float, default=None)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--point-stride", type=int, default=None)
    args = parser.parse_args()

    from utils.config import load_config

    config = load_config(args.config)
    visual = dict(config.get("visualization", {}))
    export_crossclip_visualization(
        Path(args.config),
        args.split,
        args.clip_index,
        args.output_dir or Path(str(visual.get("output_dir", "outputs/crossclip_teacher_projection/visualization"))),
        args.source,
        args.checkpoint,
        args.min_depth if args.min_depth is not None else float(visual.get("min_depth", 0.1)),
        args.max_depth if args.max_depth is not None else float(visual.get("max_depth", 10.0)),
        args.point_stride if args.point_stride is not None else int(visual.get("point_stride", 4)),
    )


if __name__ == "__main__":
    main()
