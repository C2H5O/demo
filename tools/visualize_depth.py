"""Visualize student depth and confidence for one SCARED temporal clip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/student_distillation.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--sequence-id",
        default=None,
        help="Example: dataset_8/keyframe_0; default selects the first sequence",
    )
    parser.add_argument("--clip-offset", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/student_distill3r_256x320_1epoch/visualization/depth"),
    )
    parser.add_argument("--min-depth", type=float, default=1e-4)
    parser.add_argument("--max-depth", type=float, default=100.0)
    args = parser.parse_args()

    from visualization.scared_student import export_depth_visualization

    export_depth_visualization(
        Path(args.config),
        args.checkpoint,
        args.split,
        args.sequence_id,
        args.clip_offset,
        args.output,
        args.min_depth,
        args.max_depth,
    )


if __name__ == "__main__":
    main()
