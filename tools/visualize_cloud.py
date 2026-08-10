"""Visualize one SCARED student clip as an RGB point cloud."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


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
        default=Path("outputs/student_distill3r_448x560/visualization/cloud"),
    )
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--min-depth", type=float, default=1e-4)
    parser.add_argument("--max-depth", type=float, default=100.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional [0,1] threshold; disabled by default to retain all valid points",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Open the exported point cloud in an interactive Viser server",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.15)
    parser.add_argument("--max-viewer-points", type=int, default=500000)
    args = parser.parse_args()

    from visualization.scared_student import export_cloud_visualization

    output = export_cloud_visualization(
        Path(args.config),
        args.checkpoint,
        args.split,
        args.sequence_id,
        args.clip_offset,
        args.output,
        args.point_stride,
        args.min_depth,
        args.max_depth,
        args.confidence_threshold,
    )
    if not args.serve:
        return

    try:
        import viser
    except ImportError as error:
        raise RuntimeError(
            "Install the optional viewer dependencies with "
            "`pip install -r requirements-visualization.txt`"
        ) from error

    reconstruction = np.load(output / "reconstruction.npz", allow_pickle=False)
    points = reconstruction["points"].astype(np.float32)
    colors = reconstruction["colors"].astype(np.uint8)
    frame_ids = reconstruction["frame_ids"]
    if len(points) > args.max_viewer_points:
        indices = np.linspace(
            0, len(points) - 1, args.max_viewer_points, dtype=np.int64
        )
        points, colors, frame_ids = (
            points[indices],
            colors[indices],
            frame_ids[indices],
        )

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("-y")
    for frame_id in np.unique(frame_ids):
        mask = frame_ids == frame_id
        server.scene.add_point_cloud(
            "/frames/frame_{:03d}".format(int(frame_id)),
            points=points[mask],
            colors=colors[mask],
            point_size=args.point_size,
            point_shape="circle",
        )
    print("Loaded {:,} points. Open http://localhost:{}".format(len(points), args.port))
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Viewer stopped")


if __name__ == "__main__":
    main()
