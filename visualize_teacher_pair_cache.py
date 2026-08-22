from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from datasets.scared_pair_dataset import make_scared_pair_rgb_dataset, pair_metadata
from datasets.teacher_frame_cache import frame_metadata_from_pair
from utils.config import load_config
from visualization.vggtomast3r_teacher_frame_cache import export_composed_teacher_frames


def _rgb(image) -> np.ndarray:
    return np.round(
        ((image.float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).numpy()
    ).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize two independently inferred base-teacher frame caches"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/vggtomast3r_v1.yaml"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vggtomast3r_teacher_frame_visualization"),
    )
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--point-stride", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = make_scared_pair_rgb_dataset(config["dataset"], args.split)
    if not 0 <= args.pair_index < len(dataset):
        parser.error("--pair-index is outside the dataset")
    sample = dataset[args.pair_index]
    frames = frame_metadata_from_pair(pair_metadata(dataset, args.pair_index))
    rgb = np.stack([_rgb(image) for image in sample["images"]])
    output = export_composed_teacher_frames(
        Path(config["teacher"]["cache_root"]) / args.split,
        frames,
        args.output_dir,
        (int(config["teacher"]["image_height"]), int(config["teacher"]["image_width"])),
        str(config["teacher"]["pretrained_checkpoint"]),
        rgb,
        args.min_depth,
        args.max_depth,
        args.point_stride,
    )
    print(output)


if __name__ == "__main__":
    main()
