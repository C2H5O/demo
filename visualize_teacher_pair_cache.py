from __future__ import annotations

import argparse
from pathlib import Path

from visualization.vggtomast3r_teacher_cache import (
    _default_output,
    export_teacher_pair_cache_visualization,
    resolve_config_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a VGG-to-MASt3R V1 teacher pair cache"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/vggtomast3r_v1.yaml"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--pair-index", type=int, default=None)
    parser.add_argument(
        "--sequence-id",
        default=None,
        help="Example: dataset_2/keyframe_1; use with --frame-id-a",
    )
    parser.add_argument("--frame-id-a", type=int, default=None)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Direct cache path; bypasses config/split/pair-index resolution",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/vggtomast3r_teacher_cache_visualization"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    args = parser.parse_args()

    rgb = None
    expected_variant = None
    expected_lora = None
    if args.cache is None:
        cache_path, rgb, config, pair_index = resolve_config_pair(
            args.config,
            args.split,
            args.pair_index,
            args.sequence_id,
            args.frame_id_a,
        )
        print("Resolved pair_index={}".format(pair_index))
        expected_variant = "lora"
        expected_lora = str(config["teacher"].get("lora_checkpoint", ""))
    else:
        cache_path = args.cache
    output_dir = args.output_dir or _default_output(cache_path, args.output_root)
    output = export_teacher_pair_cache_visualization(
        cache_path=cache_path,
        output_dir=output_dir,
        rgb=rgb,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        point_stride=args.point_stride,
        confidence_threshold=args.confidence_threshold,
        expected_teacher_variant=expected_variant,
        expected_lora_checkpoint=expected_lora,
    )
    print(output)


if __name__ == "__main__":
    main()
