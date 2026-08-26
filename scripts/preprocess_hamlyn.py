"""Preprocess the Endo-Depth-and-Motion rectified Hamlyn evaluation release."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.preprocessing.common import ProcessedFrame, SequenceWriter, natural_key, source_frame_id

RGB_NAMES = {"images", "image", "rgb", "left", "left_images", "rectified_left"}
DEPTH_NAMES = {"depth", "depth_data", "depths", "gt_depth"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def files(directory: Path) -> list[Path]:
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)


def read_depth_mm(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim != 2:
        raise ValueError(f"expected a one-channel depth image: {path}")
    if value.dtype != np.uint16:
        raise ValueError(f"expected official uint16 mm depth PNG/TIFF, got {value.dtype}: {path}")
    return value.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rectified-rgb", action="store_true", help="required acknowledgement for Endo-Depth's published rectified RGB")
    args = parser.parse_args()
    if not args.rectified_rgb:
        parser.error("this evaluator requires the published rectified Hamlyn RGB; pass --rectified-rgb after checking the release")
    rgb_dirs = [p for p in args.input_root.rglob("*") if p.is_dir() and p.name.lower() in RGB_NAMES and files(p)]
    results = []
    for rgb_dir in rgb_dirs:
        parent = rgb_dir.parent
        depth_dirs = [p for p in parent.iterdir() if p.is_dir() and p.name.lower() in DEPTH_NAMES]
        if not depth_dirs:
            continue
        depth_by_id = {source_frame_id(p, index): p for index, p in enumerate(files(depth_dirs[0]))}
        rgb_files = files(rgb_dir)
        pairs = []
        for index, rgb in enumerate(rgb_files):
            identifier = source_frame_id(rgb, index)
            if identifier not in depth_by_id:
                raise RuntimeError(f"missing Hamlyn GT depth for RGB {rgb} (source ID {identifier})")
            pairs.append((rgb, depth_by_id[identifier], identifier))
        if len(depth_by_id) != len(pairs):
            raise RuntimeError(f"Hamlyn RGB/depth count mismatch in {parent}; refusing silent GT omission")
        sequence_id = parent.relative_to(args.input_root).as_posix().replace("/", "_") or parent.name
        destination = args.output_root / "Hamlyn" / sequence_id
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
        with SequenceWriter(destination, dataset_name="Hamlyn", sequence_id=sequence_id, overwrite=args.overwrite, dry_run=args.dry_run, evaluation_only=True) as writer:
            for index, (rgb_path, depth_path, identifier) in enumerate(pairs):
                with Image.open(rgb_path) as rgb:
                    depth = read_depth_mm(depth_path)
                    if rgb.size != (depth.shape[1], depth.shape[0]):
                        raise RuntimeError(f"RGB/depth source-grid mismatch: {rgb_path} {rgb.size} vs {depth_path} {depth.shape[::-1]}")
                    writer.write_rgb(ProcessedFrame(rgb_path, identifier), index, rgb.copy())
                    writer.write_depth_mm(index, depth)
            writer.complete({"evaluation_only": True, "eye": "left", "rgb_rectification": "published rectified RGB attested by operator", "source_depth_unit": "mm", "output_depth_unit": "mm", "conversion_factor": 1.0, "depth_type": "uint16 PNG/TIFF source; float32 NPY output", "invalid_depth_value": 0, "depth_source_mapping": [{"processed_index": i, "source_file": str(d)} for i, (_r, d, _id) in enumerate(pairs)]})
            results.append({"sequence": sequence_id, "frames": len(writer.frames), "output": str(destination), "dry_run": args.dry_run})
    if not results:
        raise SystemExit("no Hamlyn image/depth directory pair found")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
