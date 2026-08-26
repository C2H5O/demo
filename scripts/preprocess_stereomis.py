"""StereoMIS preprocessing with explicit stereo-layout and rectification gates.

No layout is inferred from image dimensions.  A raw video may be split only
when the caller explicitly selects a documented layout, and it is accepted only
after an official/author rectification result has been supplied or attested.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.preprocessing.common import ProcessedFrame, SequenceWriter, contiguous_runs, image_files, process_image_run
from datasets.preprocessing.video import stream_video_frames


def split_stereo_frame(image: Image.Image, layout: str) -> tuple[Image.Image, Image.Image]:
    """Split only an explicit packed layout; reject odd dimensions and unknown formats."""
    width, height = image.size
    if layout == "side-by-side":
        if width % 2:
            raise ValueError(f"side-by-side frame has odd width: {width}")
        return image.crop((0, 0, width // 2, height)), image.crop((width // 2, 0, width, height))
    if layout == "top-bottom":
        if height % 2:
            raise ValueError(f"top-bottom frame has odd height: {height}")
        return image.crop((0, 0, width, height // 2)), image.crop((0, height // 2, width, height))
    raise ValueError(f"unsupported explicit stereo layout: {layout}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--already-rectified-left", action="store_true", help="required for a rectified_left/ or left/ image directory")
    parser.add_argument("--stereo-layout", choices=("side-by-side", "top-bottom"), help="required for packed video; never inferred")
    parser.add_argument("--allow-packed-video-after-official-rectification", action="store_true", help="acknowledges that decoding/split input has already been rectified by an official workflow")
    args = parser.parse_args()
    candidates = [p for p in args.input_root.rglob("*") if p.is_dir() and p.name.lower() in {"rectified_left", "left_rectified", "left"} and image_files(p)]
    videos = [p for p in args.input_root.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    results = []
    if candidates:
        if not args.already_rectified_left:
            parser.error("refusing unverified StereoMIS left images; pass --already-rectified-left only for author/official rectified left output")
        for directory in candidates:
            all_files = image_files(directory)
            for run_number, files in enumerate(contiguous_runs(all_files)):
                sequence_id = directory.parent.relative_to(args.input_root).as_posix().replace("/", "_")
                if len(files) < len(all_files):
                    sequence_id += f"_run{run_number:02d}"
                destination = args.output_root / "StereoMIS" / sequence_id
                if not args.dry_run:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                results.append(process_image_run(destination, "StereoMIS", sequence_id, files, overwrite=args.overwrite, dry_run=args.dry_run,
                    extra_metadata={"eye": "left", "stereo_layout": "separate files", "rectification": "attested official/author rectified-left input", "calibration_files": [str(p) for p in directory.parent.glob("*Calibration*.ini")], "training_gt_written": False}))
    if videos:
        if not (args.stereo_layout and args.allow_packed_video_after_official_rectification):
            parser.error("packed StereoMIS video needs explicit --stereo-layout and --allow-packed-video-after-official-rectification; this script never guesses or self-invents calibration")
        for video in videos:
            sequence_id = video.relative_to(args.input_root).with_suffix("").as_posix().replace("/", "_")
            destination = args.output_root / "StereoMIS" / sequence_id
            if not args.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
            with SequenceWriter(destination, dataset_name="StereoMIS", sequence_id=sequence_id, overwrite=args.overwrite, dry_run=args.dry_run) as writer:
                for index, timestamp, frame in stream_video_frames(video):
                    left, _right = split_stereo_frame(frame, args.stereo_layout)
                    writer.write_rgb(ProcessedFrame(video, index, timestamp), index, left)
                writer.complete({"eye": "left", "stereo_layout": args.stereo_layout, "rectification": "caller attested packed video already has official rectification", "source_video": str(video), "training_gt_written": False})
                results.append({"sequence": sequence_id, "frames": len(writer.frames), "output": str(destination), "dry_run": args.dry_run})
    if not results:
        raise SystemExit("no supported StereoMIS candidate found; expected rectified_left/ (attested) or a packed video with explicit layout")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
