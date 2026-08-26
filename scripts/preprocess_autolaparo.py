"""Stream AutoLaparo RGB files/videos into a common, non-distorted 4:5 FOV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.preprocessing.common import ProcessedFrame, SequenceWriter, contiguous_runs, image_files, process_image_run
from datasets.preprocessing.video import stream_video_frames

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rgb_dir_names = {"rgb", "images", "frames", "image_sequence"}
    image_dirs = [p for p in args.input_root.rglob("*") if p.is_dir() and p.name.lower() in rgb_dir_names and image_files(p)]
    videos = sorted((p for p in args.input_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES), key=lambda p: p.as_posix())
    results = []
    # Prefer image directories only when their parent has no video, avoiding duplicate extraction.
    for directory in image_dirs:
        if any(video.parent == directory or video.parent == directory.parent for video in videos):
            continue
        for run_number, files in enumerate(contiguous_runs(image_files(directory))):
            sequence_id = directory.relative_to(args.input_root).as_posix().replace("/", "_") + (f"_run{run_number:02d}" if len(files) < len(image_files(directory)) else "")
            destination = args.output_root / "AutoLaparo" / sequence_id
            if not args.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
            results.append(process_image_run(destination, "AutoLaparo", sequence_id, files, overwrite=args.overwrite, dry_run=args.dry_run,
                extra_metadata={"source_type": "image sequence", "source_fps": "UNVERIFIED for this local release", "temporal_step": 1, "canonical_fov_note": "full input FOV preserved with proportional contain+padding"}))
    for video in videos:
        sequence_id = video.relative_to(args.input_root).with_suffix("").as_posix().replace("/", "_")
        destination = args.output_root / "AutoLaparo" / sequence_id
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
        with SequenceWriter(destination, dataset_name="AutoLaparo", sequence_id=sequence_id, overwrite=args.overwrite, dry_run=args.dry_run) as writer:
            for index, timestamp, image in stream_video_frames(video):
                writer.write_rgb(ProcessedFrame(video, index, timestamp), index, image)
            writer.complete({"source_type": "streaming video", "source_video": str(video), "temporal_step": 1, "official_dataset_fps": 25, "canonical_fov_note": "full input FOV preserved with proportional contain+padding"})
            results.append({"sequence": sequence_id, "frames": len(writer.frames), "output": str(destination), "dry_run": args.dry_run})
    if not results:
        raise SystemExit("no AutoLaparo images or supported videos found")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
