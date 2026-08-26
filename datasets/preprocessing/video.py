"""Streaming video decode helpers; never accumulate a full video in memory."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from PIL import Image


def stream_video_frames(video_path: Path) -> Iterator[tuple[int, float, Image.Image]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("streaming video preprocessing requires opencv-python; install requirements.txt") from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield index, (index / fps if fps > 0 else None), Image.fromarray(rgb, mode="RGB")
            index += 1
    finally:
        capture.release()
