from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("cv2")

from datasets.teacher_frame_cache import (
    FRAME_CACHE_FORMAT_VERSION,
    FRAME_COORDINATE_CONVENTION,
    teacher_frame_cache_path,
)
from visualization.vggtomast3r_teacher_frame_cache import (
    export_composed_teacher_frames,
)


BASE = "./checkpoints/vggt_omega/vggt_omega_1b_512.pt"


def _write(root, metadata, value):
    shape = (4, 6)
    depth = np.full(shape, value, dtype=np.float32)
    points = np.zeros(shape + (3,), dtype=np.float32)
    points[..., 2] = depth
    path = teacher_frame_cache_path(root, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dataset_id=np.asarray(metadata["dataset_id"]),
        keyframe_id=np.asarray(metadata["keyframe_id"]),
        sequence_id=np.asarray(metadata["sequence_id"]),
        frame_id=np.asarray(metadata["frame_id"]),
        frame_index=np.asarray(metadata["frame_index"]),
        frame_name=np.asarray(metadata["frame_name"]),
        image_shape=np.asarray(shape),
        teacher_variant=np.asarray("base"),
        inference_frame_count=np.asarray(1),
        depth=depth,
        xyz_local=points,
        confidence=np.full(shape, 0.8, dtype=np.float32),
        valid_mask=np.ones(shape, dtype=bool),
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32)[:3],
        coordinate_convention=np.asarray(FRAME_COORDINATE_CONVENTION),
        cache_format_version=np.asarray(FRAME_CACHE_FORMAT_VERSION),
        base_checkpoint=np.asarray(BASE),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_teacher_frame_composition_visualization_is_coordinate_safe(tmp_path) -> None:
    common = {
        "dataset_id": 1,
        "keyframe_id": "keyframe_1",
        "sequence_id": "seq",
        "sequence_length": 3,
    }
    frames = [
        {
            **common,
            "frame_id": index * 2,
            "frame_index": index * 2,
            "frame_name": "frame_{:06d}.png".format(index * 2),
            "frame_path": "unused",
        }
        for index in range(2)
    ]
    for index, metadata in enumerate(frames):
        _write(tmp_path, metadata, 1.0 + index)
    rgb = np.stack(
        [np.full((4, 6, 3), value, dtype=np.uint8) for value in (64, 192)]
    )
    output = export_composed_teacher_frames(
        tmp_path,
        frames,
        tmp_path / "visualization",
        (4, 6),
        BASE,
        rgb,
        min_depth=0.1,
        max_depth=3.0,
        point_stride=2,
    )
    for name in (
        "teacher_frame_panel.png",
        "depth_a_fixed.png",
        "depth_a_adaptive.png",
        "teacher_camera_a_local.ply",
        "teacher_camera_b_local.ply",
        "metadata.json",
    ):
        assert (output / name).is_file()
    report = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert "separate camera-local" in report["warning"]
    assert len(report["cache_paths"]) == 2
