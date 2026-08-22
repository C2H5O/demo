from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")

from datasets.scared_pair_dataset import (
    PAIR_CACHE_FORMAT_VERSION,
    PAIR_COORDINATE_CONVENTION,
)
from visualization.vggtomast3r_teacher_cache import (
    export_teacher_pair_cache_visualization,
)


def test_teacher_pair_cache_visualization_exports_coordinate_safe_artifacts(
    tmp_path,
) -> None:
    height, width = 4, 6
    frame_paths = []
    for index, value in enumerate((64, 192)):
        path = tmp_path / "frame_{:06d}.png".format(index * 2)
        Image.fromarray(
            np.full((height, width, 3), value, dtype=np.uint8)
        ).save(path)
        frame_paths.append(str(path))
    point_a = np.zeros((height, width, 3), dtype=np.float32)
    point_b = np.zeros_like(point_a)
    point_a[..., 2] = 1.0
    point_b[..., 0] = 0.5
    point_b[..., 2] = 2.0
    confidence = np.full((height, width), 0.8, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    metadata = {
        "dataset_id": 1,
        "keyframe_id": "keyframe_1",
        "frame_paths": frame_paths,
    }
    cache_path = tmp_path / "pair_000000_000002_stride_02.npz"
    np.savez_compressed(
        cache_path,
        frame_id_a=np.asarray(0),
        frame_id_b=np.asarray(2),
        frame_name_a=np.asarray("frame_000000.png"),
        frame_name_b=np.asarray("frame_000002.png"),
        pair_stride=np.asarray(2),
        image_shape=np.asarray((height, width)),
        teacher_variant=np.asarray("lora"),
        depth_a=point_a[..., 2],
        depth_b=point_b[..., 2],
        xyz_local_a=point_a,
        xyz_local_b=point_b,
        xyz_global_a=point_a,
        xyz_global_b=point_b,
        pts3d_a_in_a=point_a,
        pts3d_b_in_a=point_b,
        confidence_a=confidence,
        confidence_b=confidence,
        valid_mask_a=valid,
        valid_mask_b=valid,
        intrinsics_a=np.eye(3),
        intrinsics_b=np.eye(3),
        extrinsics_a=np.eye(4)[:3],
        extrinsics_b=np.eye(4)[:3],
        coordinate_convention=np.asarray(PAIR_COORDINATE_CONVENTION),
        cache_format_version=np.asarray(PAIR_CACHE_FORMAT_VERSION),
        lora_checkpoint=np.asarray("./teacher.pt"),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    output = export_teacher_pair_cache_visualization(
        cache_path,
        tmp_path / "visualization",
        min_depth=0.1,
        max_depth=3.0,
        point_stride=2,
        expected_teacher_variant="lora",
        expected_lora_checkpoint="./teacher.pt",
    )

    for name in (
        "teacher_pair_panel.png",
        "teacher_local_camera_a.ply",
        "teacher_local_camera_b.ply",
        "teacher_global.ply",
        "teacher_pair_reference_camera.ply",
        "pts3d_b_in_a_z_not_b_depth.npy",
        "metadata.json",
    ):
        assert (output / name).is_file()
    report = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert report["teacher_variant"] == "lora"
    assert "not camera-B depth" in report["coordinate_semantics"]["pts3d_b_in_a"]
    assert report["point_counts"]["pair_reference_camera"] > 0
