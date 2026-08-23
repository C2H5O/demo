from __future__ import annotations

import numpy as np

from evaluation.evaluate_vda import (
    _find_sequence_gt_depths,
    _student_depth_to_vda_disparity,
)


def test_student_depth_is_converted_to_disparity() -> None:
    depth = np.asarray([[[1.0, 2.0, 4.0]]], dtype=np.float32)
    np.testing.assert_allclose(
        _student_depth_to_vda_disparity(depth),
        np.asarray([[[1.0, 0.5, 0.25]]], dtype=np.float32),
    )


def test_configured_gt_directory_is_indexed(tmp_path) -> None:
    keyframe = tmp_path / "keyframe_0"
    depth_directory = keyframe / "data" / "depth"
    depth_directory.mkdir(parents=True)
    np.save(depth_directory / "depth_0007.npy", np.ones((2, 3)))
    selected, indexed = _find_sequence_gt_depths(
        {
            "sequence_id": "dataset_8/keyframe_0",
            "keyframe_directory": str(keyframe),
            "depth_directory": str(depth_directory),
        },
        {"gt_relative_directory": "data/depth"},
        {},
    )
    assert selected == depth_directory.resolve()
    assert indexed[7].name == "depth_0007.npy"
