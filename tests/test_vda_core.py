from __future__ import annotations

import numpy as np

from evaluation.evaluate_vda import (
    _find_sequence_gt_depths,
    _streaming_scale_shift,
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


def test_separate_gt_root_is_mapped_by_dataset_and_keyframe(tmp_path) -> None:
    processed_keyframe = tmp_path / "processed" / "SCARED" / "dataset_8" / "keyframe_01"
    processed_keyframe.mkdir(parents=True)
    gt_root = tmp_path / "raw" / "scared"
    depth_directory = gt_root / "dataset_08" / "key_frame_1" / "data" / "depth"
    depth_directory.mkdir(parents=True)
    np.save(depth_directory / "depth_0007.npy", np.ones((2, 3)))

    selected, indexed = _find_sequence_gt_depths(
        {
            "sequence_id": "dataset_8/keyframe_01",
            "dataset_id": 8,
            "keyframe_id": "keyframe_01",
            "keyframe_directory": str(processed_keyframe),
            "depth_directory": None,
        },
        {
            "gt_root": str(gt_root),
            "gt_relative_directory": "data/depth",
        },
        {},
    )

    assert selected == depth_directory.resolve()
    assert indexed[7].name == "depth_0007.npy"


def test_sequence_scale_shift_matches_vda_single_lstsq(tmp_path) -> None:
    gt_metres = np.asarray(
        [
            [[1.0, 2.0], [4.0, 5.0]],
            [[1.5, 2.5], [3.5, 6.0]],
        ],
        dtype=np.float32,
    )
    predictions = np.asarray(
        [
            [[0.2, 0.4], [0.8, 1.0]],
            [[0.3, 0.5], [0.7, 1.2]],
        ],
        dtype=np.float32,
    )
    gt_paths = []
    for index, gt in enumerate(gt_metres):
        path = tmp_path / f"depth_{index:06d}.npy"
        np.save(path, gt * 1000.0)
        gt_paths.append(path)

    class Spool:
        height = width = 2

        def prediction(self, index):
            return predictions[index]

    pairs = [(index, index, path) for index, path in enumerate(gt_paths)]
    scale, shift, count = _streaming_scale_shift(
        pairs, Spool(), 0, "fixture"
    )

    valid = (gt_metres > 1e-3) & (gt_metres < 100.0)
    gt_disp_masked = 1.0 / (
        gt_metres[valid].reshape(-1, 1).astype(np.float64) + 1e-8
    )
    pred_disp_masked = (
        np.clip(predictions, a_min=1e-3, a_max=None)[valid]
        .reshape(-1, 1)
        .astype(np.float64)
    )
    A = np.concatenate(
        [pred_disp_masked, np.ones_like(pred_disp_masked)], axis=-1
    )
    expected_scale, expected_shift = np.linalg.lstsq(
        A, gt_disp_masked, rcond=None
    )[0]
    np.testing.assert_allclose(scale, expected_scale, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(shift, expected_shift, rtol=1e-7, atol=1e-12)
    assert count == int(valid.sum())
