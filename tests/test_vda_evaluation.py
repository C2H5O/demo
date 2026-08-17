from __future__ import annotations

import inspect

import numpy as np

from evaluation.evaluate_vda import (
    _SequencePredictionSpool,
    _find_sequence_gt_depths,
    _load_student_memory_efficient,
    _load_teacher_cache_clip,
    _select_vda_evaluation_config,
    _streaming_metrics,
    _streaming_scale_shift,
    _student_depth_to_vda_disparity,
    evaluate,
    evaluate_depth_core,
)
from utils.config import load_config


def test_output_adapter_converts_depth_to_disparity() -> None:
    depth = np.asarray([[[1.0, 2.0, 4.0]]], dtype=np.float32)

    disparity = _student_depth_to_vda_disparity(depth)

    np.testing.assert_allclose(disparity, [[[1.0, 0.5, 0.25]]])


def test_bilinear_experiment_selects_vda_without_removing_endo3r() -> None:
    config = load_config("configs/student_distillation_head_bilinear.yaml")

    selected, explicit = _select_vda_evaluation_config(config)

    assert explicit is True
    assert selected["protocol"] == "video-depth-anything-depth"
    assert selected["split"] == "test"
    assert selected["checkpoint"].endswith("student_distill3r_448x560_bilinear_head/last.pt")
    assert selected["output"].endswith("evaluation_test_vda.json")
    assert config["evaluation"]["protocol"] == "endo3r"
    assert config["evaluation"]["output"].endswith("evaluation_test_endo3r.json")


def test_vda_legacy_config_falls_back_to_existing_evaluation_section() -> None:
    config = {"evaluation": {"split": "test", "checkpoint": "student.pt"}}

    selected, explicit = _select_vda_evaluation_config(config)

    assert explicit is False
    assert selected == config["evaluation"]


def test_bilinear_evaluation_scripts_default_to_vda_and_retain_endo3r() -> None:
    vda_script = open(
        "scripts/evaluate_student_head_bilinear.sh", encoding="utf-8"
    ).read()
    endo3r_script = open(
        "scripts/evaluate_student_head_bilinear_endo3r.sh", encoding="utf-8"
    ).read()

    assert "python evaluate_vda.py" in vda_script
    assert "python evaluate.py" not in vda_script
    assert "python evaluate.py" in endo3r_script


def test_teacher_cache_adapter_uses_local_z_depth(tmp_path) -> None:
    path = tmp_path / "clip.npz"
    points = np.zeros((2, 3, 4, 3), dtype=np.float16)
    points[0, ..., 2] = 2.0
    points[1, ..., 2] = 4.0
    metadata = {
        "frame_names": ["frame_0001.png", "frame_0002.png"],
        "frame_indices": [1, 2],
    }
    np.savez(
        path,
        xyz_local=points,
        frame_names=np.asarray(metadata["frame_names"]),
        frame_indices=np.asarray(metadata["frame_indices"]),
        teacher_variant=np.asarray("base"),
    )

    indices, disparity, variant = _load_teacher_cache_clip(
        path, metadata
    )

    assert indices == [1, 2]
    assert variant == "base"
    np.testing.assert_allclose(disparity[0], 0.5)
    np.testing.assert_allclose(disparity[1], 0.25)


def test_gt_discovery_uses_configured_dataset_depth_directory(tmp_path) -> None:
    keyframe = tmp_path / "keyframe_0"
    depth_directory = keyframe / "data" / "depth"
    depth_directory.mkdir(parents=True)
    np.save(depth_directory / "depth_0007.npy", np.ones((2, 3)))
    sequence = {
        "sequence_id": "dataset_8/keyframe_0",
        "keyframe_directory": str(keyframe),
        "depth_directory": str(depth_directory),
    }

    selected, indexed = _find_sequence_gt_depths(
        sequence,
        {"gt_relative_directory": "data/depth"},
        {
            "ground_truth": {
                "directory_keys": ["depth_directory"],
                "relative_directories": ["data/depth"],
            }
        },
    )

    assert selected == depth_directory.resolve()
    assert indexed[7].name == "depth_0007.npy"


def test_core_retains_upstream_scale_shift_operations() -> None:
    source = inspect.getsource(evaluate_depth_core)

    required_fragments = (
        "np.linalg.lstsq(A, gt_disp_masked, rcond=None)[0]",
        "aligned_pred = scale * infs + shift",
        "pred_depth = depth2disparity(aligned_pred)",
        "valid_mask = np.logical_and((gts > 1e-3), (gts < dataset_max_depth))",
    )
    for fragment in required_fragments:
        assert fragment in source


def test_adapter_uses_bounded_memory_for_full_split() -> None:
    loader_source = inspect.getsource(_load_student_memory_efficient)
    evaluate_source = inspect.getsource(evaluate)
    spool_source = inspect.getsource(_SequencePredictionSpool)
    alignment_source = inspect.getsource(_streaming_scale_shift)
    metrics_source = inspect.getsource(_streaming_metrics)

    assert "mmap=True" in loader_source
    assert "checkpoint.clear()" in loader_source
    assert "assign=True" in loader_source
    assert "num_workers=0" in evaluate_source
    assert "max_eval_len" not in evaluate_source
    assert "np.memmap" in spool_source
    assert "np.linalg.qr" in alignment_source
    assert "np.linalg.lstsq" in alignment_source
    assert "metric_func(" in metrics_source
    assert "processed_clips == len(clip_indices)" in evaluate_source
    assert "_load_teacher_cache_clip" in evaluate_source
    assert 'evaluation_height = int(dataset_config["image_height"])' in evaluate_source
    assert 'evaluation_width = int(dataset_config["image_width"])' in evaluate_source
