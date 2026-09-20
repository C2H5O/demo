import json

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from evaluation.evaluate_crossclip_projection import evaluate_vda
from evaluation.evaluate_vda import _SequencePredictionSpool
from inference.student_video import sequence_frames


class ConstantPlane(nn.Module):
    def forward(self, images, include_global_points=False):
        b, t, _, h, w = images.shape
        return {"depth": torch.ones(b, t, h, w),
                "intrinsics": torch.eye(3).repeat(b, t, 1, 1),
                "extrinsics": torch.eye(4)[:3].repeat(b, t, 1, 1)}


def make_scared(tmp_path, count=3):
    for dataset_id in (8, 9):
        keyframe = tmp_path / f"dataset_{dataset_id}" / "keyframe_1" / "data"
        for sub in ("left_finalpass", "depth", "frame_data"):
            (keyframe / sub).mkdir(parents=True)
        for frame in range(count):
            Image.new("RGB", (6, 4)).save(keyframe / "left_finalpass" / f"frame_{frame:06d}.png")
            np.save(keyframe / "depth" / f"depth_{frame:06d}.npy", np.ones((4, 6)) * 1000)
            (keyframe / "frame_data" / f"frame_data{frame:06d}.json").write_text(json.dumps({
                "camera-calibration": {"KL": [[100,0,3],[0,100,2],[0,0,1]]},
                "camera-pose": np.eye(4).tolist()}))
    cfg = {"device": "cpu", "student": {"checkpoint": "unused"},
           "dataset": {"root": str(tmp_path), "clip_length": 16, "sample_stride": 1, "window_stride": 8,
                       "image_height": 4, "image_width": 6, "normalize_mode": "zero_one"},
           "vda_evaluation": {"rgb_root": str(tmp_path), "gt_root": str(tmp_path),
                              "checkpoint": "unused", "output": str(tmp_path / "result.json"), "split": "test"}}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    return config, cfg


def test_full_evaluation_discovers_short_sequences_scores_tae_and_writes_speed(tmp_path, monkeypatch):
    config, _ = make_scared(tmp_path)
    monkeypatch.setattr(
        "evaluation.evaluate_crossclip_projection.ensure_merged_student_checkpoint",
        lambda checkpoint, loaded: checkpoint,
    )
    monkeypatch.setattr("evaluation.evaluate_crossclip_projection._evaluation_model",
                        lambda *args: ConstantPlane().eval())
    result = evaluate_vda(config)
    assert result["full_test_set"] and result["complete_tae_coverage"]
    assert result["inference_frame_count"] == 6
    assert result["model_input_frame_count"] == 64
    assert result["metrics"]["tae"] == pytest.approx(0)
    assert result["tae_reference"] == "DepthAnything/Video-Depth-Anything benchmark/eval/eval_tae.py"
    assert result["tae_projection_collision"] == "direct_assignment_matching_vda"
    assert result["tae_empty_projection"] == "zero_error_matching_vda"
    assert result["tae_alignment"] == "single_sequence_disparity_scale_shift_fit"
    assert result["metrics"]["abs_relative_difference"] == pytest.approx(0, abs=1e-6)
    assert result["mean_frame_inference_seconds"] == result["total_model_inference_seconds"] / 6
    assert json.loads((tmp_path / "result.json").read_text())["sequence_count"] == 2
    assert not list(tmp_path.glob(".vda_spool_*"))


def test_224_inference_and_metric_grid(tmp_path, monkeypatch):
    config_path, config = make_scared(tmp_path, count=1)
    config["inference"] = {"image_height": 224, "image_width": 280}
    config["vda_evaluation"].update({"evaluation_height": 224,
                                     "evaluation_width": 280, "tae": {"enabled": False}})
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        "evaluation.evaluate_crossclip_projection.ensure_merged_student_checkpoint",
        lambda checkpoint, loaded: checkpoint)
    monkeypatch.setattr("evaluation.evaluate_crossclip_projection._evaluation_model",
                        lambda *args: ConstantPlane().eval())
    observed = []
    original_forward = ConstantPlane.forward

    def record(self, images, include_global_points=False):
        observed.append(tuple(images.shape))
        return original_forward(self, images, include_global_points)

    monkeypatch.setattr(ConstantPlane, "forward", record)
    result = evaluate_vda(config_path)
    assert observed == [(1, 32, 3, 224, 280)] * 2
    assert result["model_input_resolution_hw"] == [224, 280]
    assert result["evaluation_resolution_hw"] == [224, 280]
    assert all(item["inference"]["native_prediction_resolution_hw"] == (224, 280)
               for item in result["sequences"])
    assert set(result["metrics"]) == {"abs_relative_difference", "rmse_linear", "delta1_acc"}


def test_precomputed_training_rgb_resizes_to_224_inference_grid(tmp_path):
    image_path = tmp_path / "frame_000000.png"
    Image.new("RGB", (560, 448), color=(128, 128, 128)).save(image_path)
    sequence = {"frame_paths": [str(image_path)],
                "preprocessing_identity": "canonical_student_rgb"}
    frames = sequence_frames(
        sequence, {"image_height": 448, "image_width": 560},
        inference_height=224, inference_width=280)
    assert frames[0].shape == (3, 224, 280)
    assert (frames.height, frames.width) == (224, 280)


def test_native_metric_grid_disparity_skips_interpolation(tmp_path, monkeypatch):
    import evaluation.evaluate_vda as vda_module

    spool = _SequencePredictionSpool(tmp_path, 1, 224, 280)
    try:
        monkeypatch.setattr(vda_module, "_opencv", lambda: pytest.fail("same grid must not resize"))
        values = np.full((1, 224, 280), 0.5, dtype=np.float32)
        spool.add([0], values)
        np.testing.assert_array_equal(spool.prediction(0), values[0])
    finally:
        spool.close()


def test_native_448_disparity_and_gt_resize_to_metric_grid(tmp_path):
    import cv2
    from evaluation.evaluate_vda import _resized_gt

    spool = _SequencePredictionSpool(tmp_path, 1, 224, 280)
    try:
        native = np.arange(448 * 560, dtype=np.float32).reshape(448, 560)
        spool.add([0], native[None])
        expected = cv2.resize(native, (280, 224), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_allclose(spool.prediction(0), expected)
        gt_path = tmp_path / "depth.npy"
        np.save(gt_path, native)
        np.testing.assert_array_equal(
            _resized_gt(gt_path, 0, 224, 280),
            cv2.resize(native.astype(np.float32) * 0.001, (280, 224),
                       interpolation=cv2.INTER_NEAREST),
        )
    finally:
        spool.close()


def test_window_limit_never_claims_full_test_set(tmp_path, monkeypatch):
    config, _ = make_scared(tmp_path)
    monkeypatch.setattr(
        "evaluation.evaluate_crossclip_projection.ensure_merged_student_checkpoint",
        lambda checkpoint, loaded: checkpoint,
    )
    monkeypatch.setattr("evaluation.evaluate_crossclip_projection._evaluation_model",
                        lambda *args: ConstantPlane().eval())
    result = evaluate_vda(config, limit_clips=1)
    assert not result["full_test_set"]
    assert result["sequence_count"] == 1


def test_missing_gt_never_claims_complete_coverage(tmp_path, monkeypatch):
    config, _ = make_scared(tmp_path)
    for path in (tmp_path / "dataset_9/keyframe_1/data/depth").glob("*.npy"):
        path.unlink()
    monkeypatch.setattr("evaluation.evaluate_crossclip_projection._evaluation_model",
                        lambda *args: ConstantPlane().eval())
    monkeypatch.setattr(
        "evaluation.evaluate_crossclip_projection.ensure_merged_student_checkpoint",
        lambda checkpoint, loaded: checkpoint,
    )
    result = evaluate_vda(config)
    assert not result["full_test_set"] and not result["complete_gt_coverage"]
    assert len(result["skipped_sequences_without_gt"]) == 1


def test_missing_cameras_fails_before_model_loading(tmp_path, monkeypatch):
    config, _ = make_scared(tmp_path)
    (tmp_path / "dataset_8/keyframe_1/data/frame_data/frame_data000001.json").unlink()
    def forbidden(*args):
        pytest.fail("Model must not load before camera preflight")
    monkeypatch.setattr("evaluation.evaluate_crossclip_projection._evaluation_model", forbidden)
    with pytest.raises(FileNotFoundError, match="TAE dataset cameras"):
        evaluate_vda(config)


def test_sequence_visualizer_uses_shared_pipeline_and_exports_all_frames(tmp_path, monkeypatch):
    from datasets.scared_clip_dataset import make_scared_rgb_dataset
    from visualization.student_video import export_student_video
    _, cfg = make_scared(tmp_path)
    dataset = make_scared_rgb_dataset({**cfg["dataset"], "clip_length": 1}, "test")
    monkeypatch.setattr("visualization.student_video._evaluation_model", lambda *args: ConstantPlane().eval())
    out = export_student_video(cfg, dataset, 0, tmp_path / "vis", "student",
                               None, "test", 0.1, 10, 1)
    assert len(list((out / "depth").glob("*.npy"))) == 3
    assert len(list((out / "pointcloud_local").glob("*.ply"))) == 3
    assert len(list((out / "camera_windows").glob("*.npz"))) == 1
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["absolute_frame_ids"] == [0, 1, 2]
    assert meta["inference"]["output_frame_count"] == 3
    assert not (out / "pointcloud_global_merged.ply").exists()
