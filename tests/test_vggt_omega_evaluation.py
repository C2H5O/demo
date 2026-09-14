import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from evaluation.teacher_scale_diagnostics import TeacherWindowScaleDiagnostics
from inference.student_video import KEYFRAMES, infer_vda_video
from inference.vggt_omega_video import VGGTOmegaSequenceFrames
from evaluation.evaluate_crossclip_projection import evaluate_vggt_omega_online
from visualization.vggt_omega_video import (
    _visualization_dataset,
    export_vggt_omega_video,
)


class IndexedFrames:
    def __init__(self, count):
        self.count = count
        self.loads = []

    def __len__(self):
        return self.count

    def load_indices(self, indices):
        self.loads.append(list(indices))
        return torch.tensor(indices, dtype=torch.float32)[:, None, None, None].expand(-1, 3, 2, 3)


class FrozenDenseTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.windows = []

    def forward(self, images):
        self.windows.append(images[:, :, 0, 0, 0].to(torch.int64).tolist()[0])
        b, f, _, h, w = images.shape
        return {
            "depth": torch.ones(b, f, h, w),
            "intrinsics": torch.eye(3).repeat(b, f, 1, 1),
            "extrinsics": torch.eye(4)[:3].repeat(b, f, 1, 1),
        }


def test_teacher_uses_the_shared_32_frame_vda_windows_and_stitching():
    frames = IndexedFrames(80)
    teacher = FrozenDenseTeacher().eval()
    emitted = []

    def forward(model, images):
        return model(images)

    stats = infer_vda_video(
        teacher,
        frames,
        lambda start, disparity, intrinsics: emitted.extend(range(start, start + len(disparity))),
        device="cpu",
        forward_model=forward,
        prediction_label="VGGT-Omega",
    )

    assert teacher.windows[0] == list(range(32))
    assert teacher.windows[1] == [0, 12, *range(24, 32), *range(32, 54)]
    assert teacher.windows[2] == [0, 34, *range(46, 54), *range(54, 76)]
    assert len(teacher.windows) == 4
    assert emitted == list(range(80))
    assert stats["window_length"] == 32
    assert stats["overlap"] == 10
    assert stats["window_stride"] == 22
    assert KEYFRAMES == [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]


def test_official_teacher_preprocessing_is_called_on_raw_frame_paths(monkeypatch, tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"frame_{index:06d}.png"
        Image.new("RGB", (8, 6), color=(index, 0, 0)).save(path)
        paths.append(path)
    calls = []

    def official_loader(names, mode, image_resolution, patch_size):
        calls.append((names, mode, image_resolution, patch_size))
        return torch.zeros(len(names), 3, 32, 48)

    monkeypatch.setitem(
        sys.modules,
        "vggt_omega.utils.load_fn",
        SimpleNamespace(load_and_preprocess_images=official_loader),
    )
    frames = VGGTOmegaSequenceFrames(paths, image_resolution=512, mode="balanced")
    images = frames.load_indices([1, 0])

    assert images.shape == (2, 3, 32, 48)
    assert calls == [([str(paths[1]), str(paths[0])], "balanced", 512, 16)]
    assert frames.metadata()["observed_input_shape_hw"] == [32, 48]


def test_window_scale_diagnostic_is_record_only(monkeypatch):
    sequence = {"frame_paths": ["frame_000000.png", "frame_000001.png"]}
    diagnostic = TeacherWindowScaleDiagnostics(
        sequence, ("unused", {0: "gt0", 1: "gt1"}), 0
    )
    monkeypatch.setattr(
        "evaluation.teacher_scale_diagnostics._resized_gt",
        lambda path, channel, height, width: np.full((height, width), 6.0, np.float32),
    )
    depth = np.full((2, 2, 3), 2.0, np.float32)
    original = depth.copy()
    diagnostic(0, [0, 1], depth)

    assert np.array_equal(depth, original)
    assert diagnostic.records[0]["scale_to_gt"] == pytest.approx(3.0)
    assert diagnostic.records[0]["record_only"] is True
    assert diagnostic.records[0]["applied_to_prediction"] is False
    assert diagnostic.records[0]["used_for_stitching"] is False


def test_teacher_evaluator_reuses_sequence_metrics_and_reports_required_shape(monkeypatch, tmp_path):
    for dataset_id in (8, 9):
        data = tmp_path / f"dataset_{dataset_id}" / "keyframe_1" / "data"
        for directory in ("left_finalpass", "depth", "frame_data"):
            (data / directory).mkdir(parents=True)
        for frame_id in range(2):
            Image.new("RGB", (6, 4)).save(data / "left_finalpass" / f"frame_{frame_id:06d}.png")
            np.save(data / "depth" / f"depth_{frame_id:06d}.npy", np.ones((4, 6)) * 1000)
            (data / "frame_data" / f"frame_data{frame_id:06d}.json").write_text(
                '{"camera-calibration":{"KL":[[100,0,3],[0,100,2],[0,0,1]]},'
                '"camera-pose":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}'
            )
    output = tmp_path / "teacher.json"
    config = tmp_path / "teacher.yaml"
    config.write_text(yaml.safe_dump({
        "device": "cpu",
        "inference": {"acceleration": "none"},
        "teacher": {"pretrained_checkpoint": "unused"},
        "dataset": {"root": str(tmp_path), "image_height": 4, "image_width": 6,
                    "resize_mode": "resize", "normalize_mode": "zero_one"},
        "vggt_omega_baseline_vda_evaluation": {
            "rgb_root": str(tmp_path), "gt_root": str(tmp_path), "split": "test",
            "output": str(output), "window_scale_diagnostic": True,
        },
    }))

    class FakeFrames:
        def __init__(self, paths, **kwargs):
            self.paths = paths
        def __len__(self):
            return len(self.paths)
        def metadata(self):
            return {"implementation": "official-fake", "observed_input_shape_hw": [8, 12]}

    def fake_infer(model, frames, emit, *, inspect_window, **kwargs):
        depth = np.ones((32, 8, 12), np.float32)
        positions = [min(index, len(frames) - 1) for index in range(32)]
        inspect_window(0, positions, depth)
        emit(0, np.ones((len(frames), 8, 12), np.float32), np.zeros((len(frames), 3, 3)))
        return {
            "output_frame_count": len(frames), "window_count": 1,
            "model_input_frame_count": 32, "model_forward_seconds": 1.0,
            "timing_scope": "same model-only scope", "alignment_fallback_count": 0,
        }

    monkeypatch.setattr("evaluation.evaluate_crossclip_projection._evaluation_model", lambda *args: nn.Identity().eval())
    monkeypatch.setattr("evaluation.evaluate_crossclip_projection.VGGTOmegaSequenceFrames", FakeFrames)
    monkeypatch.setattr("evaluation.evaluate_crossclip_projection.infer_vggt_omega_video", fake_infer)
    result = evaluate_vggt_omega_online(config)

    assert result["metrics"]["abs_relative_difference"] == pytest.approx(0.0, abs=1e-7)
    assert result["metrics"]["rmse_linear"] == pytest.approx(0.0, abs=1e-7)
    assert result["metrics"]["delta1_acc"] == pytest.approx(1.0)
    assert result["metrics"]["tae"] == pytest.approx(0.0)
    assert set(result["per_sequence"]) == {"dataset_8/keyframe_1", "dataset_9/keyframe_1"}
    assert len(result["teacher_window_scale_diagnostics"]) == 2
    assert result["teacher_cache_used"] is False
    assert result["model_source"] == "vggt_omega_online"
    assert result["input_resolution_hw"] == [8, 12]
    assert result["input_resolutions_hw"] == [[8, 12]]
    assert result["evaluation_resolution_hw"] == [4, 6]


def test_online_teacher_visualizer_exports_native_depth_and_window_cameras(
    monkeypatch, tmp_path
):
    frame_paths = []
    for frame_id in (10, 11):
        path = tmp_path / "frame_{:06d}.png".format(frame_id)
        Image.new("RGB", (8, 6), color=(frame_id, 0, 0)).save(path)
        frame_paths.append(path)

    class Dataset:
        sequences = [
            {
                "sequence_id": "dataset_8/keyframe_1",
                "dataset_id": 8,
                "frame_paths": frame_paths,
            }
        ]

    class FakeFrames:
        def __init__(self, paths, **kwargs):
            self.paths = list(paths)
            self.calls = []

        def __len__(self):
            return len(self.paths)

        def load_indices(self, indices):
            self.calls.append(list(indices))
            return torch.full((len(indices), 3, 6, 8), 0.5)

        def metadata(self):
            return {
                "implementation": "official-fake",
                "observed_input_shape_hw": [6, 8],
            }

    def fake_infer(model, frames, emit, **kwargs):
        kwargs["emit_window"](
            [0, 1],
            {
                "intrinsics": np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
                "extrinsics": np.tile(
                    np.eye(4, dtype=np.float32)[:3], (2, 1, 1)
                ),
            },
        )
        emit(
            0,
            np.ones((len(frames), 6, 8), dtype=np.float32),
            np.tile(np.eye(3, dtype=np.float32), (len(frames), 1, 1)),
        )
        return {
            "output_frame_count": len(frames),
            "window_count": 1,
            "model_input_frame_count": 32,
            "model_forward_seconds": 1.0,
            "timing_scope": "synthetic",
        }

    monkeypatch.setattr(
        "visualization.vggt_omega_video._visualization_dataset",
        lambda config, eval_config, split: Dataset(),
    )
    monkeypatch.setattr("visualization.vggt_omega_video.VGGTOmegaSequenceFrames", FakeFrames)
    monkeypatch.setattr(
        "visualization.vggt_omega_video._evaluation_model",
        lambda *args: nn.Identity().eval(),
    )
    monkeypatch.setattr("visualization.vggt_omega_video.infer_vggt_omega_video", fake_infer)

    config = tmp_path / "visualization.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "teacher": {"pretrained_checkpoint": "unused"},
                "dataset": {},
                "vggt_omega_baseline_vda_evaluation": {
                    "rgb_root": str(tmp_path),
                    "preprocessing": {
                        "image_resolution": 512,
                        "mode": "balanced",
                        "patch_size": 16,
                    },
                },
                "vggt_omega_baseline_visualization": {
                    "output_dir": str(tmp_path / "output"),
                    "point_stride": 2,
                },
            }
        )
    )

    output = export_vggt_omega_video(config)

    assert len(list((output / "depth").glob("*.npy"))) == 2
    assert len(list((output / "pointcloud_local").glob("*.ply"))) == 2
    assert len(list((output / "camera_windows").glob("*.npz"))) == 1
    metadata = yaml.safe_load((output / "metadata.json").read_text())
    assert metadata["source"] == "vggt_omega_online"
    assert metadata["teacher_cache_used"] is False
    assert metadata["model_input_shape_hw"] == [6, 8]
    assert metadata["complete_sequence"] is True
    with np.load(output / "camera_windows" / "window_000000.npz") as window:
        assert window["frame_positions"].tolist() == [0, 1]
        assert window["absolute_frame_ids"].tolist() == [10, 11]


def test_online_visualizer_uses_raw_test_rgb_root(monkeypatch):
    captured = {}

    class Dataset:
        sequences = [
            {"sequence_id": "dataset_8/keyframe_1", "dataset_id": 8},
            {"sequence_id": "dataset_9/keyframe_1", "dataset_id": 9},
        ]

    def make_dataset(dataset_config, split):
        captured.update(dataset_config)
        captured["split"] = split
        return Dataset()

    monkeypatch.setattr("visualization.vggt_omega_video.make_scared_rgb_dataset", make_dataset)
    _visualization_dataset(
        {
            "dataset": {
                "root": "/processed",
                "legacy_scared_root": "/legacy",
                "canonical_root": "/canonical",
            }
        },
        {"rgb_root": "/raw/scared", "frame_source": "auto"},
        "test",
    )

    assert captured["root"] == "/raw/scared"
    assert captured["legacy_scared_root"] == "/raw/scared"
    assert captured["canonical_root"] is None
    assert captured["clip_length"] == 1
    assert captured["sample_stride"] == 1
    assert captured["window_stride"] == 1
    assert captured["drop_incomplete_clip"] is False
    assert captured["highlight"] == {"enabled": False}
    assert captured["split"] == "test"
