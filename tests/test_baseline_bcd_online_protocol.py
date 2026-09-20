from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

import models.teacher.output_adapter as output_adapter
from datasets.transforms import load_teacher_rgb_tensor
from inference.student_video import sequence_frames
from losses.attention_distillation_loss import SpatialTokenAligner
from trainers.direct_teacher_distillation_trainer import (
    _forward_full_online_teacher,
    _teacher_is_full_online,
)
from utils.config import load_config


BASELINE_ID = "C"
ATTENTION_ENABLED = True
HIGHLIGHT_MODE = "legacy"


def _install_fake_pose_decoder(monkeypatch) -> None:
    def encoding_to_camera(pose: torch.Tensor, image_shape: tuple[int, int]):
        batch, frames = pose.shape[:2]
        extrinsics = (
            torch.eye(4).reshape(1, 1, 4, 4)
            .repeat(batch, frames, 1, 1)[..., :3, :]
        )
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, frames, 1, 1)
        intrinsics[..., 0, 0] = image_shape[1]
        intrinsics[..., 1, 1] = image_shape[0]
        return extrinsics, intrinsics

    monkeypatch.setattr(
        output_adapter.importlib,
        "import_module",
        lambda _name: SimpleNamespace(encoding_to_camera=encoding_to_camera),
    )


def test_config_keeps_only_the_intended_ablation() -> None:
    config = load_config("configs/baselines/{}.yaml".format(BASELINE_ID))
    assert _teacher_is_full_online(config)
    assert (
        config["dataset"]["clip_length"],
        config["dataset"]["sample_stride"],
        config["dataset"]["window_stride"],
    ) == (32, 2, 8)
    assert (config["teacher"]["input_height"], config["teacher"]["input_width"]) == (
        512,
        640,
    )
    assert (config["student"]["image_height"], config["student"]["image_width"]) == (
        448,
        560,
    )
    assert (config["inference"]["image_height"], config["inference"]["image_width"]) == (448, 560)
    assert (448 // config["student"]["patch_size"], 560 // config["student"]["patch_size"]) == (32, 40)
    assert (config["vda_evaluation"]["evaluation_height"],
            config["vda_evaluation"]["evaluation_width"]) == (224, 280)
    assert config["vda_evaluation"]["tae"]["enabled"] is False
    assert config["training"]["epochs"] == 3
    assert config["training"]["resume"] is None
    assert config["vda_evaluation"]["checkpoint"].startswith(
        config["training"]["output_dir"]
    )
    assert config["visualization"]["output_dir"].startswith(
        config["training"]["output_dir"]
    )
    assert config["attention_distill"]["enabled"] is ATTENTION_ENABLED
    assert config["attention_distill"]["weight"] == (0.1 if ATTENTION_ENABLED else 0.0)
    assert config["loss"]["highlight_mode"] == HIGHLIGHT_MODE
    if ATTENTION_ENABLED:
        assert config["attention_distill"]["pair_chunk_size"] == 2
        assert config["attention_distill"]["teacher_probability_outside_checkpoint"] is True


def test_vda_loader_uses_explicit_448x560_inference_size(tmp_path) -> None:
    config = load_config("configs/baselines/C.yaml")
    image_path = tmp_path / "frame_000000.png"
    Image.new("RGB", (6, 4), color=(64, 128, 192)).save(image_path)
    frames = sequence_frames({"frame_paths": [image_path]}, config["dataset"],
                             raw_rgb=True, inference_config=config["inference"])
    assert frames[0].shape == (3, 448, 560)


def test_equal_attention_grids_use_identity() -> None:
    value = torch.randn(1, 2, 3, 6, 4)
    aligned = SpatialTokenAligner((2, 3), (2, 3))(value)
    assert aligned is value


def test_teacher_rgb_is_resized_before_tensor_conversion(tmp_path) -> None:
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.new("RGB", (1280, 1024), color=(64, 128, 192)).save(path)
    image = load_teacher_rgb_tensor(path, output_height=4, output_width=6)
    assert image.shape == (3, 4, 6)
    assert image.dtype == torch.float32


def test_one_teacher_forward_uses_fast_dense_adapter_without_xyz(monkeypatch) -> None:
    _install_fake_pose_decoder(monkeypatch)

    class FakeTeacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prediction_forward_count = 0

        def forward(self, images: torch.Tensor) -> dict:
            self.prediction_forward_count += 1
            batch, frames, _, height, width = images.shape
            result = {
                "pose_enc": torch.zeros(batch, frames, 9),
                "depth": torch.ones(batch, frames, height, width, 1),
                "depth_conf": torch.ones(batch, frames, height, width, 1),
            }
            if ATTENTION_ENABLED:
                result["attention"] = {
                    4: {
                        "q": torch.ones(batch, frames, 1, 6, 2),
                        "k": torch.ones(batch, frames, 1, 6, 2),
                        "metadata": {"num_frames": frames},
                    }
                }
            return result

    teacher = FakeTeacher().eval().requires_grad_(False)
    ids = torch.arange(0, 64, 2).reshape(1, 32)
    with torch.no_grad():
        supervision, attention, audit = _forward_full_online_teacher(
            teacher,
            torch.zeros(1, 32, 3, 4, 6),
            ids,
            torch.tensor([0]),
            ["sequence"],
            device=torch.device("cpu"),
            amp_enabled=False,
            amp_dtype=torch.float16,
            teacher_input_shape=(4, 6),
            supervision_shape=(2, 3),
            min_depth=0.1,
            max_depth=150.0,
            minimum_valid_fraction=0.001,
        )

    assert teacher.prediction_forward_count == audit["teacher_forward_count"] == 1
    assert set(supervision).isdisjoint({"xyz_local", "xyz_global"})
    assert supervision["depth"].shape == (1, 32, 2, 3)
    assert torch.equal(supervision["absolute_frame_ids"], ids)
    assert (attention is not None) is ATTENTION_ENABLED
    if attention is None:
        assert audit["teacher_attention_layers"] == []
