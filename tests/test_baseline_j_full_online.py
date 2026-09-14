from __future__ import annotations

from types import SimpleNamespace
from copy import deepcopy

import torch

import models.teacher.output_adapter as output_adapter
import trainers.direct_teacher_distillation_trainer as trainer
from datasets.direct_teacher_distillation_dataset import (
    FullOnlineTeacherDistillationDataset,
    direct_teacher_distillation_collate,
)
from datasets.scared_dataset import ClipRecord
from trainers.direct_teacher_distillation_trainer import (
    _forward_full_online_teacher,
    _teacher_is_full_online,
)
from losses.direct_teacher_distillation_loss import compute_camera_distillation_loss
from losses.direct_teacher_distillation_loss import CameraLossWeights
from losses.direct_teacher_distillation_loss import DirectTeacherDistillationLoss
from losses.attention_distillation_loss import CrossFrameAttentionDistillationLoss
from utils.config import load_config


def _sequence(length: int = 48) -> dict:
    return {
        "dataset_name": "SCARED",
        "dataset_id": 1,
        "keyframe_id": "keyframe_1",
        "sequence_id": "dataset_1/keyframe_1",
        "sequence_length": length,
        "frame_paths": ["frame_{:06d}.png".format(index) for index in range(length)],
        "teacher_frame_paths": [
            "frame_{:06d}.png".format(index) for index in range(length)
        ],
        "absolute_frame_ids": list(range(100, 100 + length)),
        "frame_directory": ".",
        "keyframe_directory": ".",
    }


class _FakeRGBDataset:
    clip_length = 32
    sample_stride = 1
    window_stride = 8

    def __init__(self) -> None:
        sequence = _sequence()
        self.sequences = [sequence]
        self.clips = [
            ClipRecord(sequence, tuple(range(start, start + 32)), start)
            for start in (0, 8, 16)
        ]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict:
        record = self.clips[index]
        absolute = record.sequence["absolute_frame_ids"]
        return {
            "images": torch.zeros(32, 3, 4, 6),
            "inpainted_images": torch.full((32, 3, 4, 6), 0.5),
            "highlight_masks": torch.zeros(32, 1, 4, 6, dtype=torch.bool),
            "frame_indices": torch.tensor([absolute[i] for i in record.frame_indices]),
            "clip_start": torch.tensor(record.clip_start),
        }


def test_j_config_inherits_e_losses_models_and_evaluation() -> None:
    baseline_e = load_config("configs/baselines/E.yaml")
    baseline_j = load_config("configs/baselines/J.yaml")
    assert baseline_j["loss"] == baseline_e["loss"]
    assert baseline_j["student"] == baseline_e["student"]
    assert baseline_j["evaluation"] == baseline_e["evaluation"] == {"protocol": "vda"}
    mathematical_attention_fields = (
        "teacher_layers",
        "student_layers",
        "attention_type",
        "spatial_alignment",
        "common_grid",
        "head_aggregation",
        "divergence",
        "temperature_teacher",
        "temperature_student",
        "weight",
        "frame_offsets",
        "query_chunk_size",
        "eps",
    )
    for field in mathematical_attention_fields:
        assert baseline_j["attention_distill"][field] == baseline_e["attention_distill"][field]
    teacher_architecture_fields = (
        "variant",
        "pretrained_checkpoint",
        "frozen",
        "freeze_backbone",
        "freeze_heads",
        "attention_layers",
    )
    for field in teacher_architecture_fields:
        assert baseline_j["teacher"][field] == baseline_e["teacher"][field]
    assert baseline_j["dataset"]["clip_length"] == 32
    assert baseline_j["dataset"]["sample_stride"] == 1
    assert baseline_j["dataset"]["window_stride"] == 8
    assert baseline_j["dataloader"]["batch_size"] == 1
    assert baseline_j["attention_distill"]["pair_chunk_size"] == 2
    assert baseline_j["attention_distill"]["teacher_probability_outside_checkpoint"] is True
    assert _teacher_is_full_online(baseline_j)
    assert baseline_j["teacher"]["raw_cache_root"] is None


def test_j_dataset_is_ordered_same_sequence_and_cache_free() -> None:
    dataset = FullOnlineTeacherDistillationDataset(_FakeRGBDataset(), 4, 6)

    class _FakeTeacherRGB:
        def load_images(self, index: int):
            record = dataset.rgb_dataset.clips[index]
            paths = [record.sequence["teacher_frame_paths"][i] for i in record.frame_indices]
            return torch.zeros(32, 3, 4, 6), paths

    dataset.teacher_rgb_dataset = _FakeTeacherRGB()
    sample = dataset[1]
    assert sample["sequence_id"] == sample["teacher_sequence_id"]
    assert sample["clip_start"].item() == 8
    assert sample["absolute_frame_ids"].tolist() == list(range(108, 140))
    assert torch.equal(
        sample["absolute_frame_ids"], sample["teacher_absolute_frame_ids"]
    )
    assert "teacher" not in sample
    assert not hasattr(dataset, "cache_paths")

    batch = direct_teacher_distillation_collate([sample])
    assert batch["images"].shape == (1, 32, 3, 4, 6)
    assert batch["teacher_images"].shape == (1, 32, 3, 4, 6)
    assert "teacher" not in batch
    assert torch.equal(
        batch["absolute_frame_ids"], batch["teacher_absolute_frame_ids"]
    )


def test_j_dataset_builder_never_enters_cache_builder(monkeypatch) -> None:
    config = load_config("configs/baselines/J.yaml")
    monkeypatch.setattr(trainer, "make_scared_rgb_dataset", lambda *_args: _FakeRGBDataset())

    def forbidden_cache_builder(*_args, **_kwargs):
        raise AssertionError("Baseline J must not inspect or build a Teacher cache dataset")

    monkeypatch.setattr(trainer, "make_teacher_cache_rgb_dataset", forbidden_cache_builder)
    dataset = trainer._build_dataset(config, "train")
    assert isinstance(dataset, FullOnlineTeacherDistillationDataset)
    assert len(dataset) == 3


def _install_fake_pose_decoder(monkeypatch) -> None:
    def encoding_to_camera(_pose, image_shape):
        batch, frames = _pose.shape[:2]
        height, width = image_shape
        extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, frames, 1, 1)[..., :3, :]
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, frames, 1, 1)
        intrinsics[..., 0, 0] = width
        intrinsics[..., 1, 1] = height
        intrinsics[..., 0, 2] = width / 2
        intrinsics[..., 1, 2] = height / 2
        return extrinsics, intrinsics

    monkeypatch.setattr(
        output_adapter.importlib,
        "import_module",
        lambda _name: SimpleNamespace(encoding_to_camera=encoding_to_camera),
    )


def test_full_online_teacher_runs_once_and_returns_all_supervision(monkeypatch) -> None:
    _install_fake_pose_decoder(monkeypatch)

    class _FakeTeacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, images: torch.Tensor) -> dict:
            assert not torch.is_grad_enabled()
            self.calls += 1
            batch, frames, _, height, width = images.shape
            marker = float(self.calls + 1)
            return {
                "pose_enc": torch.zeros(batch, frames, 9),
                "depth": torch.full((batch, frames, height, width, 1), marker),
                "depth_conf": torch.full((batch, frames, height, width, 1), 2.0),
                "attention": {
                    4: {
                        "q": torch.full((batch, frames, 1, height * width, 2), marker),
                        "k": torch.full((batch, frames, 1, height * width, 2), marker),
                        "metadata": {
                            "num_frames": frames,
                            "patch_grid_h": height,
                            "patch_grid_w": width,
                            "patch_size": 1,
                        },
                    }
                },
            }

    teacher_model = _FakeTeacher().eval()
    teacher_model.requires_grad_(False)
    ids = torch.arange(32).reshape(1, 32)
    with torch.no_grad():
        supervision, attention, audit = _forward_full_online_teacher(
            teacher_model,
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
    assert teacher_model.calls == audit["teacher_forward_count"] == 1
    assert supervision["depth"].shape == (1, 32, 2, 3)
    assert supervision["confidence"].shape == (1, 32, 2, 3)
    assert supervision["intrinsics"].shape == (1, 32, 3, 3)
    assert supervision["extrinsics"].shape == (1, 32, 3, 4)
    assert torch.equal(supervision["absolute_frame_ids"], ids)
    assert attention[4]["q"].shape[:2] == (1, 32)
    assert torch.all(supervision["depth"] == 2.0)
    assert torch.all(attention[4]["q"] == 2.0)
    assert not any(value.requires_grad for value in supervision.values() if isinstance(value, torch.Tensor))


def test_online_depth_adapter_does_not_add_scale_alignment(monkeypatch) -> None:
    _install_fake_pose_decoder(monkeypatch)
    predictions = {
        "pose_enc": torch.zeros(1, 32, 9),
        "depth": torch.full((1, 32, 4, 6, 1), 7.0),
        "depth_conf": torch.full((1, 32, 4, 6, 1), 2.0),
    }
    output = output_adapter.adapt_teacher_distillation_outputs(
        predictions, (4, 6), (2, 3)
    )
    assert torch.all(output["depth"] == 7.0)
    assert torch.all(output["intrinsics"][..., 0, 0] == 3.0)
    assert torch.all(output["intrinsics"][..., 1, 1] == 2.0)


def test_camera_loss_shape_contract_accepts_32_frames_without_math_change() -> None:
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 32, 1, 1)
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 32, 1, 1)[..., :3, :]
    loss, diagnostics = compute_camera_distillation_loss(
        intrinsics,
        extrinsics,
        intrinsics.clone(),
        extrinsics.clone(),
        (448, 560),
        CameraLossWeights(),
    )
    assert torch.isfinite(loss)
    assert diagnostics["intrinsics"].item() == 0.0


def test_all_existing_e_losses_support_32_frames_and_student_backward() -> None:
    batch_size, frames, height, width = 1, 32, 5, 7
    student_depth = torch.full(
        (batch_size, frames, height, width), 2.2, requires_grad=True
    )
    student_points = torch.randn(
        batch_size, frames, height, width, 3, requires_grad=True
    )
    student_intrinsics = (
        torch.eye(3).reshape(1, 1, 3, 3).repeat(batch_size, frames, 1, 1)
    )
    student_intrinsics[..., 0, 0] = width * 0.9
    student_intrinsics[..., 1, 1] = height * 0.9
    student_intrinsics.requires_grad_(True)
    student_extrinsics = (
        torch.eye(4).reshape(1, 1, 4, 4).repeat(batch_size, frames, 1, 1)[..., :3, :]
    )
    student_extrinsics[..., 0, 3] = torch.arange(frames) * 0.011
    student_extrinsics.requires_grad_(True)

    teacher_intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch_size, frames, 1, 1)
    teacher_intrinsics[..., 0, 0] = width
    teacher_intrinsics[..., 1, 1] = height
    teacher_extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch_size, frames, 1, 1)[..., :3, :]
    teacher_extrinsics[..., 0, 3] = torch.arange(frames) * 0.01
    frame_ids = torch.arange(frames).reshape(1, frames)
    prediction = {
        "depth": student_depth,
        "xyz_local": student_points,
        "intrinsics": student_intrinsics,
        "extrinsics": student_extrinsics,
    }
    batch = {
        "absolute_frame_ids": frame_ids,
        "clip_start": torch.tensor([0]),
        "clean_images": torch.rand(batch_size, frames, 3, height, width),
        "highlight_masks": torch.zeros(
            batch_size, frames, 1, height, width, dtype=torch.bool
        ),
        "teacher": {
            "depth": torch.full((batch_size, frames, height, width), 2.0),
            "confidence": torch.ones(batch_size, frames, height, width),
            "valid_mask": torch.ones(
                batch_size, frames, height, width, dtype=torch.bool
            ),
            "intrinsics": teacher_intrinsics,
            "extrinsics": teacher_extrinsics,
            "absolute_frame_ids": frame_ids.clone(),
            "clip_start": torch.tensor([0]),
        },
    }
    criterion = DirectTeacherDistillationLoss(
        load_config("configs/baselines/J.yaml")["loss"]
    )
    loss, logs = criterion(prediction, batch)
    loss.backward()
    for name in (
        "loss/depth_weighted",
        "loss/camera_weighted",
        "loss/highlight_weighted",
        "loss/smooth_weighted",
    ):
        assert name in logs
    assert student_depth.grad is not None and torch.isfinite(student_depth.grad).all()
    assert student_points.grad is not None and torch.isfinite(student_points.grad).all()
    assert student_intrinsics.grad is not None and torch.isfinite(student_intrinsics.grad).all()
    assert student_extrinsics.grad is not None and torch.isfinite(student_extrinsics.grad).all()


def test_j_e_optimized_attention_matches_legacy_loss_and_gradients() -> None:
    config = deepcopy(load_config("configs/baselines/J.yaml")["attention_distill"])
    config.update(
        {
            "teacher_layers": [4],
            "student_layers": [5],
            "query_chunk_size": 2,
        }
    )
    generator = torch.Generator().manual_seed(19)
    shape = (1, 6, 2, 6, 3)
    teacher = {
        "q": torch.randn(shape, generator=generator),
        "k": torch.randn(shape, generator=generator),
        "metadata": {
            "patch_grid_h": 2,
            "patch_grid_w": 3,
            "patch_size": 1,
            "image_height": 2,
            "image_width": 3,
        },
    }
    student_q = torch.randn(shape, generator=generator, requires_grad=True)
    student_k = torch.randn(shape, generator=generator, requires_grad=True)
    student = {
        "q": student_q,
        "k": student_k,
        "metadata": deepcopy(teacher["metadata"]),
    }
    optimized = CrossFrameAttentionDistillationLoss(config)._layer_loss(teacher, student)
    optimized_gradients = torch.autograd.grad(optimized, (student_q, student_k))

    legacy_config = deepcopy(config)
    legacy_config["pair_chunk_size"] = 1
    legacy_config["teacher_probability_outside_checkpoint"] = False
    legacy_q = student_q.detach().clone().requires_grad_(True)
    legacy_k = student_k.detach().clone().requires_grad_(True)
    legacy = CrossFrameAttentionDistillationLoss(legacy_config)._layer_loss(
        teacher,
        {"q": legacy_q, "k": legacy_k, "metadata": deepcopy(teacher["metadata"])},
    )
    legacy_gradients = torch.autograd.grad(legacy, (legacy_q, legacy_k))
    torch.testing.assert_close(optimized, legacy, rtol=1.0e-5, atol=1.0e-6)
    for optimized_gradient, legacy_gradient in zip(
        optimized_gradients, legacy_gradients
    ):
        torch.testing.assert_close(
            optimized_gradient, legacy_gradient, rtol=1.0e-5, atol=1.0e-6
        )
