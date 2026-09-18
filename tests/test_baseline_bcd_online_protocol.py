from __future__ import annotations

from typing import Any

import torch

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
from utils.config import load_config


BASELINE_ID = "D"
ATTENTION_ENABLED = False
HIGHLIGHT_MODE = "angular_soft_margin"


def _sequence(length: int = 80) -> dict[str, Any]:
    return {
        "dataset_name": "SCARED",
        "dataset_id": 1,
        "keyframe_id": "keyframe_1",
        "sequence_id": "dataset_1/keyframe_1",
        "sequence_length": length,
        "frame_paths": ["frame_{:06d}.png".format(i) for i in range(length)],
        "teacher_frame_paths": [
            "frame_{:06d}.png".format(i) for i in range(length)
        ],
        "absolute_frame_ids": list(range(100, 100 + length)),
        "frame_directory": ".",
        "keyframe_directory": ".",
    }


class _FakeRGBDataset:
    clip_length = 32
    sample_stride = 2
    window_stride = 8

    def __init__(self) -> None:
        sequence = _sequence()
        self.sequences = [sequence]
        self.clips = [
            ClipRecord(
                sequence,
                tuple(step * self.sample_stride for step in range(self.clip_length)),
                0,
            )
        ]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.clips[index]
        absolute = record.sequence["absolute_frame_ids"]
        return {
            "images": torch.zeros(self.clip_length, 3, 4, 6),
            "inpainted_images": torch.full((self.clip_length, 3, 4, 6), 0.5),
            "highlight_masks": torch.zeros(
                self.clip_length, 1, 4, 6, dtype=torch.bool
            ),
            "frame_indices": torch.tensor(
                [absolute[i] for i in record.frame_indices], dtype=torch.long
            ),
            "clip_start": torch.tensor(record.clip_start),
        }


class _FakeTeacherRGB:
    def __init__(self, rgb_dataset: _FakeRGBDataset) -> None:
        self.rgb_dataset = rgb_dataset

    def load_images(self, index: int) -> tuple[torch.Tensor, list[str]]:
        record = self.rgb_dataset.clips[index]
        paths = [
            record.sequence["teacher_frame_paths"][i] for i in record.frame_indices
        ]
        return torch.zeros(32, 3, 4, 6), paths


def test_baseline_protocol_and_ablation_contract() -> None:
    config = load_config("configs/baselines/{}.yaml".format(BASELINE_ID))
    assert _teacher_is_full_online(config)
    assert config["dataset"]["clip_length"] == 32
    assert config["dataset"]["sample_stride"] == 2
    assert config["dataset"]["window_stride"] == 8
    assert config["training"]["epochs"] == 3
    assert config["training"]["resume"] is None
    assert config["training"]["output_dir"] == (
        "./outputs/baseline_{}_online32_s2_3ep_t512x640".format(BASELINE_ID)
    )
    assert config["teacher"]["input_height"] == 512
    assert config["teacher"]["input_width"] == 640
    assert config["teacher"]["raw_cache_root"] is None
    assert config["teacher"]["cache_checkpoint_identity"] is None
    assert config["attention_distill"]["enabled"] is ATTENTION_ENABLED
    assert config["loss"]["highlight_mode"] == HIGHLIGHT_MODE
    assert config["vda_evaluation"]["checkpoint"] == (
        "./outputs/baseline_{}/last.pt".format(BASELINE_ID)
    )


def test_stride_two_dataset_keeps_teacher_and_student_ids_equal_without_cache() -> None:
    dataset = FullOnlineTeacherDistillationDataset(_FakeRGBDataset(), 4, 6)
    dataset.teacher_rgb_dataset = _FakeTeacherRGB(dataset.rgb_dataset)
    sample = dataset[0]
    assert sample["absolute_frame_ids"].tolist() == list(range(100, 164, 2))
    assert torch.equal(
        sample["absolute_frame_ids"], sample["teacher_absolute_frame_ids"]
    )
    assert "teacher" not in sample
    assert not hasattr(dataset, "cache_paths")
    batch = direct_teacher_distillation_collate([sample])
    assert batch["images"].shape == (1, 32, 3, 4, 6)
    assert batch["teacher_images"].shape == (1, 32, 3, 4, 6)


def test_one_teacher_forward_reuses_raw_cache_adaptation_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "FULL_ONLINE_TEACHER_SHAPE", (4, 6))
    monkeypatch.setattr(trainer, "TEACHER_PATCH_SIZE", 2)
    monkeypatch.setattr(trainer, "SUPERVISION_SHAPE", (2, 3))
    calls: list[str] = []

    def fake_adapt(
        predictions: dict[str, torch.Tensor],
        image_shape: tuple[int, int],
        min_depth: float,
        max_depth: float,
    ) -> dict[str, torch.Tensor]:
        calls.append("adapt")
        assert image_shape == (4, 6)
        return {"marker": predictions["depth"].new_tensor(7.0)}

    def fake_canonicalize(adapted: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        calls.append("canonicalize")
        assert float(adapted["marker"]) == 7.0
        batch, frames = 1, 32
        return {
            "depth": torch.ones(batch, frames, 2, 3),
            "confidence": torch.ones(batch, frames, 2, 3),
            "valid_mask": torch.ones(batch, frames, 2, 3, dtype=torch.bool),
            "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, frames, 1, 1),
            "extrinsics": torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, frames, 1, 1)[..., :3, :],
        }

    monkeypatch.setattr(trainer, "adapt_teacher_outputs", fake_adapt)
    monkeypatch.setattr(trainer, "canonicalize_teacher_outputs", fake_canonicalize)

    class _Teacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prediction_forward_count = 0

        def forward(self, images: torch.Tensor) -> dict[str, Any]:
            self.prediction_forward_count += 1
            batch, frames, _, height, width = images.shape
            output: dict[str, Any] = {
                "pose_enc": torch.zeros(batch, frames, 9),
                "depth": torch.ones(batch, frames, height, width, 1),
                "depth_conf": torch.ones(batch, frames, height, width, 1),
            }
            if ATTENTION_ENABLED:
                output["attention"] = {
                    4: {
                        "q": torch.ones(batch, frames, 1, 6, 2),
                        "k": torch.ones(batch, frames, 1, 6, 2),
                        "metadata": {"num_frames": frames},
                    }
                }
            return output

    teacher = _Teacher().eval().requires_grad_(False)
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
    assert calls == ["adapt", "canonicalize"]
    assert teacher.prediction_forward_count == audit["teacher_forward_count"] == 1
    assert audit["teacher_patch_grid"] == [2, 3]
    assert audit["teacher_q_shape"] is None
    assert audit["teacher_forward_ms"] >= 0.0
    assert audit["teacher_adapt_ms"] >= 0.0
    assert audit["teacher_canonicalize_ms"] >= 0.0
    assert torch.equal(supervision["absolute_frame_ids"], ids)
    assert (attention is not None) is ATTENTION_ENABLED

def test_same_forward_attention_consumer_uses_existing_qk_without_teacher_call() -> None:
    teacher_features = {
        4: {
            "q": torch.ones(1, 32, 1, 6, 2),
            "k": torch.ones(1, 32, 1, 6, 2),
            "metadata": {"num_frames": 32},
        }
    }
    student_q = torch.ones(1, 32, 1, 6, 2, requires_grad=True)
    student_features = {
        5: {
            "q": student_q,
            "k": torch.ones(1, 32, 1, 6, 2, requires_grad=True),
            "metadata": {"num_frames": 32},
        }
    }

    class _Loss:
        def __call__(self, teacher: dict, student: dict) -> tuple[torch.Tensor, dict]:
            assert teacher is teacher_features
            assert student is student_features
            return student[5]["q"].sum(), {"loss/attention": 0.0}

    loss, logs = trainer._compute_attention_loss_from_online_features(
        teacher_features, student_features, _Loss()
    )
    assert loss.requires_grad
    assert logs["stats/online_teacher_chunks"] == 1.0
