from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from cache.generate_crossclip_teacher_cache import attention_cache_bytes_per_clip
from datasets.crossclip_teacher_dataset import (
    ATTENTION_CACHE_SCHEMA_VERSION,
    attention_cache_key,
    validate_attention_teacher_cache,
)
from losses.attention_distillation_loss import (
    AttentionDistillationConfig,
    CrossFrameAttentionDistillationLoss,
    SpatialTokenAligner,
    patch_overlap_matrix,
)
from trainers.direct_teacher_distillation_trainer import (
    _check_resume_contract,
    _compute_online_teacher_attention_loss,
)
from models.teacher.vggt_omega_wrapper import VGGTOmegaTeacher
from utils.config import load_config
from utils.checkpoint import DIRECT_TEACHER_DISTILLATION_PROTOCOL


LAYERS = ((4, 5), (11, 7), (17, 9), (23, 11))


def _config(enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "teacher_source": "online",
        "online_teacher_batch_size": 1,
        "teacher_output_dtype": "float16",
        "teacher_layers": [item[0] for item in LAYERS],
        "student_layers": [item[1] for item in LAYERS],
        "attention_type": "cross_frame_global",
        "spatial_alignment": "patch_overlap",
        "common_grid": "student",
        "head_aggregation": "mean",
        "divergence": "js",
        "temperature_teacher": 1.0,
        "temperature_student": 1.0,
        "weight": 0.1 if enabled else 0.0,
        "frame_offsets": [-1, 1],
        "query_chunk_size": 1,
        "eps": 1.0e-6,
    }


def _feature(
    layer: int,
    grid: tuple[int, int],
    heads: int,
    head_dim: int,
    *,
    requires_grad: bool,
    batch_size: int = 1,
    num_frames: int = 3,
) -> dict:
    generator = torch.Generator().manual_seed(layer)
    shape = (batch_size, num_frames, heads, grid[0] * grid[1], head_dim)
    q = torch.randn(shape, generator=generator, requires_grad=requires_grad)
    k = torch.randn(shape, generator=generator, requires_grad=requires_grad)
    return {
        "q": q,
        "k": k,
        "metadata": {
            "layer_index": layer,
            "num_frames": num_frames,
            "patch_grid_h": grid[0],
            "patch_grid_w": grid[1],
            "patch_size": 1,
            "image_height": grid[0],
            "image_width": grid[1],
            "num_heads": heads,
            "head_dim": head_dim,
        },
    }


def test_experiment_b_inherits_baseline_and_changes_attention_paths() -> None:
    baseline = load_config("configs/vggtoda3.yaml")
    attention = load_config("configs/vggtoda3_attention_distill.yaml")
    assert baseline["attention_distill"]["enabled"] is False
    assert baseline["attention_distill"]["weight"] == 0.0
    assert attention["attention_distill"]["enabled"] is True
    assert attention["attention_distill"]["weight"] == pytest.approx(0.1)
    assert attention["teacher"]["save_attention"] is False
    assert attention["teacher"]["raw_cache_root"] == baseline["teacher"]["raw_cache_root"]
    assert attention["attention_distill"]["teacher_source"] == "online"
    assert attention["attention_distill"]["online_teacher_batch_size"] == 1
    for section in ("dataset", "student", "loss"):
        assert attention[section] == baseline[section]
    assert baseline["dataloader"]["batch_size"] == 16
    assert attention["dataloader"]["batch_size"] == 4
    assert attention["dataloader"]["drop_last"] == baseline["dataloader"]["drop_last"]
    assert attention["dataloader"]["num_workers"] == 0


def test_legacy_disabled_attention_checkpoint_remains_resumable() -> None:
    baseline = load_config("configs/vggtoda3.yaml")
    legacy_config = deepcopy(baseline)
    legacy_config.pop("attention_distill")
    checkpoint = {
        "objective_protocol": DIRECT_TEACHER_DISTILLATION_PROTOCOL,
        "config": legacy_config,
    }

    class FakeModel:
        def __init__(self) -> None:
            self.checked = False

        def assert_trainability_contract(self) -> None:
            self.checked = True

    model = FakeModel()
    _check_resume_contract(checkpoint, baseline, model)
    assert model.checked


def test_patch_overlap_alignment_uses_area_and_only_changes_tokens() -> None:
    projection = patch_overlap_matrix((2, 4), (1, 2))
    assert projection.shape == (2, 8)
    torch.testing.assert_close(projection.sum(dim=1), torch.ones(2))
    assert sorted(projection[0][projection[0] > 0].tolist()) == pytest.approx([0.25] * 4)
    values = torch.arange(8.0).reshape(1, 1, 1, 8, 1)
    aligned = SpatialTokenAligner((2, 4), (1, 2))(values)
    assert aligned.shape == (1, 1, 1, 2, 1)
    torch.testing.assert_close(aligned.flatten(), torch.tensor([2.5, 4.5]))
    assert "projection" not in dict(SpatialTokenAligner((64, 80), (32, 40)).named_buffers())


def test_separable_noninteger_overlap_matches_dense_reference() -> None:
    source_grid, target_grid = (3, 5), (2, 4)
    values = torch.randn(2, 3, 2, 15, 4)
    dense = torch.einsum(
        "sn,bfhnd->bfhsd",
        patch_overlap_matrix(source_grid, target_grid),
        values,
    )
    separable = SpatialTokenAligner(source_grid, target_grid)(values)
    torch.testing.assert_close(separable, dense)


def test_four_layer_js_loss_is_finite_positive_and_backpropagates_only_student() -> None:
    teacher = {
        teacher_layer: _feature(teacher_layer, (2, 4), 3, 4, requires_grad=True)
        for teacher_layer, _ in LAYERS
    }
    student = {
        student_layer: _feature(student_layer, (1, 2), 2, 3, requires_grad=True)
        for _, student_layer in LAYERS
    }
    loss_function = CrossFrameAttentionDistillationLoss(_config())
    loss, logs = loss_function(teacher, student)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert set(key for key in logs if key.startswith("loss/attn_t")) == {
        "loss/attn_t4_s5",
        "loss/attn_t11_s7",
        "loss/attn_t17_s9",
        "loss/attn_t23_s11",
    }
    for feature in teacher.values():
        assert feature["q"].grad is None and feature["k"].grad is None
    for feature in student.values():
        assert feature["q"].grad is not None
        assert feature["k"].grad is not None
        assert feature["q"].grad.abs().sum() > 0
        assert feature["k"].grad.abs().sum() > 0


def test_online_teacher_attention_is_chunked_detached_and_backpropagates_student() -> None:
    config = AttentionDistillationConfig.from_mapping(_config())
    student = {
        student_layer: _feature(
            student_layer,
            (1, 2),
            2,
            3,
            requires_grad=True,
            batch_size=2,
            num_frames=16,
        )
        for _, student_layer in LAYERS
    }

    class FakeTeacher:
        def __init__(self) -> None:
            self.batch_sizes = []
            self.requires_grad_flags = []

        def forward_attention(self, images: torch.Tensor) -> dict:
            batch_size = int(images.shape[0])
            self.batch_sizes.append(batch_size)
            features = {
                teacher_layer: _feature(
                    teacher_layer,
                    (2, 4),
                    3,
                    4,
                    requires_grad=False,
                    batch_size=batch_size,
                    num_frames=16,
                )
                for teacher_layer, _ in LAYERS
            }
            self.requires_grad_flags.extend(
                value[name].requires_grad
                for value in features.values()
                for name in ("q", "k")
            )
            return features

    teacher = FakeTeacher()
    teacher_images = torch.zeros(1).expand(2, 16, 3, 1024, 1280)
    loss, logs = _compute_online_teacher_attention_loss(
        teacher,
        teacher_images,
        student,
        CrossFrameAttentionDistillationLoss(config),
        config,
        torch.device("cpu"),
        False,
        torch.float32,
    )
    assert torch.isfinite(loss) and loss.item() > 0.0
    assert teacher.batch_sizes == [1, 1]
    assert not any(teacher.requires_grad_flags)
    assert logs["stats/online_teacher_chunks"] == 2.0
    loss.backward()
    assert all(
        feature[name].grad is not None and feature[name].grad.abs().sum() > 0
        for feature in student.values()
        for name in ("q", "k")
    )


def test_teacher_attention_forward_skips_prediction_heads_and_requires_no_grad() -> None:
    class FakeAggregator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.cached_layer_indices = {4, 11, 17, 23}
            self.observed_cached_layers = []

        def forward(self, images: torch.Tensor):
            self.calls += 1
            self.observed_cached_layers.append(set(self.cached_layer_indices))
            return [images.mean()], 0

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.aggregator = FakeAggregator()

        def forward(self, images: torch.Tensor):
            raise AssertionError("full Teacher heads must not execute")

    class FakeCapture:
        def begin(self, images: torch.Tensor) -> None:
            self.batch = int(images.shape[0])

        def take(self) -> dict:
            return {4: {"batch": self.batch}}

    model = FakeModel()
    teacher = VGGTOmegaTeacher(
        model, attention_capture=FakeCapture(), attention_only=True
    ).freeze_for_inference()
    with pytest.raises(RuntimeError, match="cannot run prediction heads"):
        teacher(torch.zeros(1, 2, 3, 4, 4))
    with pytest.raises(RuntimeError, match="gradients disabled"):
        teacher.forward_attention(torch.zeros(1, 2, 3, 4, 4))
    with torch.no_grad():
        output = teacher.forward_attention(torch.zeros(1, 2, 3, 4, 4))
    assert output == {4: {"batch": 1}}
    assert model.aggregator.calls == 1
    assert model.aggregator.observed_cached_layers == [set()]
    assert model.aggregator.cached_layer_indices == {4, 11, 17, 23}


def test_attention_cache_schema_validates_shapes_dtype_and_finiteness(tmp_path) -> None:
    arrays = {
        "attention_schema_version": np.asarray(ATTENTION_CACHE_SCHEMA_VERSION),
        "attention_num_frames": np.asarray(16),
        "attention_patch_grid_h": np.asarray(2),
        "attention_patch_grid_w": np.asarray(2),
        "attention_patch_size": np.asarray(16),
        "attention_image_height": np.asarray(32),
        "attention_image_width": np.asarray(32),
        "attention_dtype": np.asarray("float16"),
        "attention_qk_stage": np.asarray("post_qk_norm_no_rope"),
    }
    for layer, _ in LAYERS:
        arrays[attention_cache_key(layer, "q")] = np.ones((16, 2, 4, 3), np.float16)
        arrays[attention_cache_key(layer, "k")] = np.ones((16, 2, 4, 3), np.float16)
        arrays[attention_cache_key(layer, "layer_index")] = np.asarray(layer)
        arrays[attention_cache_key(layer, "num_heads")] = np.asarray(2)
        arrays[attention_cache_key(layer, "head_dim")] = np.asarray(3)
    path = tmp_path / "attention.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as cache:
        validate_attention_teacher_cache(cache, tuple(item[0] for item in LAYERS))
    arrays[attention_cache_key(4, "q")][0, 0, 0, 0] = np.nan
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as cache:
        with pytest.raises(RuntimeError, match="NaN or Inf"):
            validate_attention_teacher_cache(cache, tuple(item[0] for item in LAYERS))


def test_old_cache_fails_closed_when_attention_is_required(tmp_path) -> None:
    path = tmp_path / "old.npz"
    np.savez(path, depth=np.ones(1, np.float32))
    with np.load(path, allow_pickle=False) as cache:
        with pytest.raises(RuntimeError, match="Regenerate teacher cache"):
            validate_attention_teacher_cache(cache)


def test_native_teacher_attention_cache_size_estimate_is_explicit() -> None:
    value = attention_cache_bytes_per_clip(4, 16, 64 * 80, 16, 64, 2)
    assert value == 1_342_177_280
    assert value / 1024**3 == pytest.approx(1.25)


def test_attention_config_rejects_non_global_or_silent_disabled_weight() -> None:
    valid = AttentionDistillationConfig.from_mapping(_config())
    assert valid.teacher_layers == (4, 11, 17, 23)
    invalid = _config(enabled=False)
    invalid["weight"] = 0.1
    with pytest.raises(ValueError, match="Disabled"):
        AttentionDistillationConfig.from_mapping(invalid)
