from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import losses.attention_distillation_loss as attention_loss_module
from losses.attention_distillation_loss import (
    AttentionDistillationConfig,
    CrossFrameAttentionDistillationLoss,
    _directed_frame_pair_indices,
    _head_mean_attention_pairs,
    _teacher_pair_probability,
)
from utils.config import load_config


def _config(*, pair_chunk_size: int = 1, teacher_outside: bool = False) -> dict:
    return {
        "enabled": True,
        "teacher_source": "online",
        "online_teacher_batch_size": 1,
        "teacher_output_dtype": "float16",
        "teacher_layers": [4],
        "student_layers": [5],
        "attention_type": "cross_frame_global",
        "spatial_alignment": "patch_overlap",
        "common_grid": "student",
        "head_aggregation": "mean",
        "divergence": "js",
        "temperature_teacher": 1.0,
        "temperature_student": 1.0,
        "weight": 0.1,
        "frame_offsets": [-1, 1],
        "query_chunk_size": 2,
        "eps": 1.0e-6,
        "pair_chunk_size": pair_chunk_size,
        "teacher_probability_outside_checkpoint": teacher_outside,
    }


def _feature(
    seed: int,
    grid: tuple[int, int],
    heads: int,
    head_dim: int,
    *,
    requires_grad: bool,
    batch_size: int = 2,
    frames: int = 4,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    shape = (batch_size, frames, heads, grid[0] * grid[1], head_dim)
    return {
        "q": torch.randn(shape, generator=generator, requires_grad=requires_grad),
        "k": torch.randn(shape, generator=generator, requires_grad=requires_grad),
        "metadata": {
            "patch_grid_h": grid[0],
            "patch_grid_w": grid[1],
            "patch_size": 1,
            "image_height": grid[0],
            "image_width": grid[1],
        },
    }


def test_only_baseline_e_opts_into_optimized_relation_compute() -> None:
    e_config = AttentionDistillationConfig.from_mapping(
        load_config("configs/baselines/E.yaml")["attention_distill"]
    )
    assert e_config.pair_chunk_size == 2
    assert e_config.teacher_probability_outside_checkpoint is True

    for baseline in "ABCD FG".replace(" ", ""):
        config = AttentionDistillationConfig.from_mapping(
            load_config("configs/baselines/{}.yaml".format(baseline))["attention_distill"]
        )
        assert config.pair_chunk_size == 1
        assert config.teacher_probability_outside_checkpoint is False


def test_sixteen_frames_keep_all_thirty_directed_neighbor_pairs() -> None:
    sources, targets = _directed_frame_pair_indices(
        16, (-1, 1), torch.device("cpu")
    )
    expected = [
        (source, source + offset)
        for source in range(16)
        for offset in (-1, 1)
        if 0 <= source + offset < 16
    ]
    assert list(zip(sources.tolist(), targets.tolist())) == expected
    assert len(expected) == 30
    assert set(expected) == {
        (source, target)
        for source in range(16)
        for target in range(16)
        if abs(source - target) == 1
    }
    pair_batches = [expected[start : start + 2] for start in range(0, 30, 2)]
    assert len(pair_batches) == 15
    assert all(len(batch) == 2 for batch in pair_batches)
    assert pair_batches[0] == [(0, 1), (1, 0)]
    assert pair_batches[1] == [(1, 2), (2, 1)]


def test_pair_batched_relation_has_expected_shape_and_stays_fp32() -> None:
    q = torch.randn(2, 2, 3, 4, 5, requires_grad=True)
    k = torch.randn(2, 2, 3, 6, 5, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        probability = _head_mean_attention_pairs(q, k, 1.0)
    assert probability.shape == (2, 2, 4, 6)
    assert probability.dtype == torch.float32
    probability.square().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()


@pytest.mark.parametrize("pair_chunk_size", [2, 4])
def test_optimized_loss_and_student_gradients_match_legacy(
    pair_chunk_size: int,
) -> None:
    teacher = _feature(11, (2, 6), 3, 4, requires_grad=True)
    legacy_student = _feature(29, (1, 3), 2, 3, requires_grad=True)
    optimized_student = {
        "q": legacy_student["q"].detach().clone().requires_grad_(True),
        "k": legacy_student["k"].detach().clone().requires_grad_(True),
        "metadata": deepcopy(legacy_student["metadata"]),
    }

    legacy_loss = CrossFrameAttentionDistillationLoss(
        _config()
    )._layer_loss(teacher, legacy_student)
    optimized_loss = CrossFrameAttentionDistillationLoss(
        _config(pair_chunk_size=pair_chunk_size, teacher_outside=True)
    )._layer_loss(teacher, optimized_student)
    legacy_gradients = torch.autograd.grad(
        legacy_loss, (legacy_student["q"], legacy_student["k"])
    )
    optimized_gradients = torch.autograd.grad(
        optimized_loss, (optimized_student["q"], optimized_student["k"])
    )

    torch.testing.assert_close(optimized_loss, legacy_loss, rtol=1.0e-5, atol=1.0e-6)
    for optimized, legacy in zip(optimized_gradients, legacy_gradients):
        torch.testing.assert_close(optimized, legacy, rtol=1.0e-5, atol=1.0e-6)
    assert teacher["q"].grad is None and teacher["k"].grad is None


def test_teacher_probability_is_detached_and_not_recomputed_in_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = _feature(
        7, (1, 2), 2, 3, requires_grad=True, batch_size=1, frames=3
    )
    student = _feature(
        13, (1, 2), 2, 3, requires_grad=True, batch_size=1, frames=3
    )
    calls = 0
    original = attention_loss_module._teacher_pair_probability

    def recording_teacher_probability(*args, **kwargs):
        nonlocal calls
        calls += 1
        probability = original(*args, **kwargs)
        assert probability.requires_grad is False
        return probability

    monkeypatch.setattr(
        attention_loss_module,
        "_teacher_pair_probability",
        recording_teacher_probability,
    )
    loss = CrossFrameAttentionDistillationLoss(
        _config(pair_chunk_size=2, teacher_outside=True)
    )._layer_loss(teacher, student)
    forward_calls = calls
    assert forward_calls > 0
    loss.backward()
    assert calls == forward_calls

    direct_probability = _teacher_pair_probability(
        teacher["q"][:, :2], teacher["k"][:, :2], 1.0
    )
    assert direct_probability.requires_grad is False
    assert teacher["q"].grad is None and teacher["k"].grad is None
