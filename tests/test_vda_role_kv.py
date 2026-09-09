"""Optional CPU contract checks; no checkpoints, training or video data required.

These tests are supplied for the next validation stage. The implementation turn
performs syntax/config checks only, as requested.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from inference.da3_kv_attention import DA3KVAttention, frame_slots_to_token_indices
from inference.kv_sampling import (
    KVSamplingConfig, WindowFrameMetadata, eligible_frame_slots,
    select_kv_frames, uniform_select,
)
from inference.student_video import infer_student_video


def normal_window(first=False):
    positions = tuple(range(32)) if first else (0, 12, *range(24, 54))
    return WindowFrameMetadata(
        window_id=0 if first else 1, frame_positions=positions,
        absolute_frame_ids=tuple(1000 + i * 3 for i in positions),
        frame_roles=("new",) * 32 if first else ("key",) * 2 + ("overlap",) * 8 + ("new",) * 22,
        is_padding=(False,) * 32, first_window=first, absolute_id_source="test_dataset_ids",
    )


@pytest.mark.parametrize("length,count,expected", [
    (8, 2, [0, 7]), (22, 4, [0, 7, 14, 21]),
    (32, 8, [0, 4, 9, 13, 18, 22, 27, 31]),
    (3, 8, [0, 1, 2]), (8, 1, [3]), (0, 8, []), (8, 0, []),
])
def test_uniform_temporal_coverage(length, count, expected):
    assert uniform_select(list(range(length)), count) == expected


def test_normal_role_budget_and_first_window():
    config = KVSamplingConfig(enabled=True)
    budget = config.frame_budget(32)
    assert select_kv_frames(normal_window(), config, budget) == [0, 1, 2, 9, 10, 17, 24, 31]
    assert select_kv_frames(normal_window(True), config, budget) == [0, 4, 9, 13, 18, 22, 27, 31]
    fixed = replace(config, method="spark3r_fixed_stride")
    for first in (False, True):
        assert select_kv_frames(normal_window(first), fixed, budget) == list(range(0, 32, 4))
    # Dataset IDs are neither slots nor roles and do not change this policy.
    changed_ids = replace(normal_window(), absolute_frame_ids=tuple(reversed(range(32))))
    assert select_kv_frames(changed_ids, config, budget) == select_kv_frames(normal_window(), config, budget)


def test_budget_validation_and_disabled_dense():
    with pytest.raises(ValueError, match="Role quotas"):
        replace(KVSamplingConfig(enabled=True), new_frames=3).frame_budget(32)
    with pytest.raises(ValueError, match="same full-window"):
        KVSamplingConfig(enabled=True, method="spark3r_fixed_stride", temporal_stride=3).frame_budget(32)
    disabled = KVSamplingConfig(enabled=False)
    assert select_kv_frames(normal_window(), disabled, disabled.frame_budget(32)) == list(range(32))


@pytest.mark.parametrize("length", [1, 7, 22, 24, 31, 32, 33, 44, 54, 55, 79])
def test_real_window_provenance_tail_budget_and_complete_output(length, monkeypatch):
    metadata_records = []
    original_begin = DA3KVAttention.begin_window
    def record(self, metadata, images):
        metadata_records.append(metadata)
        return original_begin(self, metadata, images)
    monkeypatch.setattr(DA3KVAttention, "begin_window", record)

    class Plane(nn.Module):
        def forward(self, images, include_global_points=False):
            b, t, _, h, w = images.shape
            return {"depth": torch.ones(b, t, h, w),
                    "intrinsics": torch.eye(3).repeat(b, t, 1, 1)}

    output = []
    result = infer_student_video(Plane().eval(), [torch.zeros(3, 2, 3)] * length,
                                 lambda start, disparity, k: output.extend(disparity), device="cpu")
    assert result["output_frame_count"] == len(output) == length
    np.testing.assert_array_equal(output, np.ones((length, 2, 3)))
    role_config = KVSamplingConfig(enabled=True)
    stride_config = replace(role_config, method="spark3r_fixed_stride")
    for index, metadata in enumerate(metadata_records):
        if index == 0:
            assert metadata.frame_roles == ("new",) * 32
        else:
            previous = metadata_records[index - 1]
            assert metadata.frame_roles == ("key",) * 2 + ("overlap",) * 8 + ("new",) * 22
            assert metadata.frame_positions[:10] == tuple(previous.frame_positions[j] for j in
                                                          [0, 12, 24, 25, 26, 27, 28, 29, 30, 31])
        selections = [select_kv_frames(metadata, config, 8) for config in (role_config, stride_config)]
        assert len(selections[0]) == len(selections[1]) == min(8, len(eligible_frame_slots(metadata)))
        for slots in selections:
            assert len(slots) == len(set(slots))
            assert len(slots) == len({metadata.frame_positions[slot] for slot in slots})
            assert all(not metadata.is_padding[slot] and 0 <= slot < 32 for slot in slots)


def test_token_indices_preserve_all_specials_and_original_frame_identity():
    # Two batch elements choose different DA3 reference views. The oracle builds
    # the literal reordered layout independently of the vectorized index mapping.
    selected = [0, 2, 4]
    references = torch.tensor([3, 0])
    indices = frame_slots_to_token_indices(selected, num_frames=5, tokens_per_frame=7,
                                           special_tokens=3, reference_indices=references)
    for batch, reference in enumerate(references.tolist()):
        order = [reference] + [i for i in range(5) if i != reference]
        expected = {slot * 7 + token for slot in range(5) for token in range(3)}
        expected.update(order.index(frame) * 7 + token for frame in selected for token in range(3, 7))
        assert set(indices[batch].tolist()) == expected
        assert indices.shape[1] == len(expected) == 5 * 3 + 3 * 4


def tiny_da3(strategy):
    upstream = pytest.importorskip("depth_anything_3.model.dinov2.vision_transformer")
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.pretrained = upstream.DinoVisionTransformer(
                img_size=28, patch_size=14, embed_dim=24, depth=6, num_heads=2,
                alt_start=4, qknorm_start=4, rope_start=4,
            )
        def forward(self, images, **kwargs):
            return self.pretrained.get_intermediate_layers(images, n=[5], **kwargs)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
        def forward(self, images):
            return self.backbone(images, cam_token=None, ref_view_strategy=strategy)
    return Model().eval(), upstream


@pytest.mark.parametrize("strategy", ["first", "middle", "saddle_balanced"])
def test_upstream_sdpa_really_receives_sparse_kv_and_all_queries(strategy, monkeypatch):
    model, upstream = tiny_da3(strategy)
    images = torch.rand(1, 32, 3, 28, 28)
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    actual_references, captured = [], []
    original_selector = upstream.select_reference_view
    def record_selector(*args, **kwargs):
        result = original_selector(*args, **kwargs)
        actual_references.append(result.clone())
        return result
    monkeypatch.setattr(upstream, "select_reference_view", record_selector)
    adapter = DA3KVAttention(model, KVSamplingConfig(enabled=True), 32)
    original_attend = adapter.attend
    def observe(layer, kernel, query, key, value, **kwargs):
        full_query = query.clone()
        def audited_kernel(q, k, v, **kernel_kwargs):
            assert q.shape[-2] == 32 * 5
            assert k.shape[-2] == v.shape[-2] == 8 * 4 + 32
            torch.testing.assert_close(q, full_query, rtol=0, atol=0)
            expected_indices = adapter.token_indices[:, None, :, None].expand(1, 2, -1, 12)
            torch.testing.assert_close(k, key.gather(2, expected_indices), rtol=0, atol=0)
            torch.testing.assert_close(v, value.gather(2, expected_indices), rtol=0, atol=0)
            result = kernel(q, k, v, **kernel_kwargs)
            # Independently check the small rectangular attention, including all
            # query rows. This also catches a mask-only or query-pruning adapter.
            oracle = torch.softmax((q / (q.shape[-1] ** .5)) @ k.transpose(-2, -1), dim=-1) @ v
            torch.testing.assert_close(result, oracle, rtol=2e-5, atol=2e-6)
            captured.append((q.shape[-2], k.shape[-2]))
            return result
        return original_attend(layer, audited_kernel, query, key, value, **kwargs)
    adapter.attend = observe
    with torch.inference_mode(), adapter:
        adapter.begin_window(normal_window(), images)
        features, _ = model(images)
        torch.testing.assert_close(adapter.reference_indices, actual_references[-1])
        assert features[0][0].shape[:3] == (1, 32, 4)
        adapter.finish_window()
    assert captured == [(160, 64)]
    assert adapter.summary()["attention_shapes"] == [
        {"layer": 5, "q_token_count": 160, "kv_token_count": 64, "calls": 1}]
    assert "forward" not in vars(model.backbone.pretrained.blocks[5].attn)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)


def test_dense_is_unchanged_and_adapter_restores_on_error():
    model, _ = tiny_da3("middle")
    images = torch.rand(1, 32, 3, 28, 28)
    block = model.backbone.pretrained.blocks[5]
    source = model.backbone.pretrained.blocks[2]
    original_hook_count = len(source._forward_hooks)
    with torch.inference_mode():
        before, _ = model(images)
        with DA3KVAttention(model, KVSamplingConfig(enabled=False), 32) as adapter:
            assert "forward" not in vars(block.attn)
            adapter.begin_window(normal_window(), images)
            after, _ = model(images)
            adapter.finish_window()
        torch.testing.assert_close(before[0][0], after[0][0], rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="deliberate"):
            with DA3KVAttention(model, KVSamplingConfig(enabled=True), 32):
                raise RuntimeError("deliberate inference failure")
    assert "forward" not in vars(block.attn)
    assert len(source._forward_hooks) == original_hook_count
