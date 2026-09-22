"""Layer-wise spatial/temporal KV stride contracts for Baseline-H."""
from pathlib import Path
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference.da3_kv_attention import DA3KVAttention, frame_slots_to_token_indices
from inference.layer_stride_attention import (
    LayerStrideKVAttention,
    project_full_q_sparse_kv,
    sparse_attention_output,
    split_fused_qkv,
)
from inference.kv_sampling import (
    KVSamplingConfig,
    WindowFrameMetadata,
    build_spatial_patch_indices,
    resolve_layer_stride_policy,
    select_kv_frames,
    temporal_stride_slots,
)
from inference.student_video import infer_student_video
from test_vda_role_kv import normal_window, tiny_da3
from utils.config import load_config


POLICIES = {
    "A": {5: "spatial", 7: "spatial", 9: "temporal", 11: "temporal"},
    "B": {5: "temporal", 7: "temporal", 9: "spatial", 11: "spatial"},
    "C": {5: "spatial", 7: "temporal", 9: "spatial", 11: "temporal"},
    "D": {5: "temporal", 7: "spatial", 9: "temporal", 11: "spatial"},
}


class _TestRope(torch.nn.Module):
    def forward(self, value, pos):
        delta = (pos[..., :1] + 2 * pos[..., 1:2])[:, None]
        return value + delta.to(value.dtype)


class _TestAttention(torch.nn.Module):
    def __init__(self, hidden=12, heads=3):
        super().__init__()
        self.num_heads = heads
        self.scale = (hidden // heads) ** -0.5
        self.fused_attn = True
        self.qkv = torch.nn.Linear(hidden, 3 * hidden, bias=True)
        self.q_norm = torch.nn.LayerNorm(hidden // heads)
        self.k_norm = torch.nn.LayerNorm(hidden // heads)
        self.rope = _TestRope()
        self.attn_drop = torch.nn.Dropout(0.0)
        self.proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.proj_drop = torch.nn.Dropout(0.0)
        self.eval()


def _old_post_projection_gather(attention, x, indices, pos):
    batch, tokens, hidden = x.shape
    heads, head_dim = attention.num_heads, hidden // attention.num_heads
    qkv = attention.qkv(x).reshape(batch, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = attention.q_norm(q), attention.k_norm(k)
    q, k = attention.rope(q, pos), attention.rope(k, pos)
    k = k.index_select(2, indices)
    v = v.index_select(2, indices)
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    output = output.transpose(1, 2).reshape(batch, tokens, hidden)
    return attention.proj_drop(attention.proj(output)), (q, k, v)


def _run_wrapped_optimized_attention(attention, x, indices, pos):
    adapter = object.__new__(LayerStrideKVAttention)
    adapter.blocks = [None] * 5 + [SimpleNamespace(attn=attention)]
    adapter.metadata = object()
    adapter.model = SimpleNamespace(training=False)
    adapter.token_indices_by_layer = {5: indices[None]}
    adapter.frames = 3
    adapter.tokens_per_frame = 3
    adapter.config = SimpleNamespace(profile_attention=False)
    adapter.calls = []
    adapter.projection_token_counts = {}
    with torch.inference_mode():
        output = adapter._wrap(attention.forward, 5)(x, pos=pos)
    return output, adapter.projection_token_counts[5]


def stride_config(variant="A"):
    return KVSamplingConfig.from_mapping(
        load_config(f"configs/baselines/H_{variant}.yaml")["kv_sampling"]
    )


def test_spatial_stride2_is_two_dimensional_fixed_lattice():
    indices = build_spatial_patch_indices(32, 40, 2)
    assert len(indices) == len(set(indices)) == 320
    assert all(divmod(index, 40)[0] % 2 == 0 for index in indices)
    assert all(divmod(index, 40)[1] % 2 == 0 for index in indices)


def test_temporal_stride2_uses_current_vda_slots_only():
    assert temporal_stride_slots(32, 2) == tuple(range(0, 32, 2))
    assert len(temporal_stride_slots(32, 2)) == 16


def test_fused_qkv_split_is_three_parameter_views_with_correct_dimensions():
    attention = _TestAttention()
    weights, biases = split_fused_qkv(attention.qkv, 12)
    assert [tuple(value.shape) for value in weights] == [(12, 12)] * 3
    assert [tuple(value.shape) for value in biases] == [(12,)] * 3
    assert all(value.untyped_storage().data_ptr() == attention.qkv.weight.untyped_storage().data_ptr()
               for value in weights)
    assert all(value.untyped_storage().data_ptr() == attention.qkv.bias.untyped_storage().data_ptr()
               for value in biases)


def test_preprojection_qkv_matches_full_fused_projection_then_selection():
    torch.manual_seed(7)
    attention = _TestAttention()
    x = torch.randn(1, 9, 12)
    indices = torch.tensor([0, 2, 5, 8])
    old = attention.qkv(x).reshape(1, 9, 3, 3, 4).permute(2, 0, 3, 1, 4)
    q, k, v = project_full_q_sparse_kv(attention, x, x.index_select(1, indices))
    torch.testing.assert_close(q, old[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(k, old[1].index_select(2, indices), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(v, old[2].index_select(2, indices), rtol=1e-6, atol=1e-6)


def test_preprojection_complete_attention_matches_old_post_projection_gather():
    torch.manual_seed(11)
    attention = _TestAttention()
    x = torch.randn(1, 9, 12)
    pos = torch.randint(0, 4, (1, 9, 2))
    indices = torch.tensor([0, 2, 5, 8])
    old_output, _ = _old_post_projection_gather(attention, x, indices, pos)
    new_output, audit = _run_wrapped_optimized_attention(attention, x, indices, pos)
    torch.testing.assert_close(new_output, old_output, rtol=2e-6, atol=2e-6)
    assert audit == {
        "q_projection_tokens": 9,
        "k_projection_tokens": 4,
        "v_projection_tokens": 4,
        "sdpa_q_tokens": 9,
        "sdpa_kv_tokens": 4,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast unavailable")
def test_bfloat16_autocast_attention_output_matches_old_path():
    torch.manual_seed(13)
    attention = _TestAttention().cuda()
    x = torch.randn(1, 9, 12, device="cuda")
    pos = torch.randint(0, 4, (1, 9, 2), device="cuda")
    indices = torch.tensor([0, 2, 5, 8], device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        old_output, _ = _old_post_projection_gather(attention, x, indices, pos)
        new_output = sparse_attention_output(attention, x, indices, pos)
    torch.testing.assert_close(new_output, old_output, rtol=2e-2, atol=2e-2)


def test_a_projection_and_sdpa_token_budgets_are_exact():
    assert {5: 10272, 7: 10272, 9: 20512, 11: 20512} == {
        layer: 32 + (32 * 320 if policy == "spatial" else 16 * 1280)
        for layer, policy in POLICIES["A"].items()
    }


def test_optimized_adapter_has_no_full_projected_kv_gather():
    source = inspect.getsource(LayerStrideKVAttention)
    assert "torch.gather" not in source
    assert ".gather(" not in source
    assert "x.index_select(1, indices)" in source


def test_tail_window_keeps_all_tensor_slots_even_when_padding_repeats_frames():
    positions = tuple(range(10)) + (9,) * 22
    metadata = WindowFrameMetadata(
        window_id=0,
        frame_positions=positions,
        absolute_frame_ids=positions,
        frame_roles=("new",) * 32,
        is_padding=(False,) * 10 + (True,) * 22,
        first_window=True,
    )
    config = stride_config("A")
    assert select_kv_frames(metadata, config, config.frame_budget(32)) == list(range(32))


@pytest.mark.parametrize("variant", "ABCD")
def test_abcd_configs_are_independent_and_have_exact_layer_policies(variant):
    raw = load_config(f"configs/baselines/H_{variant}.yaml")
    config = KVSamplingConfig.from_mapping(raw["kv_sampling"])
    assert config.method == "layer_stride_kv"
    assert resolve_layer_stride_policy(config.layer_policy) == POLICIES[variant]
    assert config.frame_budget(32) == 32
    assert config.keep_all_special_tokens is True
    assert config.special_tokens == {"keep_all": True}
    assert raw["inference"]["image_height"] == 448
    assert raw["inference"]["image_width"] == 560
    assert raw["vda_evaluation"]["evaluation_height"] == 224
    assert raw["vda_evaluation"]["evaluation_width"] == 280
    assert raw["vda_evaluation"]["tae"]["enabled"] is False
    if variant == "A":
        assert raw["kv_sampling"]["debug"] is False
        assert raw["kv_sampling"]["profile_attention"] is False
    assert raw["vda_evaluation"]["output"].endswith(
        f"outputs/baseline_H/output_{variant}/evaluation.json"
    )
    for forbidden in (
        "global_anchor_bank", "anchor_selection", "provider_policy",
        "persistent_kv_cache", "highlight_detection", "lightweight_highlight",
        "bucket_highlight", "new_frame_selection",
    ):
        assert forbidden not in raw["kv_sampling"]


@pytest.mark.parametrize("reference", [0, 7, 15, 31])
@pytest.mark.parametrize("policy", ["spatial", "temporal"])
def test_reference_permutation_preserves_original_stride_policy(reference, policy):
    tokens_per_frame, special_tokens = 1281, 1
    if policy == "spatial":
        selected = list(range(32))
        lattice = build_spatial_patch_indices(32, 40, 2)
        patches = {slot: lattice for slot in selected}
        expected_patch_count = 32 * 320
    else:
        selected = list(temporal_stride_slots(32, 2))
        patches = None
        expected_patch_count = 16 * 1280
    actual = frame_slots_to_token_indices(
        selected,
        num_frames=32,
        tokens_per_frame=tokens_per_frame,
        special_tokens=special_tokens,
        reference_indices=torch.tensor([reference], dtype=torch.long),
        patch_indices_by_slot=patches,
    )[0]
    order = [reference] + [slot for slot in range(32) if slot != reference]
    expected = {internal * tokens_per_frame for internal in range(32)}
    offsets = lattice if policy == "spatial" else range(1280)
    expected.update(
        order.index(slot) * tokens_per_frame + special_tokens + patch
        for slot in selected
        for patch in offsets
    )
    assert set(actual.tolist()) == expected
    assert actual.numel() == 32 + expected_patch_count


@pytest.mark.parametrize("variant", "ABCD")
def test_real_global_sdpa_keeps_full_q_and_uses_layer_stride_budgets(variant):
    model, _ = tiny_da3("middle", depth=12)
    adapter = DA3KVAttention(model, stride_config(variant), 32)
    images = torch.zeros(1, 32, 3, 28, 28)
    assert isinstance(adapter, LayerStrideKVAttention)
    with torch.inference_mode(), adapter:
        adapter.begin_window(normal_window(), images)
        features, _ = model(images)
        assert features[0][0].shape[:3] == (1, 32, 4)
        adapter.finish_window()
    summary = adapter.summary()
    assert summary["attention_shapes"] == [
        {"layer": layer, "q_token_count": 160,
         "kv_token_count": 64 if POLICIES[variant][layer] == "spatial" else 96,
         "calls": 1}
        for layer in (5, 7, 9, 11)
    ]
    audit = summary["kv_selection_examples"][0]
    for layer, policy in POLICIES[variant].items():
        item = audit["layer_stride_kv"][layer]
        assert item["q_frame_count"] == 32
        assert item["kv_patch_frame_count"] == (32 if policy == "spatial" else 16)
        assert item["spatial_patch_count_per_frame"] == (1 if policy == "spatial" else 4)
        expected_kv = 64 if policy == "spatial" else 96
        assert item["q_projection_tokens"] == item["sdpa_q_tokens"] == 160
        assert item["k_projection_tokens"] == expected_kv
        assert item["v_projection_tokens"] == expected_kv
        assert item["sdpa_kv_tokens"] == expected_kv


def test_inference_still_emits_thirty_two_depth_maps():
    backbone_model, _ = tiny_da3("middle", depth=12)

    class DepthModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone_model.backbone
            self.attention_capture = None

        def forward(self, images, include_global_points=False):
            self.backbone(images, cam_token=None, ref_view_strategy="middle")
            batch, frames, _, height, width = images.shape
            return {
                "depth": torch.ones(batch, frames, height, width),
                "intrinsics": torch.eye(3).repeat(batch, frames, 1, 1),
            }

    frames = [torch.zeros(3, 28, 28) for _ in range(32)]
    emitted = []
    result = infer_student_video(
        DepthModel().eval(), frames,
        lambda start, disparity, intrinsics: emitted.extend(disparity),
        device="cpu", amp=False, kv_sampling=stride_config("A"),
    )
    assert result["output_frame_count"] == len(emitted) == 32
    assert np.asarray(emitted).shape == (32, 28, 28)
    assert result["descriptor_seconds"] == 0.0
    assert result["anchor_backbone_seconds"] == 0.0
    assert result["highlight_selection_seconds"] == 0.0


def test_runner_is_fail_fast_and_covers_each_config_once():
    text = Path("scripts/run_baseline_h_stride_ablation.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "for variant in A B C D" in text
    assert 'configs/baselines/H_${variant}.yaml' in text


def test_microbenchmark_is_a_only_and_excludes_warmup():
    text = Path("scripts/benchmark_layer_stride_attention.py").read_text(encoding="utf-8")
    assert 'default=Path("configs/baselines/H_A.yaml")' in text
    assert 'parser.add_argument("--warmup", type=int, default=20)' in text
    assert 'parser.add_argument("--iterations", type=int, default=50)' in text
    assert "warmup_windows_excluded" in text
    assert "H_B.yaml" not in text and "H_C.yaml" not in text and "H_D.yaml" not in text
