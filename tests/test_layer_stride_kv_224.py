"""224x280 layer-wise spatial/temporal KV stride contracts for Baseline-H A."""
from pathlib import Path
import inspect
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference.da3_kv_attention import DA3KVAttention, frame_slots_to_token_indices
from inference.layer_stride_attention import (
    LayerStrideKVAttention,
    project_full_q_sparse_kv,
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


POLICY_A = {5: "spatial", 7: "spatial", 9: "temporal", 11: "temporal"}


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


def stride_config(profile=False):
    raw = load_config("configs/baselines/H_A.yaml")
    mapping = dict(raw["kv_sampling"])
    mapping["profile_attention"] = profile
    return KVSamplingConfig.from_mapping(mapping)


def _run_wrapped(attention, x, indices, pos, *, profile):
    adapter = object.__new__(LayerStrideKVAttention)
    adapter.blocks = [None] * 5 + [SimpleNamespace(attn=attention)]
    adapter.metadata = object()
    adapter.model = SimpleNamespace(training=False)
    adapter.token_indices_by_layer = {5: indices[None]}
    adapter.frames = 3
    adapter.tokens_per_frame = 3
    adapter.config = SimpleNamespace(profile_attention=profile)
    adapter.calls = []
    adapter.projection_token_counts = {}
    adapter.profile_calls = Counter()
    adapter.profile_seconds = Counter()
    adapter.profile_events = []
    adapter.profiled_calls = 0
    with torch.inference_mode():
        output = adapter._wrap(attention.forward, 5)(x, pos=pos)
    return output, adapter.projection_token_counts[5]


def test_224_grid_and_exact_scheme_a_token_budgets():
    patch_size, height, width, frames, special = 14, 224, 280, 32, 1
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = grid_h * grid_w
    lattice = build_spatial_patch_indices(grid_h, grid_w, 2)
    assert (grid_h, grid_w, patches) == (16, 20, 320)
    assert len(lattice) == 80
    assert all(divmod(index, grid_w)[0] % 2 == 0 for index in lattice)
    assert all(divmod(index, grid_w)[1] % 2 == 0 for index in lattice)
    assert frames * patches + frames * special == 10272
    assert frames * len(lattice) + frames * special == 2592
    assert len(temporal_stride_slots(frames, 2)) * patches + frames * special == 5152


def test_scheme_a_config_is_clean_224_and_formal_profiling_is_off():
    raw = load_config("configs/baselines/H_A.yaml")
    config = KVSamplingConfig.from_mapping(raw["kv_sampling"])
    assert config.method == "layer_stride_kv"
    assert resolve_layer_stride_policy(config.layer_policy) == POLICY_A
    assert config.frame_budget(32) == 32
    assert config.special_tokens == {"keep_all": True}
    assert config.debug is False and config.profile_attention is False
    assert (raw["inference"]["image_height"], raw["inference"]["image_width"]) == (224, 280)
    assert (raw["vda_evaluation"]["evaluation_height"],
            raw["vda_evaluation"]["evaluation_width"]) == (224, 280)
    for forbidden in (
        "highlight_detection", "lightweight_highlight", "bucket_highlight",
        "new_frame_selection", "spatial_sampling",
    ):
        assert forbidden not in raw["kv_sampling"]


def test_fused_qkv_views_and_preprojection_token_counts():
    torch.manual_seed(7)
    attention = _TestAttention()
    weights, biases = split_fused_qkv(attention.qkv, 12)
    assert [tuple(value.shape) for value in weights] == [(12, 12)] * 3
    assert [tuple(value.shape) for value in biases] == [(12,)] * 3
    assert all(value.untyped_storage().data_ptr() == attention.qkv.weight.untyped_storage().data_ptr()
               for value in weights)
    x = torch.randn(1, 9, 12)
    indices = torch.tensor([0, 2, 5, 8])
    old = attention.qkv(x).reshape(1, 9, 3, 3, 4).permute(2, 0, 3, 1, 4)
    q, k, v = project_full_q_sparse_kv(attention, x, x.index_select(1, indices))
    torch.testing.assert_close(q, old[0])
    torch.testing.assert_close(k, old[1].index_select(2, indices))
    torch.testing.assert_close(v, old[2].index_select(2, indices))
    assert (q.shape[-2], k.shape[-2], v.shape[-2]) == (9, 4, 4)


def test_diagnostics_toggle_does_not_change_sparse_attention_output():
    torch.manual_seed(11)
    attention = _TestAttention()
    x = torch.randn(1, 9, 12)
    pos = torch.randint(0, 4, (1, 9, 2))
    indices = torch.tensor([0, 2, 5, 8])
    unprofiled, plain_audit = _run_wrapped(attention, x, indices, pos, profile=False)
    profiled, profiled_audit = _run_wrapped(attention, x, indices, pos, profile=True)
    torch.testing.assert_close(profiled, unprofiled, rtol=0, atol=0)
    expected = {
        "q_projection_tokens": 9,
        "k_projection_tokens": 4,
        "v_projection_tokens": 4,
        "sdpa_q_tokens": 9,
        "sdpa_kv_tokens": 4,
    }
    assert plain_audit == profiled_audit == expected


def test_exact_224_projection_and_sdpa_token_counts_are_observed(monkeypatch):
    attention = _TestAttention()
    adapter = object.__new__(LayerStrideKVAttention)
    adapter.blocks = [None] * 12
    for layer in POLICY_A:
        adapter.blocks[layer] = SimpleNamespace(attn=attention)
    adapter.metadata = object()
    adapter.model = SimpleNamespace(training=False)
    adapter.frames = 32
    adapter.tokens_per_frame = 321
    adapter.config = SimpleNamespace(profile_attention=False)
    adapter.calls = []
    adapter.projection_token_counts = {}

    reference = torch.tensor([7], dtype=torch.long)
    spatial_lattice = build_spatial_patch_indices(16, 20, 2)
    adapter.token_indices_by_layer = {}
    for layer, policy in POLICY_A.items():
        selected = list(range(32)) if policy == "spatial" else list(range(0, 32, 2))
        patches = ({slot: spatial_lattice for slot in selected}
                   if policy == "spatial" else None)
        adapter.token_indices_by_layer[layer] = frame_slots_to_token_indices(
            selected, num_frames=32, tokens_per_frame=321, special_tokens=1,
            reference_indices=reference, patch_indices_by_slot=patches,
        )

    observed_sdpa = []

    def fake_sdpa(q, k, v, **kwargs):
        observed_sdpa.append((q.shape[-2], k.shape[-2], v.shape[-2]))
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    x = torch.zeros(1, 10272, 12)
    with torch.inference_mode():
        for layer in POLICY_A:
            output = adapter._wrap(attention.forward, layer)(x)
            assert output.shape == x.shape

    assert adapter.projection_token_counts == {
        5: {"q_projection_tokens": 10272, "k_projection_tokens": 2592,
            "v_projection_tokens": 2592, "sdpa_q_tokens": 10272,
            "sdpa_kv_tokens": 2592},
        7: {"q_projection_tokens": 10272, "k_projection_tokens": 2592,
            "v_projection_tokens": 2592, "sdpa_q_tokens": 10272,
            "sdpa_kv_tokens": 2592},
        9: {"q_projection_tokens": 10272, "k_projection_tokens": 5152,
            "v_projection_tokens": 5152, "sdpa_q_tokens": 10272,
            "sdpa_kv_tokens": 5152},
        11: {"q_projection_tokens": 10272, "k_projection_tokens": 5152,
             "v_projection_tokens": 5152, "sdpa_q_tokens": 10272,
             "sdpa_kv_tokens": 5152},
    }
    assert observed_sdpa == [
        (10272, 2592, 2592), (10272, 2592, 2592),
        (10272, 5152, 5152), (10272, 5152, 5152),
    ]


@pytest.mark.parametrize("reference", [0, 7, 15, 31])
@pytest.mark.parametrize("policy", ["spatial", "temporal"])
def test_reference_permutation_preserves_original_224_stride_policy(reference, policy):
    tokens_per_frame, special_tokens = 321, 1
    if policy == "spatial":
        selected = list(range(32))
        lattice = build_spatial_patch_indices(16, 20, 2)
        patches = {slot: lattice for slot in selected}
        patch_offsets = lattice
        expected_patch_count = 32 * 80
    else:
        selected = list(temporal_stride_slots(32, 2))
        patches = None
        patch_offsets = range(320)
        expected_patch_count = 16 * 320
    actual = frame_slots_to_token_indices(
        selected, num_frames=32, tokens_per_frame=tokens_per_frame,
        special_tokens=special_tokens,
        reference_indices=torch.tensor([reference], dtype=torch.long),
        patch_indices_by_slot=patches,
    )[0]
    internal_order = [reference] + [slot for slot in range(32) if slot != reference]
    expected = {internal * tokens_per_frame for internal in range(32)}
    expected.update(
        internal_order.index(slot) * tokens_per_frame + special_tokens + patch
        for slot in selected for patch in patch_offsets
    )
    assert set(actual.tolist()) == expected
    assert actual.numel() == 32 + expected_patch_count


def test_tail_window_keeps_all_thirty_two_query_slots():
    positions = tuple(range(10)) + (9,) * 22
    metadata = WindowFrameMetadata(
        window_id=0, frame_positions=positions, absolute_frame_ids=positions,
        frame_roles=("new",) * 32, is_padding=(False,) * 10 + (True,) * 22,
        first_window=True,
    )
    config = stride_config()
    assert select_kv_frames(metadata, config, config.frame_budget(32)) == list(range(32))


def test_real_global_attention_keeps_full_q_and_uses_dynamic_stride_budgets():
    model, _ = tiny_da3("middle", depth=12)
    adapter = DA3KVAttention(model, stride_config(), 32)
    assert isinstance(adapter, LayerStrideKVAttention)
    images = torch.zeros(1, 32, 3, 28, 28)
    with torch.inference_mode(), adapter:
        adapter.begin_window(normal_window(), images)
        features, _ = model(images)
        assert features[0][0].shape[:3] == (1, 32, 4)
        adapter.finish_window()
    audit = adapter.summary()["kv_selection_examples"][0]["layer_stride_kv"]
    for layer, policy in POLICY_A.items():
        expected_kv = 64 if policy == "spatial" else 96
        item = audit[layer]
        assert item["q_frame_count"] == 32
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
        device="cpu", amp=False, kv_sampling=stride_config(),
    )
    assert result["output_frame_count"] == len(emitted) == 32
    assert np.asarray(emitted).shape == (32, 28, 28)


def test_new_path_has_no_projected_kv_gather_and_old_h_method_remains_configured():
    source = inspect.getsource(LayerStrideKVAttention)
    assert "torch.gather" not in source and ".gather(" not in source
    assert "x.index_select(1, indices)" in source
    old = load_config("configs/baselines/H.yaml")
    old_config = KVSamplingConfig.from_mapping(old["kv_sampling"])
    assert old_config.method == "role_layer_spatial_kv"
    assert old_config.frame_budget(32) == 24


def test_microbenchmark_uses_fixed_224_protocol_and_excludes_io():
    text = Path("scripts/benchmark_layer_stride_attention.py").read_text(encoding="utf-8")
    assert "torch.rand(1, WINDOW, 3, 224, 280, device=device)" in text
    assert 'parser.add_argument("--warmup", type=int, default=20)' in text
    assert 'parser.add_argument("--iterations", type=int, default=50)' in text
    assert "warmup_windows_excluded" in text
    assert "profiled_vs_unprofiled_sparse_a_allclose" in text
    assert "excludes RGB decode, stitching, finish_window audit, and scoring" in text


def test_evaluation_cli_exposes_complete_sequence_limit():
    text = Path("evaluate_crossclip_projection.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--limit-sequences"' in text
    assert "limit_sequences=args.limit_sequences" in text
