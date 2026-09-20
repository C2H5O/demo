"""CPU-only QG-K20 selection and grouped-attention contracts."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from inference.da3_kv_attention import DA3KVAttention
from inference.kv_sampling import KVSamplingConfig, WindowFrameMetadata
from inference.query_group_kv import (
    build_group_token_indices,
    build_query_group_providers,
    grouped_scaled_dot_product_attention,
)
from test_vda_role_kv import normal_window, tiny_da3
from utils.config import load_config


def current_h() -> KVSamplingConfig:
    # QG remains a standalone optional policy after H switches to Hybrid KV.
    return KVSamplingConfig(enabled=True, method="query_group", diagnostics=False)


def test_h_selects_hybrid_policy_and_qg_stays_explicitly_configurable() -> None:
    raw = load_config("configs/baselines/H.yaml")
    config = current_h()
    assert raw["kv_sampling"]["method"] == "hybrid_global_kv"
    assert raw["kv_sampling"]["spatial_sampling"]["block5"] == {
        "local_stride": 1, "global_stride": 2}
    assert config.method == "query_group"
    assert config.query_group_size == 8
    assert config.kv_frames == 20
    assert config.provider_selection == "temporal_uniform"
    assert config.preserve_vda_history is True
    assert config.keep_all_special_tokens is True
    assert config.apply_layers is None
    assert config.batched_sdpa is True
    assert raw["vda_evaluation"]["tae"]["enabled"] is False
    assert (raw["inference"]["image_height"], raw["inference"]["image_width"]) == (448, 560)
    assert raw["vda_evaluation"]["evaluation_height"] == 224
    assert raw["vda_evaluation"]["evaluation_width"] == 280


def test_each_query_group_has_own_temporal_k20_and_mandatory_frames() -> None:
    metadata = normal_window()
    queries, providers = build_query_group_providers(
        metadata, 8, 20, preserve_vda_history=True
    )
    assert queries == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
    ]
    assert all(len(group) == 20 for group in providers)
    assert len({tuple(group) for group in providers}) == 4
    for query_group, provider_group in zip(queries, providers):
        query_positions = {metadata.frame_positions[slot] for slot in query_group}
        provider_positions = {metadata.frame_positions[slot] for slot in provider_group}
        assert query_positions <= provider_positions
        assert provider_group == sorted(
            provider_group,
            key=lambda slot: (metadata.frame_positions[slot], slot),
        )
    # Later-window history is preferred before uniform filling.
    assert set(range(10)) <= set(providers[-1])


@pytest.mark.parametrize(
    "group_size,kv_frames,group_count",
    ((4, 16, 8), (8, 20, 4), (16, 24, 2)),
)
def test_group_size_and_k_are_yaml_style_parameters(
    group_size: int, kv_frames: int, group_count: int
) -> None:
    config = replace(
        current_h(), query_group_size=group_size, kv_frames=kv_frames
    )
    assert config.frame_budget(32) == kv_frames
    queries, providers = build_query_group_providers(
        normal_window(),
        config.query_group_size,
        config.kv_frames,
        preserve_vda_history=config.preserve_vda_history,
    )
    assert len(queries) == len(providers) == group_count
    assert all(len(provider) == kv_frames for provider in providers)


def test_tail_uses_all_unique_real_frames_without_repeating_padding() -> None:
    positions = tuple(range(10)) + (9,) * 22
    metadata = WindowFrameMetadata(
        window_id=3,
        frame_positions=positions,
        absolute_frame_ids=positions,
        frame_roles=("new",) * 32,
        is_padding=(False,) * 10 + (True,) * 22,
        first_window=True,
    )
    queries, providers = build_query_group_providers(
        metadata, 8, 20, preserve_vda_history=True
    )
    assert len(queries) == 4
    assert all(provider == list(range(10)) for provider in providers)


def test_special_token_switch_and_full_spatial_provider_tokens() -> None:
    queries = [[0, 1], [2, 3]]
    providers = [[0, 1, 2], [1, 2, 3]]
    references = torch.tensor([2, 0], dtype=torch.long)
    query_indices, all_special = build_group_token_indices(
        queries,
        providers,
        num_frames=4,
        tokens_per_frame=5,
        special_tokens=1,
        reference_indices=references,
        keep_all_special_tokens=True,
    )
    _, provider_special = build_group_token_indices(
        queries,
        providers,
        num_frames=4,
        tokens_per_frame=5,
        special_tokens=1,
        reference_indices=references,
        keep_all_special_tokens=False,
    )
    assert all(index.shape == (2, 10) for index in query_indices)
    assert all(index.shape == (2, 4 + 3 * 4) for index in all_special)
    assert all(index.shape == (2, 3 * 5) for index in provider_special)


def test_standard_groups_use_one_batched_sdpa_and_restore_token_order() -> None:
    metadata = normal_window(True)
    queries, providers = build_query_group_providers(
        metadata, 8, 20, preserve_vda_history=True
    )
    query_indices, provider_indices = build_group_token_indices(
        queries,
        providers,
        num_frames=32,
        tokens_per_frame=3,
        special_tokens=1,
        reference_indices=torch.tensor([7]),
        keep_all_special_tokens=True,
    )
    torch.manual_seed(7)
    query = torch.randn(1, 2, 96, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    observed = []

    def kernel(q, k, v, **kwargs):
        observed.append((tuple(q.shape), tuple(k.shape), tuple(v.shape)))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)

    actual, calls = grouped_scaled_dot_product_attention(
        kernel,
        query,
        key,
        value,
        query_indices,
        provider_indices,
        batched_sdpa=True,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    assert observed == [((4, 2, 24, 4), (4, 2, 72, 4), (4, 2, 72, 4))]
    assert calls == [
        {"group_count": 4, "query_tokens_per_group": 24, "kv_tokens_per_group": 72}
    ]
    oracle = torch.empty_like(actual)
    for q_index, kv_index in zip(query_indices, provider_indices):
        q = query.gather(2, q_index[:, None, :, None].expand(1, 2, -1, 4))
        k = key.gather(2, kv_index[:, None, :, None].expand(1, 2, -1, 4))
        v = value.gather(2, kv_index[:, None, :, None].expand(1, 2, -1, 4))
        result = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        oracle.scatter_(2, q_index[:, None, :, None].expand(1, 2, -1, 4), result)
    torch.testing.assert_close(actual, oracle)


def test_restore_uses_sdpa_output_dtype_under_autocast() -> None:
    query = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
    key = query.clone()
    value = query.clone()
    query_indices = [torch.tensor([[0, 1]]), torch.tensor([[2, 3]])]
    provider_indices = [torch.tensor([[0, 2]]), torch.tensor([[1, 3]])]

    def autocast_like_kernel(q, k, v, **kwargs):
        return q.to(torch.bfloat16)

    actual, calls = grouped_scaled_dot_product_attention(
        autocast_like_kernel,
        query,
        key,
        value,
        query_indices,
        provider_indices,
        batched_sdpa=True,
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, query.to(torch.bfloat16))
    assert calls == [
        {"group_count": 2, "query_tokens_per_group": 2, "kv_tokens_per_group": 2}
    ]


@pytest.mark.parametrize("strategy", ("first", "middle", "saddle_balanced"))
def test_real_da3_global_layers_use_qg_full_spatial_batched_sdpa(strategy) -> None:
    model, _ = tiny_da3(strategy, depth=12)
    adapter = DA3KVAttention(model, current_h(), 32)
    observed = []
    original = adapter.attend

    def inspect(layer, kernel, query, key, value, **kwargs):
        def checked(q, k, v, **kernel_kwargs):
            observed.append((layer, tuple(q.shape), tuple(k.shape)))
            return kernel(q, k, v, **kernel_kwargs)

        return original(layer, checked, query, key, value, **kwargs)

    adapter.attend = inspect
    with torch.inference_mode(), adapter:
        images = torch.zeros(1, 32, 3, 28, 28)
        adapter.begin_window(normal_window(), images)
        features, _ = model(images)
        adapter.finish_window()
    assert features[0][0].shape[:3] == (1, 32, 4)
    assert [layer for layer, _, _ in observed] == [5, 7, 9, 11]
    assert all(q_shape == (4, 2, 40, 12) for _, q_shape, _ in observed)
    assert all(k_shape == (4, 2, 112, 12) for _, _, k_shape in observed)
    assert adapter.summary()["grouped_sdpa_kernel_calls"] == {
        5: 1,
        7: 1,
        9: 1,
        11: 1,
    }


def test_apply_layers_can_select_a_global_layer_subset() -> None:
    model, _ = tiny_da3("middle", depth=12)
    adapter = DA3KVAttention(model, replace(current_h(), apply_layers=[7, 9]), 32)
    with torch.inference_mode(), adapter:
        images = torch.zeros(1, 32, 3, 28, 28)
        adapter.begin_window(normal_window(), images)
        model(images)
        adapter.finish_window()
    assert adapter.layers == [7, 9]
    assert adapter.summary()["grouped_sdpa_kernel_calls"] == {7: 1, 9: 1}
