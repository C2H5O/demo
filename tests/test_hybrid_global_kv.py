"""Synthetic contracts for Hybrid Global KV sequence and provider planning."""
from dataclasses import replace

import numpy as np
import pytest
import yaml

from inference.global_anchor_bank import farthest_point_indices, select_diverse_anchors, select_window_providers
from inference.kv_sampling import KVSamplingConfig, WindowFrameMetadata, build_spatial_patch_indices
from inference.hybrid_global_kv import HybridGlobalKVAttention


def _metadata(first=False, overlap=None):
    positions = tuple(range(32)) if first else tuple((0, 12, *range(24, 32), *range(32, 54)))
    if overlap is not None:
        positions = tuple((0, 12, *overlap, *range(32, 54)))
    return WindowFrameMetadata(
        window_id=0 if first else 1, frame_positions=positions,
        absolute_frame_ids=positions,
        frame_roles=("new",) * 32 if first else ("key",) * 2 + ("overlap",) * 8 + ("new",) * 22,
        is_padding=(False,) * 32, first_window=first,
    )


def test_fps_diverse_deterministic_and_valid():
    rng = np.random.default_rng(4)
    descriptors = rng.normal(size=(90, 12))
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    bank = select_diverse_anchors(descriptors)
    again = select_diverse_anchors(descriptors)
    assert bank.bank_positions == again.bank_positions
    assert bank.active_positions == again.active_positions
    assert len(set(bank.bank_positions)) == 50
    assert len(set(bank.active_positions)) == 15
    assert bank.bank_positions[0] == bank.active_positions[0] == 0
    assert set(bank.active_positions) <= set(bank.bank_positions)
    # For orthogonal points, FPS takes a different direction rather than a duplicate.
    assert farthest_point_indices(np.eye(4), 4) == [0, 1, 2, 3]


def test_first_standard_and_deduplicated_provider_budget():
    rng = np.random.default_rng(8)
    bank = select_diverse_anchors(rng.normal(size=(90, 8)))
    local, global_positions = select_window_providers(_metadata(first=True), bank)
    assert local == ()
    assert len(global_positions) == 25
    assert len(set(global_positions)) == 25
    local, global_positions = select_window_providers(_metadata(), bank)
    assert len(local) == 10
    assert len(global_positions) == 15
    assert len(set(local + global_positions)) == 25
    assert not set(local) & set(global_positions)
    short = select_diverse_anchors(rng.normal(size=(12, 8)))
    local, global_positions = select_window_providers(_metadata(first=True), short)
    assert len(global_positions) == 12


def test_spatial_budget_and_h_config():
    with open("configs/baselines/H.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["kv_sampling"]
    policy = KVSamplingConfig.from_mapping(config)
    assert policy.frame_budget(32) == 25
    assert policy.method == "hybrid_global_kv"
    assert len(build_spatial_patch_indices(32, 40, 1)) == 1280
    assert len(build_spatial_patch_indices(32, 40, 2)) == 320
    assert 10 * 1280 + 15 * 320 == 17600
    assert 25 * 1280 == 32000
    assert "lightweight_highlight" not in config
    assert "new_frame_selection" not in config
    with pytest.raises(ValueError, match="must not configure highlight"):
        replace(policy, lightweight_highlight={"brightness_threshold": .9}).frame_budget(32)


def test_runtime_spatial_plan_keeps_local_full_and_global_staggered():
    adapter = object.__new__(HybridGlobalKVAttention)
    adapter.metadata = _metadata()
    adapter.local_positions = adapter.metadata.frame_positions[:10]
    adapter.global_positions = tuple(range(100, 115))
    adapter.external_positions = list(adapter.global_positions)
    adapter.schedule = {layer: {"local_stride": 1,
                                "global_stride": 2 if layer in (5, 7) else 1}
                        for layer in (5, 7, 9, 11)}
    adapter.grid_height, adapter.grid_width = 32, 40
    for layer in (5, 7, 9, 11):
        current, external = adapter._patch_layout(layer)
        assert len(current) == 10 and len(external) == 15
        patches = sum(map(len, current.values())) + sum(map(len, external.values()))
        assert patches == (17600 if layer in (5, 7) else 32000)
    early5 = adapter._patch_layout(5)[1]
    early7 = adapter._patch_layout(7)[1]
    assert early5[0] != early7[0]
    assert set(early5[0]).isdisjoint(early7[0])
