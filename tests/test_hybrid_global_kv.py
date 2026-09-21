"""Synthetic contracts for persistent Fixed Global-50 K/V."""
from dataclasses import replace

import numpy as np
import pytest
import torch
import yaml

from inference.global_anchor_bank import farthest_point_indices, select_diverse_anchors
from inference.hybrid_global_kv import HybridGlobalKVAttention
from inference.kv_sampling import KVSamplingConfig, WindowFrameMetadata, build_spatial_patch_indices


def metadata(window_id):
    return WindowFrameMetadata(
        window_id=window_id, frame_positions=tuple(range(window_id * 22, window_id * 22 + 32)),
        absolute_frame_ids=tuple(range(window_id * 22, window_id * 22 + 32)),
        frame_roles=("new",) * 32, is_padding=(False,) * 32, first_window=window_id == 0)


def test_fps_selects_fifty_unique_deterministic_anchors():
    descriptors = np.random.default_rng(4).normal(size=(90, 12))
    first, again = select_diverse_anchors(descriptors), select_diverse_anchors(descriptors)
    assert first.anchor_positions == again.anchor_positions
    assert len(first.anchor_positions) == len(set(first.anchor_positions)) == 50
    assert first.anchor_positions[0] == 0
    assert farthest_point_indices(np.eye(4), 4) == [0, 1, 2, 3]
    assert len(select_diverse_anchors(descriptors[:12]).anchor_positions) == 12


def test_config_and_fixed_global_spatial_budgets():
    with open("configs/baselines/H.yaml", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)["kv_sampling"]
    policy = KVSamplingConfig.from_mapping(raw)
    assert policy.method == "fixed_global_kv"
    assert policy.frame_budget(32) == 50
    assert policy.provider_policy == "fixed_sequence_global"
    assert policy.persistent_kv_cache is True
    assert policy.as_dict()["anchor_count"] == 50
    assert "local_history" not in policy.as_dict()
    assert "local_history" not in raw and "active_global_frames" not in raw
    assert len(build_spatial_patch_indices(32, 40, 2)) == 320
    assert 50 * 320 == 16000 and 50 * 1280 == 64000
    assert [policy.spatial_sampling[f"block{i}"]["global_stride"] for i in (5, 7, 9, 11)] == [2, 2, 2, 1]
    with pytest.raises(ValueError, match="must not configure highlight"):
        replace(policy, lightweight_highlight={"x": 1}).frame_budget(32)


def test_windows_reuse_same_cache_and_anchor_identity_without_building():
    adapter = object.__new__(HybridGlobalKVAttention)
    adapter.layers = [5, 7, 9, 11]
    adapter.global_kv_cache = {layer: (torch.zeros(1), torch.zeros(1)) for layer in adapter.layers}
    adapter.cache_build_count = 1
    adapter.anchor_positions = tuple(range(50))
    adapter.anchor_frames = 50
    adapter.calls, adapter.events = [], []
    adapter.reference_indices = adapter.token_indices_by_layer = None
    adapter._build_cache_once = lambda *args: pytest.fail("begin_window rebuilt anchor cache")
    images = torch.zeros(1, 32, 3, 448, 560)
    identities = []
    for window in range(3):
        adapter.begin_window(metadata(window), images)
        identities.append(adapter.anchor_positions)
        assert adapter.cache_build_count == 1
    assert identities[0] == identities[1] == identities[2]


def test_anchor_token_indices_have_expected_patch_and_special_budgets():
    adapter = object.__new__(HybridGlobalKVAttention)
    adapter.schedule = {i: {"global_stride": 1 if i == 11 else 2} for i in (5, 7, 9, 11)}
    adapter.anchor_frames, adapter.grid_height, adapter.grid_width = 50, 32, 40
    adapter.tokens_per_frame, adapter.special_tokens = 1281, 1
    for layer in (5, 7, 9, 11):
        indices = adapter._anchor_token_indices(layer, torch.device("cpu"))
        expected_patches = 64000 if layer == 11 else 16000
        assert indices.shape == (1, 50 + expected_patches)
