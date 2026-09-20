"""Two-level role/layer sparse-KV contracts for H; no checkpoint or video data."""
from dataclasses import replace

import pytest
import torch

from inference.da3_kv_attention import DA3KVAttention, frame_slots_to_token_indices
from inference.kv_sampling import (
    KVSamplingConfig, build_role_layer_patch_indices, build_spatial_patch_indices,
    provider_spatial_plan,
    select_kv_frames, spatial_strides_for_layer, temporal_buckets,
)
from utils.config import load_config
from test_vda_role_kv import normal_window, tiny_da3


STANDARD_BUCKETS = [list(range(10, 13)), list(range(13, 16)), list(range(16, 19)),
                    list(range(19, 22)), list(range(22, 25)), list(range(25, 28)),
                    list(range(28, 32))]
TIE_WINNERS = [10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29]


def current_h():
    return KVSamplingConfig.from_mapping(load_config("configs/baselines/H.yaml")["kv_sampling"])


def test_h_config_and_twenty_two_new_frames_form_seven_contiguous_buckets():
    raw = load_config("configs/baselines/H.yaml")
    assert (raw["dataset"]["image_height"], raw["dataset"]["image_width"]) == (448, 560)
    assert (raw["student"]["image_height"], raw["student"]["image_width"]) == (448, 560)
    assert (raw["inference"]["image_height"], raw["inference"]["image_width"]) == (224, 280)
    assert (raw["vda_evaluation"]["evaluation_height"],
            raw["vda_evaluation"]["evaluation_width"]) == (224, 280)
    assert raw["vda_evaluation"]["tae"]["enabled"] is False
    config = current_h()
    assert config.method == "role_layer_spatial_kv"
    assert config.frame_budget(32) == 24
    assert config.frame_budget(32, first_window=True) == 16
    assert (config.key_frames, config.overlap_frames, config.new_frames,
            config.selected_new_frames) == (2, 8, 22, 14)
    assert temporal_buckets(range(10, 32), 7) == STANDARD_BUCKETS
    assert [len(bucket) for bucket in STANDARD_BUCKETS] == [3, 3, 3, 3, 3, 3, 4]


def test_each_bucket_keeps_two_lowest_scores_and_ties_are_stable():
    config = current_h()
    scores = dict.fromkeys(range(32), .5)
    audit = {}
    selected = select_kv_frames(normal_window(), config, 24, scores, audit)
    assert selected == [*range(10), *TIE_WINNERS]
    assert audit["new_temporal_buckets"] == STANDARD_BUCKETS
    assert audit["bucket_keep_counts"] == [2] * 7
    assert audit["bucket_selected_slots"] == TIE_WINNERS
    for bucket in STANDARD_BUCKETS:
        scores[bucket[-1]] = 0.
        scores[bucket[-2]] = .1
    selected = select_kv_frames(normal_window(), config, 24, scores)
    expected_new = sorted(slot for bucket in STANDARD_BUCKETS for slot in bucket[-2:])
    assert selected == [*range(10), *expected_new]


def test_standard_window_has_exactly_twenty_four_providers_and_eight_q_only_new_frames():
    selected = select_kv_frames(normal_window(), current_h(), 24,
                                dict.fromkeys(range(32), 0.))
    assert len(selected) == 24
    assert selected[:10] == list(range(10))
    assert len(set(range(10, 32)) - set(selected)) == 8


def test_spatial_indices_are_original_grid_stride_two_and_follow_layer_role_schedule():
    assert build_spatial_patch_indices(3, 5, 2) == (0, 2, 4, 10, 12, 14)
    config = current_h()
    assert spatial_strides_for_layer(5, config.spatial_sampling) == {
        "key": 1, "overlap": 2, "new": 2}
    assert spatial_strides_for_layer(7, config.spatial_sampling) == {
        "key": 1, "overlap": 2, "new": 2}
    assert spatial_strides_for_layer(9, config.spatial_sampling) == {
        "key": 1, "overlap": 1, "new": 1}
    selected = [0, 2, 10]
    early = build_role_layer_patch_indices(
        normal_window(), selected, 5, 16, 20, config.spatial_sampling)
    late = build_role_layer_patch_indices(
        normal_window(), selected, 9, 16, 20, config.spatial_sampling)
    assert len(early[0]) == 320 and len(early[2]) == len(early[10]) == 80
    assert all(len(late[slot]) == 320 for slot in selected)


def test_four_phases_partition_the_grid_with_equal_budget():
    phases = [set(build_spatial_patch_indices(16, 20, 2, row, col))
              for row, col in ((0, 0), (0, 1), (1, 0), (1, 1))]
    assert all(len(phase) == 80 for phase in phases)
    assert len(set.union(*phases)) == 320
    assert all(not phases[a] & phases[b] for a in range(4) for b in range(a + 1, 4))
    with pytest.raises(ValueError, match="offsets"):
        build_spatial_patch_indices(16, 20, 1, 0, 1)


def test_rank_phases_rotate_across_blocks_and_ignore_slot_modulo():
    config = current_h()
    selected = [0, 1, 2, 3, 10, 12, 15, 18]
    first = normal_window(True)
    for layer, expected in ((5, [0, 1, 2, 3, 0, 1, 2, 3]),
                            (7, [2, 3, 0, 1, 2, 3, 0, 1])):
        plan = provider_spatial_plan(first, selected, layer, config.spatial_sampling)
        assert [item["phase"] for item in plan] == expected
        assert [item["provider_rank"] for item in plan] == list(range(8))
        patches = build_role_layer_patch_indices(first, selected, layer, 16, 20,
                                                  config.spatial_sampling)
        assert all(len(patches[slot]) == 80 for slot in selected)


def test_key_is_full_and_late_blocks_are_dense_for_all_roles():
    config = current_h()
    selected = [0, 1, 2, 3, 10]
    for layer in (5, 7):
        plan = provider_spatial_plan(normal_window(), selected, layer, config.spatial_sampling)
        patches = build_role_layer_patch_indices(normal_window(), selected, layer,
                                                  16, 20, config.spatial_sampling)
        assert [(item["stride"], item["phase"]) for item in plan[:2]] == [(1, None)] * 2
        assert all(len(patches[slot]) == 320 for slot in selected[:2])
        assert all(len(patches[slot]) == 80 for slot in selected[2:])
    for layer in (9, 11):
        plan = provider_spatial_plan(normal_window(), selected, layer, config.spatial_sampling)
        patches = build_role_layer_patch_indices(normal_window(), selected, layer,
                                                  16, 20, config.spatial_sampling)
        assert all(item["stride"] == 1 and item["phase"] is None for item in plan)
        assert all(len(patches[slot]) == 320 for slot in selected)


def test_fixed_pattern_keeps_its_original_lattice():
    options = {key: value for key, value in current_h().spatial_sampling.items()
               if key not in {"pattern", "staggered"}}
    selected = [0, 2, 10]
    plan = provider_spatial_plan(normal_window(), selected, 5, options)
    patches = build_role_layer_patch_indices(normal_window(), selected, 5, 16, 20, options)
    assert all(item["phase"] is None for item in plan)
    assert patches[2] == patches[10] == build_spatial_patch_indices(16, 20, 2)


def test_token_mapping_keeps_every_special_and_no_patch_from_unselected_frames():
    selected = [0, 2, 4]
    patches = {0: (0, 1, 2, 3), 2: (0, 2), 4: (1, 3)}
    references = torch.tensor([3, 0], dtype=torch.long)
    indices = frame_slots_to_token_indices(
        selected, num_frames=5, tokens_per_frame=7, special_tokens=3,
        reference_indices=references, patch_indices_by_slot=patches)
    assert indices.shape == (2, 5 * 3 + 4 + 2 + 2)
    all_specials = {frame * 7 + token for frame in range(5) for token in range(3)}
    for batch, reference in enumerate(references.tolist()):
        order = [reference] + [slot for slot in range(5) if slot != reference]
        actual = set(indices[batch].tolist())
        assert all_specials <= actual
        expected_patches = {
            order.index(slot) * 7 + 3 + patch
            for slot, offsets in patches.items() for patch in offsets
        }
        assert actual - all_specials == expected_patches
        assert all(order.index(slot) * 7 + token not in actual
                   for slot in {1, 3} for token in range(3, 7))


def test_staggered_original_patch_lattices_survive_reference_permutation():
    metadata = normal_window()
    selected = [0, 2, 10]
    references = torch.tensor([3, 10], dtype=torch.long)
    for layer in (5, 7):
        patches = build_role_layer_patch_indices(
            metadata, selected, layer, 16, 20, current_h().spatial_sampling)
        actual = frame_slots_to_token_indices(
            selected, num_frames=32, tokens_per_frame=321, special_tokens=1,
            reference_indices=references, patch_indices_by_slot=patches)
        assert actual.shape == (2, 32 + 320 + 80 + 80)
        for batch, reference in enumerate(references.tolist()):
            order = [reference] + [slot for slot in range(32) if slot != reference]
            expected = {internal * 321 for internal in range(32)}
            expected.update(order.index(slot) * 321 + 1 + patch
                            for slot in selected for patch in patches[slot])
            assert set(actual[batch].tolist()) == expected


@pytest.mark.parametrize("strategy", ["first", "middle", "saddle_balanced"])
def test_real_sdpa_gets_layer_specific_sparse_kv_with_full_q_and_output(strategy):
    model, _ = tiny_da3(strategy, depth=12)
    adapter = DA3KVAttention(model, current_h(), 32)
    original = adapter.attend
    observed = []

    def inspect(layer, kernel, query, key, value, **kwargs):
        full_query = query.clone()

        def checked(q, k, v, **kernel_kwargs):
            expected_kv = 62 if layer in (5, 7) else 128
            assert tuple(q.shape) == (1, 2, 160, 12)
            assert tuple(k.shape) == tuple(v.shape) == (1, 2, expected_kv, 12)
            torch.testing.assert_close(q, full_query, rtol=0, atol=0)
            indices = adapter.token_indices_by_layer[layer][:, None, :, None].expand(1, 2, -1, 12)
            torch.testing.assert_close(k, key.gather(2, indices), rtol=0, atol=0)
            torch.testing.assert_close(v, value.gather(2, indices), rtol=0, atol=0)
            result = kernel(q, k, v, **kernel_kwargs)
            assert result.shape == q.shape
            observed.append((layer, q.shape[-2], k.shape[-2]))
            return result

        return original(layer, checked, query, key, value, **kwargs)

    adapter.attend = inspect
    with torch.inference_mode(), adapter:
        images = torch.zeros(1, 32, 3, 28, 28)
        adapter.begin_window(normal_window(), images)
        assert adapter.selected == [*range(10), *TIE_WINNERS]
        features, _ = model(images)
        assert features[0][0].shape[:3] == (1, 32, 4)
        adapter.finish_window()
    assert observed == [(5, 160, 62), (7, 160, 62), (9, 160, 128), (11, 160, 128)]
    audit = adapter.summary()["kv_selection_examples"][0]
    assert audit["selected_role_counts"] == {"key": 2, "overlap": 8, "new": 14}
    assert audit["total_kv_frame_count"] == 24
    assert audit["kv_token_count_by_layer"] == {5: 62, 7: 62, 9: 128, 11: 128}
    assert audit["q_token_count"] == 160
    assert audit["special_tokens_per_frame"] == 1


def test_first_window_keeps_existing_sixteen_bucket_winners_then_uses_layer_schedule():
    config = current_h()
    scores = {slot: (31 - slot) / 32 for slot in range(32)}
    audit = {}
    selected = select_kv_frames(normal_window(True), config, 16, scores, audit)
    assert selected == list(range(1, 32, 2))
    assert audit["new_temporal_buckets"] == [[slot, slot + 1] for slot in range(0, 32, 2)]
    assert audit["bucket_keep_counts"] == [1] * 16
    early = build_role_layer_patch_indices(
        normal_window(True), selected, 5, 2, 2, config.spatial_sampling)
    late = build_role_layer_patch_indices(
        normal_window(True), selected, 9, 2, 2, config.spatial_sampling)
    assert all(len(offsets) == 1 for offsets in early.values())
    assert all(len(offsets) == 4 for offsets in late.values())
    assert [item["phase"] for item in provider_spatial_plan(
        normal_window(True), selected, 5, config.spatial_sampling)] == list(range(4)) * 4
    assert [item["phase"] for item in provider_spatial_plan(
        normal_window(True), selected, 7, config.spatial_sampling)] == [2, 3, 0, 1] * 4


@pytest.mark.parametrize("change", [
    {"retention_ratio": .625}, {"key_frames": 1}, {"overlap_frames": 7},
    {"new_frames": 21}, {"selected_new_frames": 13},
    {"first_window_num_frames": 15}, {"first_window_num_buckets": 15},
    {"new_frame_selection": {"method": "temporal_bucket_highlight"}},
    {"special_tokens": {"keep_all": False}},
])
def test_invalid_two_level_contract_fails_before_inference(change):
    with pytest.raises(ValueError):
        replace(current_h(), **change).frame_budget(32)
