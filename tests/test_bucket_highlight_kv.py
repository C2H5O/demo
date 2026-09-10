"""Bucket/lightweight H contracts, supplied for later execution on the server."""
from dataclasses import replace

import pytest
import torch
from torch.overrides import TorchFunctionMode

import inference.da3_kv_attention as attention_module
from inference.da3_kv_attention import DA3KVAttention
from inference.kv_sampling import (
    KVSamplingConfig, bucket_keep_count, eligible_frame_slots, select_bucket_highlight_frames,
    select_kv_frames, temporal_buckets,
)
from inference.lightweight_highlight import compute_lightweight_highlight_scores
from utils.config import load_config
from test_highlight_kv import SCORES, legacy_highlight_config
from test_vda_role_kv import normal_window, tiny_da3


BUCKETS = [[10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20],
           [21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]
KEEP_COUNTS = [1, 1, 1, 1, 2, 4]
WINNERS = [12, 15, 18, 21, 24, 27, 28, 29, 30, 31]


def bucket_config():
    # Pin the previous six-bucket ablation independently of the current H policy.
    config = KVSamplingConfig.from_mapping(load_config("configs/baselines/H.yaml")["kv_sampling"])
    return replace(config, bucket_highlight={"num_buckets": 6, "keep_policy": "fixed", "keep_counts": KEEP_COUNTS})


def test_normal_window_keeps_history_and_ten_new_with_fixed_111124_counts():
    config = bucket_config()
    assert config.method == "vda_role_bucket_highlight" and config.frame_budget(32) == 20
    assert config.frame_budget(32, first_window=True) == 16
    assert config.bucket_highlight == {"num_buckets": 6, "keep_policy": "fixed", "keep_counts": KEEP_COUNTS}
    assert config.new_frames == 10 and config.first_window_num_buckets == 16
    scores = {**dict.fromkeys(range(10), 1.), **dict(zip(range(10, 32), SCORES))}
    audit = {}
    selected = select_kv_frames(normal_window(), config, 20, scores, audit)
    assert selected == [*range(10), *WINNERS]
    assert len(selected) == len(set(selected)) == 20
    assert audit["new_temporal_buckets"] == BUCKETS
    assert audit["bucket_selected_slots"] == WINNERS
    assert audit["bucket_keep_counts"] == KEEP_COUNTS
    for bucket, keep in zip(BUCKETS, KEEP_COUNTS):
        chosen = set(selected).intersection(bucket)
        assert len(chosen) == keep
        assert sorted(scores[slot] for slot in chosen) == sorted(scores[slot] for slot in bucket)[:keep]


def test_first_window_uses_sixteen_buckets_and_lowest_score_in_each():
    scores = {slot: (31 - slot) / 32 for slot in range(32)}
    audit = {}
    selected = select_kv_frames(normal_window(True), bucket_config(), 16, scores, audit)
    assert selected == list(range(1, 32, 2))
    assert audit["new_temporal_buckets"] == [[i, i + 1] for i in range(0, 32, 2)]
    assert audit["bucket_selected_slots"] == selected
    assert audit["bucket_keep_counts"] == [1] * 16


def test_ties_use_earlier_temporal_candidate_and_scores_can_change_winners():
    scores = dict.fromkeys(range(32), 0.)
    selected = select_kv_frames(normal_window(), bucket_config(), 20, scores)
    assert selected == [*range(10), 10, 13, 17, 21, 24, 25, 28, 29, 30, 31]
    assert select_kv_frames(normal_window(True), bucket_config(), 16, scores) == list(range(0, 32, 2))
    scores = dict.fromkeys(range(32), 1.)
    for bucket in BUCKETS:
        scores[bucket[-1]] = 0.
    assert select_kv_frames(normal_window(), bucket_config(), 20, scores) == [
        *range(10), 12, 16, 20, 23, 24, 27, 28, 29, 30, 31]


@pytest.mark.parametrize("size,expected", [(-1, 0), (0, 0), (1, 1), (2, 1),
                                           (3, 1), (4, 2), (5, 3), (6, 3), (7, 4)])
def test_bucket_keep_count_general_rule(size, expected):
    assert bucket_keep_count(size) == expected


@pytest.mark.parametrize("scores,expected", [
    ([.010, .009, .009, .012], [14, 15]),
    ([.009, .009, .009, .012], [13, 14]),
    ([.03, .01, .04, .02], [14, 16]),
])
def test_four_frame_bucket_keeps_two_lowest_with_stable_ties(scores, expected):
    selected, buckets = select_bucket_highlight_frames(
        [13, 14, 15, 16], dict(zip(range(13, 17), scores)), 1, keep_policy="size_dependent")
    assert buckets == [[13, 14, 15, 16]] and selected == expected


def test_three_frame_bucket_keeps_only_one():
    selected, _ = select_bucket_highlight_frames(
        [10, 11, 12], {10: .03, 11: .01, 12: .02}, 1, keep_policy="size_dependent")
    assert selected == [11]


def test_fixed_last_bucket_keeps_all_four_even_with_high_scores():
    scores = dict.fromkeys(range(32), 0.)
    scores.update({28: 1., 29: .9, 30: .8, 31: .7})
    selected = select_kv_frames(normal_window(), bucket_config(), 20, scores)
    assert selected[-4:] == [28, 29, 30, 31]


def test_previous_size_dependent_policy_remains_reproducible():
    config = replace(bucket_config(), bucket_highlight={"num_buckets": 6, "keep_policy": "size_dependent"})
    audit = {}
    selected = select_kv_frames(normal_window(), config, 20, dict(zip(range(10, 32), SCORES)), audit)
    assert selected == [*range(10), 12, 15, 16, 18, 19, 21, 24, 27, 28, 30]
    assert audit["bucket_keep_counts"] == [1, 2, 2, 1, 2, 2]


@pytest.mark.parametrize("counts", [None, [], [1, 1, 1], [1, 1, 1, 1, 2, 3],
                                    [1, 1, 1, 1, 1, 5], [0, 2, 1, 1, 2, 4],
                                    [True, 1, 1, 1, 2, 4], [1., 1, 1, 1, 2, 4]])
def test_fixed_counts_invalid_config_fails_before_inference(counts):
    config = replace(bucket_config(), bucket_highlight={"keep_policy": "fixed", "keep_counts": counts})
    with pytest.raises(ValueError, match="keep_counts"):
        config.frame_budget(32)


@pytest.mark.parametrize("size,quota", [(0, 6), (1, 6), (5, 6), (7, 6), (22, 6), (32, 16)])
def test_integer_bucket_partition_covers_every_candidate_once(size, quota):
    candidates = list(range(size))
    buckets = temporal_buckets(candidates, quota)
    assert len(buckets) == min(size, quota)
    assert all(buckets)
    assert [slot for bucket in buckets for slot in bucket] == candidates
    assert all(abs(len(a) - len(b)) <= 1 for a in buckets for b in buckets)


def test_tail_excludes_padding_duplicates_and_history_duplicates():
    metadata = normal_window()
    positions = list(metadata.frame_positions)
    positions[11] = positions[10]  # duplicate new source
    positions[12] = positions[0]   # history takes priority over new
    metadata = replace(metadata, frame_positions=tuple(positions),
                       is_padding=(False,) * 15 + (True,) * 17)
    audit = {}
    selected = select_kv_frames(metadata, bucket_config(), 20, dict.fromkeys(range(32), 0.), audit)
    assert selected == [*range(10), 10, 13, 14]
    assert audit["new_temporal_buckets"] == [[10], [13], [14]]
    assert len({metadata.frame_positions[slot] for slot in selected}) == len(selected)
    assert set(selected) <= set(eligible_frame_slots(metadata))
    no_new = replace(metadata, is_padding=(False,) * 10 + (True,) * 22)
    assert select_kv_frames(no_new, bucket_config(), 20, {}) == list(range(10))
    short = replace(normal_window(True), frame_positions=(0,) * 32,
                    is_padding=(False,) + (True,) * 31)
    assert select_kv_frames(short, bucket_config(), 16, {0: .5}) == [0]


def test_bucket_order_uses_source_time_but_final_indices_use_window_order():
    metadata = normal_window()
    metadata = replace(metadata, frame_positions=(*metadata.frame_positions[:10],
                                                 *reversed(metadata.frame_positions[10:])))
    audit = {}
    selected = select_kv_frames(metadata, bucket_config(), 20, dict.fromkeys(range(32), 0.), audit)
    assert audit["new_temporal_buckets"][0] == [31, 30, 29]
    assert audit["bucket_selected_slots"] == [31, 28, 24, 20, 17, 16, 13, 12, 11, 10]
    assert selected == sorted([*range(10), 31, 28, 24, 20, 17, 16, 13, 12, 11, 10])


@pytest.mark.parametrize("new_count,expected_new", [(0, 0), (1, 1), (5, 5), (7, 7),
                                                    (18, 9), (19, 10), (21, 10), (22, 10)])
def test_tail_budget_follows_bucket_sizes_instead_of_forcing_twenty(new_count, expected_new):
    metadata = replace(normal_window(), is_padding=(False,) * (10 + new_count) + (True,) * (22 - new_count))
    audit = {}
    selected = select_kv_frames(metadata, bucket_config(), 20, dict.fromkeys(range(32), 0.), audit)
    assert len(selected) == 10 + expected_new
    assert sum(audit["bucket_keep_counts"]) == expected_new
    assert len(selected) == len(set(selected))


def test_legacy_selectors_keep_their_original_policies():
    scores = dict.fromkeys(range(32), 0.)
    role = KVSamplingConfig(enabled=True)
    stride = replace(role, method="spark3r_fixed_stride")
    legacy = legacy_highlight_config()
    assert select_kv_frames(normal_window(), role, 8) == [0, 1, 2, 9, 10, 17, 24, 31]
    assert select_kv_frames(normal_window(True), role, 8) == [0, 4, 9, 13, 18, 22, 27, 31]
    assert select_kv_frames(normal_window(), stride, 8) == list(range(0, 32, 4))
    assert select_kv_frames(normal_window(True), stride, 8) == list(range(0, 32, 4))
    assert select_kv_frames(normal_window(), legacy, 16, scores) == list(range(16))
    assert select_kv_frames(normal_window(True), legacy, 16, scores) == list(range(16))


@pytest.mark.parametrize("name,enabled,budget", [("B", False, 32), ("C", False, 32), ("G", True, 8)])
def test_other_baseline_configuration_budgets_are_unchanged(name, enabled, budget):
    config = KVSamplingConfig.from_mapping(load_config(f"configs/baselines/{name}.yaml").get("kv_sampling"))
    assert config.enabled == enabled
    assert config.frame_budget(32) == budget
    assert config.frame_budget(32, first_window=True) == budget


@pytest.mark.parametrize("options", [
    {"brightness_threshold": -1}, {"brightness_threshold": float("nan")},
    {"saturation_threshold": 1.1}, {"saturation_threshold": "0.2"},
    {"downsample_factor": 0}, {"downsample_factor": 1.5},
    {"downsample_factor": True}, {"unknown": 4}, [],
])
def test_invalid_proxy_configuration_fails_at_budget_validation(options):
    with pytest.raises(ValueError):
        replace(bucket_config(), lightweight_highlight=options).frame_budget(32)


@pytest.mark.parametrize("change", [
    {"retention_ratio": .5}, {"key_frames": 1, "overlap_frames": 9},
    {"new_frames": 5}, {"first_window_num_frames": 8}, {"first_window_method": "highlight"},
    {"first_window_num_frames": 20}, {"first_window_num_buckets": 20},
    {"first_window_num_buckets": None}, {"first_window_num_buckets": 16.0},
    {"bucket_highlight": {"num_buckets": 10}}, {"bucket_highlight": {"num_buckets": 6.0}},
    {"bucket_highlight": {"keep_policy": "one"}}, {"bucket_highlight": {"unknown": 1}},
])
def test_first_and_later_budget_validation(change):
    with pytest.raises(ValueError):
        replace(bucket_config(), **change).frame_budget(32)
    with pytest.raises(ValueError):
        bucket_config().frame_budget(16)


def test_selector_rejects_a_budget_for_the_wrong_window():
    scores = dict.fromkeys(range(32), 0.)
    with pytest.raises(ValueError, match="first/later"):
        select_kv_frames(normal_window(True), bucket_config(), 20, scores)
    with pytest.raises(ValueError, match="first/later"):
        select_kv_frames(normal_window(), bucket_config(), 16, scores)


def test_bad_scores_and_bucket_arguments_fail():
    for scores in ({}, {10: float("nan")}, {10: 1.1}):
        with pytest.raises(ValueError, match="pixel ratio"):
            select_bucket_highlight_frames([10], scores, 1)
    with pytest.raises(ValueError):
        temporal_buckets([1, 1], 2)
    with pytest.raises(ValueError):
        temporal_buckets([1], -1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_proxy_brightness_saturation_pooling_and_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # White, bright saturated red, dark gray, one white pooled cell, checkerboard.
    frames = torch.zeros(5, 3, 8, 8, device=device)
    frames[0] = 1.
    frames[1, 0] = 1.
    frames[2] = .5
    frames[3, :, :4, :4] = 1.
    frames[4, :, ::2, ::2] = 1.
    frames[4, :, 1::2, 1::2] = 1.
    before = frames.clone()
    scores = compute_lightweight_highlight_scores(frames)
    assert scores.shape == (5,) and scores.dtype == torch.float32 and scores.device == frames.device
    torch.testing.assert_close(scores, torch.tensor([1., 0., 0., .25, 0.], device=device))
    torch.testing.assert_close(frames, before)
    assert compute_lightweight_highlight_scores(frames[:0]).shape == (0,)
    # With pooling disabled, the checkerboard has half specular-like pixels.
    score = compute_lightweight_highlight_scores(frames[4:5], {"downsample_factor": 1})
    torch.testing.assert_close(score, torch.tensor([.5], device=device))


def test_proxy_respects_inclusive_thresholds_and_black_epsilon():
    frames = torch.tensor([[.9, .9, .9], [1., .75, .75], [0., 0., 0.]]).reshape(3, 3, 1, 1)
    scores = compute_lightweight_highlight_scores(frames, {
        "brightness_threshold": .9, "saturation_threshold": .25, "downsample_factor": 1})
    torch.testing.assert_close(scores, torch.tensor([1., 1., 0.]))
    with pytest.raises(ValueError):
        compute_lightweight_highlight_scores(frames)  # factor 4 cannot pool 1x1


@pytest.mark.parametrize("strategy", ["first", "middle", "saddle_balanced"])
def test_adapter_variable_kv_windows_preserve_q_and_update_gather_audit(strategy, monkeypatch):
    from datasets.highlight import SpecularHighlightProcessor
    def forbidden(*args, **kwargs):
        raise AssertionError("New H must not instantiate PC-Depth or detect individual frames")
    monkeypatch.setattr(SpecularHighlightProcessor, "__init__", forbidden)
    model, _ = tiny_da3(strategy)
    config = replace(bucket_config(), debug=True, debug_max_windows=3)
    adapter = DA3KVAttention(model, config, 32)
    assert adapter.highlight_processor is None
    batches, transfers = [], []
    scorer = attention_module.compute_lightweight_highlight_scores
    def tracked(frames, options):
        batches.append(tuple(frames.shape))
        return scorer(frames, options)
    monkeypatch.setattr(attention_module, "compute_lightweight_highlight_scores", tracked)

    class TransferAudit(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func.__name__ in {"cpu", "item", "numpy"}:
                assert func.__name__ == "cpu"  # never per-frame item/numpy
                transfers.append(tuple(args[0].shape))
            return func(*args, **(kwargs or {}))

    original = adapter.attend
    observed = []
    def observe(layer, kernel, query, key, value, **kwargs):
        def checked(q, k, v, **kw):
            assert tuple(q.shape) == (1, 2, 160, 12)
            expected_tokens = 96 if adapter.metadata.first_window else 112
            assert tuple(k.shape) == tuple(v.shape) == (1, 2, expected_tokens, 12)
            torch.testing.assert_close(q, query, rtol=0, atol=0)
            ids = adapter.token_indices[:, None, :, None].expand(1, 2, -1, 12)
            torch.testing.assert_close(k, key.gather(2, ids), rtol=0, atol=0)
            torch.testing.assert_close(v, value.gather(2, ids), rtol=0, atol=0)
            observed.append(tuple(k.shape))
            return kernel(q, k, v, **kw)
        return original(layer, checked, query, key, value, **kwargs)
    adapter.attend = observe
    with torch.inference_mode(), adapter:
        images = torch.zeros(1, 32, 3, 28, 28)
        for window_id in range(3):
            with TransferAudit():
                adapter.begin_window(replace(normal_window(first=window_id == 0), window_id=window_id), images)
            if window_id == 0:
                assert adapter.budget == 16 and adapter.selected == list(range(0, 32, 2))
            else:
                assert adapter.budget == 20
                assert adapter.selected == [*range(10), 10, 13, 17, 21, 24, 25, 28, 29, 30, 31]
            features, _ = model(images)
            assert features[0][0].shape[:3] == (1, 32, 4)
            adapter.finish_window()
    assert batches == [(32, 3, 28, 28), (22, 3, 28, 28), (22, 3, 28, 28)]
    assert transfers == [(32,), (22,), (22,)]
    assert observed == [(1, 2, 96, 12), (1, 2, 112, 12), (1, 2, 112, 12)]
    audits = adapter.summary()["kv_selection_examples"]
    assert len(audits) == 3
    first = audits[0]
    assert first["selected_role_counts"] == {"key": 0, "overlap": 0, "new": 16}
    assert first["total_kv_frame_count"] == 16 and first["kv_token_count"] == 96
    assert first["q_token_count"] == 160 and first["kv_patch_token_count"] == 64
    assert first["frame_retention_ratio"] == .5 and first["token_retention_ratio"] == 96 / 160
    assert first["bucket_keep_counts"] == [1] * 16 and first["new_candidate_count"] == 32
    for audit in audits[1:]:
        assert audit["selected_role_counts"] == {"key": 2, "overlap": 8, "new": 10}
        assert audit["total_kv_frame_count"] == 20 and audit["kv_token_count"] == 112
        assert audit["q_token_count"] == 160 and audit["kv_patch_token_count"] == 80
        assert audit["frame_retention_ratio"] == .625 and audit["token_retention_ratio"] == 112 / 160
        assert audit["new_temporal_buckets"] == BUCKETS
        assert audit["bucket_keep_counts"] == KEEP_COUNTS
        assert audit["highlight_score_type"] == "gpu_brightness_low_saturation_ratio"
        assert audit["new_candidate_count"] == 22
