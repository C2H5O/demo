"""Current H first14-keep2/final8 contracts; supplied for later server execution."""
from dataclasses import replace

import pytest
import torch

from inference.da3_kv_attention import DA3KVAttention
from inference.kv_sampling import KVSamplingConfig, eligible_frame_slots, select_kv_frames
from utils.config import load_config
from test_highlight_kv import SCORES
from test_vda_role_kv import normal_window, tiny_da3


def current_h():
    return KVSamplingConfig.from_mapping(load_config("configs/baselines/H.yaml")["kv_sampling"])


def test_first_fourteen_global_lowest_two_and_all_original_recent_eight():
    scores = dict(zip(range(10, 32), SCORES))
    scores.update(dict.fromkeys(range(24, 32), 1.))
    audit = {}
    config = current_h()
    assert config.frame_budget(32) == 20
    selected = select_kv_frames(normal_window(), config, 20, scores, audit)
    assert selected == [*range(10), 12, 21, *range(24, 32)]
    assert audit["new_temporal_buckets"] == [list(range(10, 24)), list(range(24, 32))]
    assert audit["bucket_keep_counts"] == [2, 8]
    assert audit["bucket_selected_slots"] == [12, 21, *range(24, 32)]


def test_prefix_ties_prefer_earlier_slots_and_can_select_adjacent_frames():
    scores = dict.fromkeys(range(32), .5)
    assert select_kv_frames(normal_window(), current_h(), 20, scores) == [*range(12), *range(24, 32)]
    scores.update({22: 0., 23: 0.})
    assert select_kv_frames(normal_window(), current_h(), 20, scores) == [*range(10), *range(22, 32)]


@pytest.mark.parametrize("count,expected_new", [(0, 0), (1, 1), (5, 2), (8, 2),
                                                (14, 2), (15, 3), (18, 6), (21, 9), (22, 10)])
def test_tail_preserves_original_fourteen_eight_boundary(count, expected_new):
    metadata = replace(normal_window(), is_padding=(False,) * (10 + count) + (True,) * (22 - count))
    audit = {}
    selected = select_kv_frames(metadata, current_h(), 20, dict.fromkeys(range(32), 0.), audit)
    assert len(selected) == 10 + expected_new
    assert audit["bucket_keep_counts"] == [min(2, count), max(0, count - 14)]
    assert audit["new_temporal_buckets"][1] == list(range(24, 10 + count))
    assert len(selected) == len(set(selected))
    assert set(selected) <= set(eligible_frame_slots(metadata))


def test_duplicate_sources_do_not_shift_split_or_repeat_history():
    metadata = normal_window()
    positions = list(metadata.frame_positions)
    positions[11] = positions[10]
    positions[12] = positions[0]
    positions[25] = positions[24]
    metadata = replace(metadata, frame_positions=tuple(positions))
    audit = {}
    selected = select_kv_frames(metadata, current_h(), 20, dict.fromkeys(range(32), 0.), audit)
    assert selected == [*range(10), 10, 13, 24, *range(26, 32)]
    assert audit["bucket_keep_counts"] == [2, 7]
    assert len({metadata.frame_positions[i] for i in selected}) == len(selected)


def test_first_window_remains_sixteen_one_per_pair():
    config = current_h()
    assert config.frame_budget(32, first_window=True) == 16
    audit = {}
    scores = {i: (31 - i) / 32 for i in range(32)}
    assert select_kv_frames(normal_window(True), config, 16, scores, audit) == list(range(1, 32, 2))
    assert audit["bucket_keep_counts"] == [1] * 16
    assert audit["new_temporal_buckets"] == [[i, i + 1] for i in range(0, 32, 2)]


@pytest.mark.parametrize("changes", [{"prefix_frames": 13}, {"prefix_keep": 3},
                                      {"recent_frames": 7}, {"num_buckets": 6},
                                      {"prefix_frames": 14.}, {"keep_counts": [2, 8]}])
def test_bad_prefix_recent_config_fails_before_inference(changes):
    config = current_h()
    with pytest.raises(ValueError):
        replace(config, bucket_highlight={**config.bucket_highlight, **changes}).frame_budget(32)


@pytest.mark.parametrize("strategy", ["first", "middle", "saddle_balanced"])
def test_adapter_first16_later20_keeps_full_q_and_correct_recent_frame_identity(strategy):
    model, _ = tiny_da3(strategy)
    adapter = DA3KVAttention(model, current_h(), 32)
    original = adapter.attend
    observed = []
    def inspect(layer, kernel, query, key, value, **kwargs):
        def checked(q, k, v, **kw):
            tokens = 96 if adapter.metadata.first_window else 112
            assert tuple(q.shape) == (1, 2, 160, 12)
            assert tuple(k.shape) == tuple(v.shape) == (1, 2, tokens, 12)
            torch.testing.assert_close(q, query, rtol=0, atol=0)
            indices = adapter.token_indices[:, None, :, None].expand(1, 2, -1, 12)
            torch.testing.assert_close(k, key.gather(2, indices), rtol=0, atol=0)
            torch.testing.assert_close(v, value.gather(2, indices), rtol=0, atol=0)
            observed.append(tokens)
            return kernel(q, k, v, **kw)
        return original(layer, checked, query, key, value, **kwargs)
    adapter.attend = inspect
    with torch.inference_mode(), adapter:
        images = torch.zeros(1, 32, 3, 28, 28)
        for first in (True, False):
            adapter.begin_window(normal_window(first), images)
            expected = list(range(0, 32, 2)) if first else [*range(12), *range(24, 32)]
            assert adapter.selected == expected
            features, _ = model(images)
            assert features[0][0].shape[:3] == (1, 32, 4)
            adapter.finish_window()
    assert observed == [96, 112]
    audit = adapter.summary()["kv_selection_examples"][1]
    assert audit["selected_role_counts"] == {"key": 2, "overlap": 8, "new": 10}
    assert audit["bucket_keep_counts"] == [2, 8]
    assert audit["total_kv_frame_count"] == 20 and audit["kv_token_count"] == 112
