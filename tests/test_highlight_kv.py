"""Small CPU contracts for H; no dataset, checkpoints or GPU inference."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from datasets.highlight import HighlightDetectionConfig, SpecularHighlightProcessor
from inference.da3_kv_attention import DA3KVAttention
from inference.kv_sampling import KVSamplingConfig, select_kv_frames
from utils.config import load_config
from test_vda_role_kv import normal_window, tiny_da3


SCORES = [.18, .27, .05, .31, .22, .08, .14, .29, .06, .16, .24,
          .04, .19, .25, .13, .35, .21, .03, .17, .28, .07, .20]
EXPECTED = [*range(10), 12, 15, 18, 21, 27, 30]


def legacy_highlight_config():
    # Pin the old experiment independently of the current H selection method.
    return KVSamplingConfig(enabled=True, method="vda_role_highlight", retention_ratio=.5,
                            key_frames=2, overlap_frames=8, new_frames=6,
                            first_window_method="highlight", first_window_num_frames=16)


def test_exact_example_and_deterministic_ties():
    config = legacy_highlight_config()
    assert config.frame_budget(32) == 16
    scores = dict(zip(range(10, 32), SCORES))
    assert select_kv_frames(normal_window(), config, 16, scores) == EXPECTED
    assert select_kv_frames(normal_window(), config, 16, dict.fromkeys(range(32), 0.0)) == list(range(16))
    # Historical scores cannot delete any key/overlap frame.
    scores.update(dict.fromkeys(range(10), 1.0))
    assert select_kv_frames(normal_window(), config, 16, scores) == EXPECTED
    for name, budget in (("F", 8), ("G", 8), ("H", 20)):
        policy = KVSamplingConfig.from_mapping(load_config(f"configs/baselines/{name}.yaml")["kv_sampling"])
        assert policy.frame_budget(32) == budget
    with pytest.raises(ValueError):
        replace(config, overlap_frames=2, new_frames=12).frame_budget(32)


def test_first_window_tail_and_invalid_scores():
    config = legacy_highlight_config()
    scores = {i: (31-i) / 32 for i in range(32)}
    assert select_kv_frames(normal_window(True), config, 16, scores) == list(range(16, 32))
    metadata = replace(normal_window(), is_padding=(False,) * 13 + (True,) * 19)
    assert select_kv_frames(metadata, config, 16, scores) == list(range(13))
    first = replace(normal_window(True), frame_positions=(0,) * 32,
                    is_padding=(False,) + (True,) * 31)
    assert select_kv_frames(first, config, 16, {0: .3}) == [0]
    for invalid in (None, {}, {**scores, 10: float("nan")}, {**scores, 10: -1}):
        with pytest.raises(ValueError, match="pixel ratio"):
            select_kv_frames(normal_window(), config, 16, invalid)


def test_detector_mask_matches_original_pipeline_without_output_inpainting(monkeypatch):
    pytest.importorskip("cv2")
    detector = SpecularHighlightProcessor(HighlightDetectionConfig())
    image = np.random.default_rng(7).integers(0, 256, (40, 56, 3), dtype=np.uint8)
    # Literal detector operations from the pre-change process_numpy.
    red, green, blue = (image[..., i].astype(np.float32) for i in range(3))
    luminance = .2989 * red + .5870 * green + .1140 * blue
    absolute = detector._module1(luminance, green, blue, 250.)
    candidate = detector._module1(luminance, green, blue, 230.)
    relative = detector._relative_mask(detector._fill_components(candidate, image), red, green, blue)
    expected = detector._classify(detector._ellipse(
        ((absolute | relative) & candidate).astype(np.uint8) * 255, 2, "dilate"))
    mask, inpainted = detector.process_numpy(image)
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(inpainted, detector._inpaint(expected, image) / 255.)
    def forbidden(*args):
        raise AssertionError("Selector must not generate inpainted output")
    monkeypatch.setattr(detector, "_inpaint", forbidden)
    np.testing.assert_array_equal(detector.detect_mask_numpy(image), expected)


@pytest.mark.parametrize("strategy", ["first", "middle", "saddle_balanced"])
def test_h_actual_rectangular_sdpa_and_score_audit(strategy, monkeypatch):
    model, _ = tiny_da3(strategy)
    adapter = DA3KVAttention(model, legacy_highlight_config(), 32)
    calls = []
    def synthetic_mask(image):
        index = len(calls)
        calls.append(index)
        mask = np.zeros(100, dtype=np.float32)
        mask[:round(SCORES[index] * 100)] = 1
        return mask
    monkeypatch.setattr(adapter.highlight_processor, "detect_mask_numpy", synthetic_mask)
    original = adapter.attend
    observed = []
    def observe(layer, kernel, query, key, value, **kwargs):
        def checked(q, k, v, **kw):
            assert q.shape == (1, 2, 160, 12)
            assert k.shape == v.shape == (1, 2, 96, 12)
            torch.testing.assert_close(q, query, rtol=0, atol=0)
            ids = adapter.token_indices[:, None, :, None].expand(1, 2, -1, 12)
            torch.testing.assert_close(k, key.gather(2, ids), rtol=0, atol=0)
            torch.testing.assert_close(v, value.gather(2, ids), rtol=0, atol=0)
            output = kernel(q, k, v, **kw)
            oracle = ((q / 12**.5) @ k.transpose(-2, -1)).softmax(-1) @ v
            torch.testing.assert_close(output, oracle, rtol=2e-5, atol=2e-6)
            observed.append(tuple(k.shape))
            return output
        return original(layer, checked, query, key, value, **kwargs)
    adapter.attend = observe
    with torch.inference_mode(), adapter:
        images = torch.rand(1, 32, 3, 28, 28)
        adapter.begin_window(normal_window(), images)
        assert adapter.selected == EXPECTED and len(calls) == 22
        features, _ = model(images)
        assert features[0][0].shape[:3] == (1, 32, 4)
        adapter.finish_window()
    assert observed == [(1, 2, 96, 12)]
    audit = adapter.summary()["kv_selection_examples"][0]
    assert audit["selected_role_counts"] == {"key": 2, "overlap": 8, "new": 6}
    assert audit["new_candidate_count"] == 22
