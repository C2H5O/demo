"""Layer-wise spatial/temporal KV stride contracts for Baseline-H."""
from pathlib import Path

import numpy as np
import pytest
import torch

from inference.da3_kv_attention import DA3KVAttention, frame_slots_to_token_indices
from inference.kv_sampling import (
    KVSamplingConfig,
    build_spatial_patch_indices,
    resolve_layer_stride_policy,
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
    observed = []
    original = adapter.attend

    def inspect(layer, kernel, query, key, value, **kwargs):
        def checked(q, k, v, **kernel_kwargs):
            expected_kv = 64 if POLICIES[variant][layer] == "spatial" else 96
            assert q.shape[-2] == 160
            assert k.shape[-2] == v.shape[-2] == expected_kv
            result = kernel(q, k, v, **kernel_kwargs)
            assert result.shape == q.shape
            observed.append((layer, q.shape[-2], k.shape[-2]))
            return result
        return original(layer, checked, query, key, value, **kwargs)

    adapter.attend = inspect
    with torch.inference_mode(), adapter:
        adapter.begin_window(normal_window(), images)
        features, _ = model(images)
        assert features[0][0].shape[:3] == (1, 32, 4)
        adapter.finish_window()
    assert [item[0] for item in observed] == [5, 7, 9, 11]
    audit = adapter.summary()["kv_selection_examples"][0]
    for layer, policy in POLICIES[variant].items():
        item = audit["layer_stride_kv"][layer]
        assert item["q_frame_count"] == 32
        assert item["kv_patch_frame_count"] == (32 if policy == "spatial" else 16)
        assert item["spatial_patch_count_per_frame"] == (1 if policy == "spatial" else 4)


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
