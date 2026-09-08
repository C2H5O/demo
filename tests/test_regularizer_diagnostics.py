import copy
import json
import math
from pathlib import Path

import pytest
import torch

from audit_training_losses import audit
from losses.direct_teacher_distillation_loss import compute_highlight_surface_loss, compute_highlight_aware_smoothness_loss
from losses.regularizer_diagnostics import regularizer_diagnostics, loss_share_logs
from utils.da3_geometry import depth_intrinsics_to_local_points
from utils.config import load_config


def plane(angle=30., depth_scale=1.):
    y, x = torch.meshgrid(torch.arange(7.), torch.arange(9.), indexing="ij")
    k = torch.tensor([[80., 0, 4.], [0, 80., 3.], [0, 0, 1.]]).reshape(1, 1, 3, 3)
    depth = (depth_scale / (1 + math.tan(math.radians(angle)) * (x - 4) / 80))[None, None].requires_grad_()
    points = depth_intrinsics_to_local_points(depth, k)
    mask = torch.zeros(1, 1, 1, 7, 9, dtype=torch.bool)
    mask[..., 3, 4] = True
    return depth, points, mask, torch.full((1, 1, 3, 7, 9), 0.5)


@pytest.mark.parametrize("angle", [0., 15., 30., 60.])
def test_highlight_remains_the_existing_soft_parallel_prior(angle):
    _, points, mask, _ = plane(angle)
    loss = compute_highlight_surface_loss(points, mask)
    assert loss.item() == pytest.approx((1 - math.cos(math.radians(angle))) ** 2, abs=2e-6)


def test_weight_change_scales_depth_gradient_without_changing_formula():
    depth, points, mask, _ = plane()
    loss = compute_highlight_surface_loss(points, mask)
    (loss * 0.01).backward(retain_graph=True)
    original = depth.grad.clone()
    depth.grad.zero_()
    (loss * 0.03).backward()
    assert original.abs().sum() > 0
    torch.testing.assert_close(depth.grad, 3 * original)


@pytest.mark.parametrize("angle", [0., 3., 5., 7., 10., 30.])
def test_full_ten_degree_cone_has_five_degree_half_angle(angle):
    depth, points, mask, _ = plane(angle)
    config = load_config("configs/baselines/E.yaml")
    half_angle = config["loss"]["highlight_cone_full_angle_degrees"] / 2
    assert half_angle == 5.
    softness = config["loss"]["highlight_softness"]
    loss = compute_highlight_surface_loss(points, mask, mode="angular_soft_margin",
                                         margin_degrees=half_angle, softness=softness)
    excess = math.cos(math.radians(5)) - math.cos(math.radians(angle))
    expected = (softness * torch.nn.functional.softplus(torch.tensor(excess / softness))).square()
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=1e-3)
    loss.backward()
    assert torch.isfinite(depth.grad).all()


def test_baselines_preserve_comparison_and_memory_contracts():
    configs = {letter: load_config("configs/baselines/{}.yaml".format(letter)) for letter in "ABCDEFG"}
    for letter, config in configs.items():
        assert config["dataloader"]["batch_size"] == 1
        assert config["training"]["gradient_accumulation_steps"] == 4
        assert config["student"]["head_chunk_size"] == 1
        assert config["dataset"]["root"].endswith("processed/SCARED")
        assert config["attention_distill"]["enabled"] == (letter in "CEFG")
        assert config["loss"]["highlight_mode"] == ("angular_soft_margin" if letter in "DEFG" else "legacy")
        assert config["loss"]["lambda_highlight"] == .01
        assert config["loss"]["lambda_smooth"] == .1
        assert config["experiment"]["training_required"] == (letter in "BCDE")


def test_smoothness_reaches_depth_and_is_scale_invariant():
    depth, points, mask, clean = plane()
    value = compute_highlight_aware_smoothness_loss(points, clean, mask)
    value.backward()
    assert value > 0 and depth.grad.abs().sum() > 0
    assert torch.isfinite(depth.grad).all()
    _, scaled, _, _ = plane(depth_scale=2.)
    torch.testing.assert_close(value, compute_highlight_aware_smoothness_loss(scaled, clean, mask))


def test_empty_masks_and_normal_conditioning_are_explicit():
    depth, points, mask, clean = plane(depth_scale=0.01)
    stats = regularizer_diagnostics(points, mask, clean)
    assert stats["stats/highlight_valid_pixel_count"] == 1
    assert stats["stats/highlight_normal_short_fraction"] == 1
    assert stats["stats/highlight_normal_view_angle_valid"] == 0
    assert depth.grad is None
    mask.zero_()
    stats = regularizer_diagnostics(points, mask, clean)
    assert stats["stats/highlight_empty"] == 1
    value = compute_highlight_surface_loss(points, mask)
    value.backward()
    assert value == 0 and torch.isfinite(depth.grad).all()


def test_final_loss_shares_include_attention_in_denominator():
    stats = loss_share_logs({"loss/total": 10, "loss/depth_weighted": 5,
                            "loss/camera_weighted": 3, "loss/highlight_weighted": .1,
                            "loss/smooth_weighted": .2, "loss/attention_weighted": 1.7})
    assert stats["stats/loss_share_highlight"] == .01
    assert stats["stats/loss_share_attention"] == pytest.approx(.17)
    assert stats["stats/loss_sum_residual"] == pytest.approx(0, abs=1e-12)


def test_audit_reports_conflicting_resume_records_and_keeps_latest(tmp_path):
    def record(step, total):
        return {"phase": "train", "epoch": 0, "global_step": step,
                "loss/total": total, "loss/highlight": .2, "loss/smooth": .01,
                "loss/depth_weighted": total - .003, "loss/camera_weighted": 0,
                "loss/highlight_weighted": .002, "loss/smooth_weighted": .001,
                "loss/attention_weighted": 0}
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (record(1, 1.), record(2, 1.), record(2, 2.))))
    result = audit(path)
    assert result["unique_steps"] == 2
    assert result["conflicting_duplicate_records"] == 1
    assert result["epochs"][0]["total_mean"] == 1.5


def test_highlight_ablation_changes_one_loss_weight_and_cannot_silently_resume():
    from trainers.direct_teacher_distillation_trainer import _check_resume_contract
    from utils.checkpoint import DIRECT_TEACHER_DISTILLATION_PROTOCOL
    from datasets.crossclip_teacher_dataset import CROSSCLIP_CACHE_PROTOCOL
    root = Path(__file__).resolve().parents[1]
    baseline = load_config(root / "configs/vggtoda3_attention_distill.yaml")
    candidate = load_config(root / "configs/vggtoda3_attention_highlight_x3.yaml")
    expected = copy.deepcopy(baseline["loss"])
    expected["lambda_highlight"] = .03
    assert candidate["loss"] == expected
    assert baseline["loss"]["lambda_highlight"] == .01
    checkpoint = {"config": baseline, "objective_protocol": DIRECT_TEACHER_DISTILLATION_PROTOCOL,
                  "cache_protocol": CROSSCLIP_CACHE_PROTOCOL}
    # The loss mismatch must be rejected before model inspection.
    with pytest.raises(ValueError, match="loss settings differ"):
        _check_resume_contract(checkpoint, candidate, None)
