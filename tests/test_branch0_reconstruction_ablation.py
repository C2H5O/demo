import numpy as np
import torch

from diagnostics.ablate_branch0_reconstruction import (
    LocalDPTInputCapture,
    _requested_modes,
    diagnose_ablation,
    lowpass_correlation,
    phase_tied_weight,
    reconstruct_branch0,
    relative_reduction,
)


def test_phase_tied_weight_shares_all_spatial_mappings_without_mutation() -> None:
    torch.manual_seed(0)
    module = torch.nn.ConvTranspose2d(2, 2, kernel_size=4, stride=4)
    original = module.weight.detach().clone()

    tied = phase_tied_weight(module)

    assert tied.shape == original.shape
    assert torch.allclose(tied, tied[:, :, :1, :1].expand_as(tied))
    assert torch.allclose(tied[:, :, 0, 0], original.mean(dim=(-2, -1)))
    assert torch.equal(module.weight.detach(), original)


def test_three_reconstruction_modes_have_same_shape_and_leave_weight_unchanged() -> None:
    torch.manual_seed(0)
    module = torch.nn.ConvTranspose2d(2, 2, kernel_size=4, stride=4)
    original = module.weight.detach().clone()
    projected = torch.rand(1, 2, 3, 5)

    outputs = {
        mode: reconstruct_branch0(projected, module, mode)
        for mode in ("baseline", "phase_tied", "bilinear")
    }

    assert all(output.shape == (1, 2, 12, 20) for output in outputs.values())
    assert torch.equal(module.weight.detach(), original)


def test_local_dpt_capture_combines_chunks_and_selects_one_frame() -> None:
    class FakeDPT(torch.nn.Module):
        def forward(self, tokens):
            return tokens[0]

    dpt = FakeDPT()
    capture = LocalDPTInputCapture(dpt, frame_count=3, frame_index=1)
    capture.register()
    first = [torch.tensor([[[1.0]], [[2.0]]]), torch.tensor([[[10.0]], [[20.0]]])]
    second = [torch.tensor([[[3.0]]]), torch.tensor([[[30.0]]])]
    dpt(first)
    dpt(second)
    capture.remove()

    selected = capture.selected_tokens()

    assert selected[0].item() == 2.0
    assert selected[1].item() == 20.0


def test_relative_reduction_handles_cv_and_ratio_excess() -> None:
    assert np.isclose(relative_reduction(0.1, 0.2), 0.5)
    assert np.isclose(relative_reduction(1.2, 1.4, excess_ratio=True), 0.5)


def test_lowpass_correlation_preserves_scaled_large_structure() -> None:
    value = np.arange(256, dtype=np.float32).reshape(16, 16)
    assert np.isclose(lowpass_correlation(value, value * 3.0), 1.0)


def _metric_row(reduction: float):
    return {
        "reduction_excess_ratio_x": reduction,
        "reduction_excess_ratio_y": reduction,
        "reduction_phase_cv_x": reduction,
        "reduction_phase_cv_y": reduction,
    }


def test_ablation_diagnosis_identifies_phase_mapping_case_e() -> None:
    rows = {}
    for mode in ("phase_tied", "bilinear"):
        rows[(mode, "branch0_resized")] = _metric_row(0.8)
        rows[(mode, "path1")] = _metric_row(0.7)
        rows[(mode, "depth")] = _metric_row(0.7)
    correlations = {
        "phase_tied": {"mean_correlation": 0.95},
        "bilinear": {"mean_correlation": 0.80},
    }

    diagnosis = diagnose_ablation(
        rows, correlations, ("baseline", "phase_tied", "bilinear"), 0.5
    )

    assert diagnosis["case"] == "E"
    assert "Independent subpixel phase mappings" in diagnosis["diagnosis"]


def test_requested_modes_keep_baseline_for_variant_comparison() -> None:
    assert _requested_modes("baseline") == ("baseline",)
    assert _requested_modes("phase_tied") == ("baseline", "phase_tied")
    assert _requested_modes("bilinear") == ("baseline", "bilinear")
    assert _requested_modes("all") == ("baseline", "phase_tied", "bilinear")
