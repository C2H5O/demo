import numpy as np
import torch

from diagnostics.trace_distill3r_artifacts_stage2 import (
    BranchStage2Trace,
    deconv_kernel_energy,
    diagnose,
    gradient_statistics,
    mean_phase_template,
    phase_statistics,
    projection_comparison,
)


def test_mean_phase_template_recovers_repeated_4x4_pattern() -> None:
    template = np.arange(16, dtype=np.float32).reshape(4, 4)
    repeated = np.tile(template, (3, 5))

    recovered = mean_phase_template(repeated, 4)

    assert np.allclose(recovered, template)


def test_phase_statistics_detects_mod4_gradient_imbalance() -> None:
    template = np.asarray(
        [[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]], dtype=np.float32
    )
    spatial_map = np.tile(template, (8, 10))

    result = phase_statistics(spatial_map, 4)

    assert result["period_x"] == 4
    assert result["phase_ratio_x"] > 1e6
    assert result["phase_cv_x"] > 0.5


def test_gradient_statistics_are_scale_invariant_after_normalization() -> None:
    value = np.arange(20, dtype=np.float32).reshape(4, 5) + 1.0
    first = gradient_statistics(value)
    second = gradient_statistics(value * 7.0)

    assert np.isclose(first["spatial_cv"], second["spatial_cv"])
    assert np.isclose(
        first["normalized_neighbor_difference"],
        second["normalized_neighbor_difference"],
    )


def test_deconv_kernel_energy_preserves_spatial_kernel_shape() -> None:
    module = torch.nn.ConvTranspose2d(3, 5, kernel_size=4, stride=4)
    energy = deconv_kernel_energy(module)

    assert energy.shape == (4, 4)
    assert np.all(energy >= 0.0)


def test_projection_comparison_uses_spatial_maps_not_channel_subtraction() -> None:
    input_map = np.arange(20, dtype=np.float32).reshape(4, 5) + 1.0
    projected_map = input_map * 3.0
    input_row = gradient_statistics(input_map)
    projected_row = gradient_statistics(projected_map)

    comparison = projection_comparison(
        input_map, projected_map, input_row, projected_row
    )

    assert np.isclose(comparison["norm_map_pearson_correlation"], 1.0)
    assert np.isclose(comparison["spatial_cv_ratio"], 1.0)
    assert np.isclose(comparison["normalized_neighbor_difference_ratio"], 1.0)


def _row(phase_ratio: float, phase_cv: float, spatial_cv: float = 1.0, neighbor: float = 1.0):
    return {
        "phase_ratio_x": phase_ratio,
        "phase_ratio_y": phase_ratio,
        "phase_cv_x": phase_cv,
        "phase_cv_y": phase_cv,
        "spatial_cv": spatial_cv,
        "normalized_neighbor_difference": neighbor,
    }


def test_diagnosis_distinguishes_x4_from_generic_deconvolution() -> None:
    rows = {
        "branch0_input": _row(np.nan, np.nan),
        "branch0_projected": _row(np.nan, np.nan),
        "branch0_resized": _row(2.0, 0.3),
        "branch1_resized": _row(1.05, 0.02),
    }
    x4 = diagnose(rows, 1.25, 0.1, 1.5)
    assert x4["case"] == 1
    assert "ConvTranspose4" in x4["diagnosis"]

    rows["branch1_resized"] = _row(1.8, 0.2)
    generic = diagnose(rows, 1.25, 0.1, 1.5)
    assert generic["case"] == 3
    assert "generic" in generic["diagnosis"]


def test_stage2_hooks_split_projection_from_resize_in_one_execution() -> None:
    dpt = torch.nn.Module()
    dpt.act_postprocess = torch.nn.ModuleList(
        [
            torch.nn.Sequential(
                torch.nn.Conv2d(384, 96, kernel_size=1),
                torch.nn.ConvTranspose2d(96, 96, kernel_size=4, stride=4),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(384, 192, kernel_size=1),
                torch.nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2),
            ),
            torch.nn.Sequential(torch.nn.Conv2d(384, 384, kernel_size=1)),
        ]
    )
    model = torch.nn.Module()
    model.student = torch.nn.Module()
    model.student.downstream_head_local = torch.nn.Module()
    model.student.downstream_head_local.dpt = dpt
    trace = BranchStage2Trace(model, frame_index=1, frame_count=2)
    trace.register()
    source = torch.rand(2, 384, 2, 3)
    for branch in dpt.act_postprocess:
        branch(source)
    trace.remove()

    features, consistency = trace.selected()

    assert features["branch0_input"].shape == (384, 2, 3)
    assert features["branch0_projected"].shape == (96, 2, 3)
    assert features["branch0_resized"].shape == (96, 8, 12)
    assert features["branch1_resized"].shape == (192, 4, 6)
    assert features["branch2_resized"].shape == (384, 2, 3)
    assert all(value == 0.0 for value in consistency.values())
