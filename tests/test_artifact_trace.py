import numpy as np
import torch

from diagnostics.trace_distill3r_artifacts import (
    _feature_maps,
    _require_import_below,
    boundary_metrics,
    modulo_gradient_profile,
)


def test_feature_maps_preserve_native_resolution_and_avoid_sign_cancellation() -> None:
    feature = torch.tensor(
        [
            [[1.0, -2.0], [3.0, -4.0]],
            [[-1.0, 2.0], [-3.0, 4.0]],
        ]
    )
    maps = _feature_maps(feature)

    assert maps["mean"].shape == (2, 2)
    assert np.allclose(maps["mean"], 0.0)
    assert np.all(maps["absmean"] > 0.0)
    assert np.all(maps["norm"] > 0.0)


def test_boundary_metric_detects_token_cell_discontinuities() -> None:
    # A 4x4 token grid expanded into 2x2 constant cells.
    token_values = np.arange(16, dtype=np.float32).reshape(4, 4)
    spatial_map = np.repeat(np.repeat(token_values, 2, axis=0), 2, axis=1)

    metrics = boundary_metrics(spatial_map, token_grid=(4, 4))

    assert metrics["cell_h"] == 2
    assert metrics["cell_w"] == 2
    assert metrics["boundary_ratio_x"] > 1e6
    assert metrics["boundary_ratio_y"] > 1e6


def test_native_token_grid_has_no_interior_gradient_baseline() -> None:
    metrics = boundary_metrics(np.ones((32, 40), dtype=np.float32))

    assert metrics["cell_h"] == 1
    assert metrics["cell_w"] == 1
    assert np.isnan(metrics["boundary_ratio_x"])
    assert np.isnan(metrics["boundary_ratio_y"])


def test_modulo_profile_places_cell_boundary_at_phase_zero() -> None:
    token_values = np.arange(16, dtype=np.float32).reshape(4, 4)
    spatial_map = np.repeat(np.repeat(token_values, 2, axis=0), 2, axis=1)

    profile = modulo_gradient_profile(spatial_map, cell_h=2, cell_w=2)

    assert profile["modulo_x"] == 2
    assert profile["modulo_y"] == 2
    assert profile["mod_x"]["phase0"] > profile["mod_x"]["phase1"]
    assert profile["mod_y"]["phase0"] > profile["mod_y"]["phase1"]


def test_import_guard_accepts_only_the_pinned_source_tree(tmp_path) -> None:
    pinned = tmp_path / "external" / "Distill3R"
    module = pinned / "distill3r" / "student" / "model.py"
    _require_import_below(module, pinned, "student")

    try:
        _require_import_below(tmp_path / "other" / "model.py", pinned, "student")
    except RuntimeError as error:
        assert "outside the pinned source" in str(error)
    else:
        raise AssertionError("Expected an import outside the pinned source to fail")
