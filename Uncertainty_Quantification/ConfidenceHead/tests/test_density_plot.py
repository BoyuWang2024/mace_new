"""TDD tests for the continuous ConfidenceHead density primitives."""
from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pytest
import torch

from confidence_head.density_plot import (
    DensityMetrics,
    DensityPlotError,
    build_error_pairs,
    compute_density_grid,
    compute_log_metrics,
    sample_scatter_indices,
)


def test_build_error_pairs_force_filters_pairwise_and_preserves_metadata() -> None:
    result = build_error_pairs(
        expected=torch.tensor([0.1, float("nan"), 0.3, 0.0, float("inf"), 0.5]),
        actual=torch.tensor([0.2, 0.2, float("inf"), 0.4, 0.5, 0.6]),
        sample_ids=["s0", "s1", "s2", "s3", "s4", "s5"],
        atom_indices=[0, 1, 2, 3, 4, 5], unit="eV/A", task="force", order=None,
    )
    assert result.expected.tolist() == pytest.approx([0.1, 0.5])
    assert result.actual.tolist() == pytest.approx([0.2, 0.6])
    assert result.sample_ids == ["s0", "s5"]
    assert result.atom_indices == [0, 5]
    assert result.unit == "eV/A" and result.task == "force" and result.order is None
    assert result.audit == {"total_count": 6, "valid_count": 2, "excluded_nonfinite": 3, "excluded_nonpositive": 1}


def test_build_error_pairs_energy_uses_none_atom_indices_and_order_metadata() -> None:
    result = build_error_pairs(
        expected=torch.tensor([0.1, 0.2]), actual=torch.tensor([0.05, 0.08]),
        sample_ids=["a", "b"], atom_indices=None, unit="eV/atom", task="energy", order=8,
    )
    assert result.atom_indices is None
    assert result.unit == "eV/atom" and result.task == "energy" and result.order == 8


def test_build_error_pairs_rejects_mismatched_lengths_and_invalid_inputs() -> None:
    with pytest.raises(DensityPlotError, match="length"):
        build_error_pairs(expected=torch.ones(2), actual=torch.ones(1), sample_ids=["a", "b"], atom_indices=None, unit="eV", task="energy", order=None)
    with pytest.raises(DensityPlotError, match="positive|finite"):
        build_error_pairs(expected=torch.tensor([0.1]), actual=torch.tensor([0.0]), sample_ids=["a"], atom_indices=None, unit="eV", task="energy", order=None)


def test_compute_log_metrics_returns_dataclass_on_all_points() -> None:
    result = compute_log_metrics(torch.logspace(-3, 1, 100), torch.logspace(-2, 2, 100))
    assert is_dataclass(result) and isinstance(result, DensityMetrics)
    assert result.valid_count == 100
    assert result.spearman_rho == pytest.approx(1.0)
    assert result.log10_pearson_r == pytest.approx(1.0)


def test_metrics_are_computed_on_full_data_not_scatter_sample() -> None:
    expected = torch.logspace(-3, 1, 100)
    actual = torch.logspace(-2, 2, 100)
    indices = sample_scatter_indices(100, max_points=20, seed=20260714)
    assert len(indices) == 20
    metrics = compute_log_metrics(expected, actual)
    assert metrics.valid_count == 100
    assert np.array_equal(indices, sample_scatter_indices(100, 20, 20260714))
    assert not np.array_equal(indices, sample_scatter_indices(100, 20, 20260715))


def test_compute_log_metrics_rejects_nonpositive_nonfinite_and_too_few_points() -> None:
    for values in (torch.tensor([0.0, 1.0]), torch.tensor([float("nan"), 1.0]), torch.tensor([float("inf"), 1.0])):
        with pytest.raises(DensityPlotError, match="positive|finite"):
            compute_log_metrics(values, torch.tensor([1.0, 2.0]))
    with pytest.raises(DensityPlotError, match="two"):
        compute_log_metrics(torch.tensor([1.0]), torch.tensor([2.0]))


def test_sample_scatter_indices_is_bounded_and_deterministic() -> None:
    first = sample_scatter_indices(100, max_points=20, seed=20260714)
    assert len(first) == 20 and len(np.unique(first)) == 20
    assert np.all((first >= 0) & (first < 100))
    assert np.array_equal(first, sample_scatter_indices(100, 20, 20260714))
    assert np.array_equal(sample_scatter_indices(10, 20, 1), np.arange(10))


def test_compute_density_grid_has_160_grid_normalized_positive_ascending_thresholds() -> None:
    x = torch.logspace(-3, 1, 500)
    result = compute_density_grid(x, x * 2, grid_size=160, sigma=1.2)
    assert result.density.shape == (160, 160)
    assert result.x_centers.shape == (160,)
    assert result.y_centers.shape == (160,)
    assert np.isfinite(result.x_centers).all()
    assert np.isfinite(result.y_centers).all()
    assert result.density.sum() == pytest.approx(1.0)
    assert np.isfinite(result.density).all() and (result.density >= 0).all()
    assert len(result.contour_levels) == 5
    assert all(level > 0 for level in result.contour_levels)
    assert all(a <= b for a, b in zip(result.contour_levels, result.contour_levels[1:]))


@pytest.mark.parametrize("x,y,pattern", [
    (torch.tensor([]), torch.tensor([]), "two"),
    (torch.tensor([1.0]), torch.tensor([2.0]), "two"),
    (torch.ones(3), torch.tensor([1.0, 2.0, 3.0]), "constant"),
    (torch.tensor([1.0, 2.0, 3.0]), torch.ones(3), "constant"),
    (torch.tensor([1.0, 2.0]), torch.tensor([1.0]), "length"),
    (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0]), "positive"),
    (torch.tensor([float("nan"), 1.0]), torch.tensor([1.0, 2.0]), "finite"),
])
def test_compute_density_grid_rejects_invalid_inputs(x, y, pattern):
    with pytest.raises(DensityPlotError, match=pattern):
        compute_density_grid(x, y)
