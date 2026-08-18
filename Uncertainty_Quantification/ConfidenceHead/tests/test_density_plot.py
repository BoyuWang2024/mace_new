"""TDD tests for the continuous ConfidenceHead density primitives."""
from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

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
        atom_indices=[0, 1, 2, 3, 4, 5],
        unit="eV/A",
        task="force",
        order=None,
    )
    assert result.expected.tolist() == pytest.approx([0.1, 0.5])
    assert result.actual.tolist() == pytest.approx([0.2, 0.6])
    assert result.sample_ids == ["s0", "s5"]
    assert result.atom_indices == [0, 5]
    assert result.unit == "eV/A"
    assert result.task == "force"
    assert result.order is None
    assert result.audit == {
        "total_count": 6,
        "valid_count": 2,
        "excluded_nonfinite": 3,
        "excluded_nonpositive": 1,
    }


def test_build_error_pairs_energy_uses_none_atom_indices_and_order_metadata() -> None:
    result = build_error_pairs(
        expected=torch.tensor([0.1, 0.2]),
        actual=torch.tensor([0.05, 0.08]),
        sample_ids=["a", "b"],
        atom_indices=None,
        unit="eV/atom",
        task="energy",
        order=8,
    )
    assert result.atom_indices is None
    assert result.unit == "eV/atom"
    assert result.task == "energy"
    assert result.order == 8


def test_build_error_pairs_rejects_mismatched_lengths_and_invalid_inputs() -> None:
    with pytest.raises(DensityPlotError, match="length"):
        build_error_pairs(
            expected=torch.ones(2),
            actual=torch.ones(1),
            sample_ids=["a", "b"],
            atom_indices=None,
            unit="eV",
            task="energy",
            order=None,
        )
    with pytest.raises(DensityPlotError, match="length"):
        build_error_pairs(
            expected=torch.ones(2),
            actual=torch.ones(2),
            sample_ids=["a"],
            atom_indices=None,
            unit="eV",
            task="energy",
            order=None,
        )
    with pytest.raises(DensityPlotError, match="length"):
        build_error_pairs(
            expected=torch.ones(2),
            actual=torch.ones(2),
            sample_ids=["a", "b"],
            atom_indices=[0],
            unit="eV",
            task="force",
            order=None,
        )
    invalid_actual = (
        torch.tensor([0.0]),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([-1.0]),
    )
    for actual in invalid_actual:
        with pytest.raises(DensityPlotError, match="positive|finite"):
            build_error_pairs(
                expected=torch.tensor([0.1]),
                actual=actual,
                sample_ids=["a"],
                atom_indices=None,
                unit="eV",
                task="energy",
                order=None,
            )


def test_compute_log_metrics_returns_dataclass_on_all_points() -> None:
    expected = torch.logspace(-3, 1, 100)
    actual = expected * torch.exp(0.7 * torch.sin(torch.arange(100, dtype=torch.float32) * 0.31))
    result = compute_log_metrics(expected, actual)
    full_log_expected = torch.log10(expected).numpy()
    full_log_actual = torch.log10(actual).numpy()
    full_log_spearman = spearmanr(full_log_expected, full_log_actual).statistic
    full_log_pearson = np.corrcoef(full_log_expected, full_log_actual)[0, 1]
    raw_pearson = np.corrcoef(expected.numpy(), actual.numpy())[0, 1]
    assert is_dataclass(result)
    assert isinstance(result, DensityMetrics)
    assert result.valid_count == 100
    assert result.spearman_rho == pytest.approx(full_log_spearman)
    assert result.log10_pearson_r == pytest.approx(full_log_pearson)
    assert result.log10_pearson_r != pytest.approx(raw_pearson)


def test_metrics_are_computed_on_full_data_not_scatter_sample() -> None:
    expected = torch.logspace(-3, 1, 100)
    actual = expected * torch.exp(0.7 * torch.sin(torch.arange(100, dtype=torch.float32) * 0.31))
    indices = sample_scatter_indices(100, max_points=20, seed=20260714)
    assert len(indices) == 20
    metrics = compute_log_metrics(expected, actual)
    full_log_pearson = np.corrcoef(
        torch.log10(expected).numpy(), torch.log10(actual).numpy()
    )[0, 1]
    sampled_log_pearson = np.corrcoef(
        torch.log10(expected[indices]).numpy(), torch.log10(actual[indices]).numpy()
    )[0, 1]
    assert metrics.valid_count == 100
    assert metrics.log10_pearson_r == pytest.approx(full_log_pearson)
    assert metrics.log10_pearson_r != pytest.approx(sampled_log_pearson)
    assert np.array_equal(indices, sample_scatter_indices(100, 20, 20260714))
    assert not np.array_equal(indices, sample_scatter_indices(100, 20, 20260715))


def test_compute_log_metrics_rejects_nonpositive_nonfinite_and_too_few_points() -> None:
    invalid_expected = (
        torch.tensor([0.0, 1.0]),
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([float("inf"), 1.0]),
    )
    for values in invalid_expected:
        with pytest.raises(DensityPlotError, match="positive|finite"):
            compute_log_metrics(values, torch.tensor([1.0, 2.0]))
    invalid_actual = (
        torch.tensor([0.0, 1.0]),
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([float("inf"), 1.0]),
        torch.tensor([-1.0, 1.0]),
    )
    for values in invalid_actual:
        with pytest.raises(DensityPlotError, match="positive|finite"):
            compute_log_metrics(torch.tensor([1.0, 2.0]), values)
    with pytest.raises(DensityPlotError, match="two"):
        compute_log_metrics(torch.tensor([1.0]), torch.tensor([2.0]))


def test_sample_scatter_indices_is_bounded_and_deterministic() -> None:
    first = sample_scatter_indices(100, max_points=20, seed=20260714)
    assert len(first) == 20
    assert len(np.unique(first)) == 20
    assert np.all((first >= 0) & (first < 100))
    assert np.array_equal(first, sample_scatter_indices(100, 20, 20260714))
    assert np.array_equal(sample_scatter_indices(10, 20, 1), np.arange(10))


def test_compute_density_grid_has_160_grid_normalized_positive_ascending_thresholds() -> None:
    x = torch.logspace(-3, 1, 500)
    y = x * 2
    result = compute_density_grid(x, y, grid_size=160, sigma=1.2)
    x_log = torch.log10(x).numpy()
    y_log = torch.log10(y).numpy()
    x_margin = 0.05 * (x_log.max() - x_log.min())
    y_margin = 0.05 * (y_log.max() - y_log.min())
    assert result.density.shape == (160, 160)
    assert result.x_centers.shape == (160,)
    assert result.y_centers.shape == (160,)
    assert np.isfinite(result.x_centers).all()
    assert np.isfinite(result.y_centers).all()
    assert np.all(np.diff(result.x_centers) > 0)
    assert np.all(np.diff(result.y_centers) > 0)
    assert x_log.min() - x_margin <= result.x_centers[0] <= x_log.min()
    assert x_log.max() <= result.x_centers[-1] <= x_log.max() + x_margin
    assert y_log.min() - y_margin <= result.y_centers[0] <= y_log.min()
    assert y_log.max() <= result.y_centers[-1] <= y_log.max() + y_margin
    assert result.density.sum() == pytest.approx(1.0)
    assert np.isfinite(result.density).all()
    assert (result.density >= 0).all()
    assert len(result.contour_levels) == 5
    assert all(level > 0 for level in result.contour_levels)
    assert all(a <= b for a, b in zip(result.contour_levels, result.contour_levels[1:]))


@pytest.mark.parametrize(
    "x,y,pattern",
    [
        (torch.tensor([]), torch.tensor([]), "two"),
        (torch.tensor([1.0]), torch.tensor([2.0]), "two"),
        (torch.ones(3), torch.tensor([1.0, 2.0, 3.0]), "constant"),
        (torch.tensor([1.0, 2.0, 3.0]), torch.ones(3), "constant"),
        (torch.tensor([1.0, 2.0]), torch.tensor([1.0]), "length"),
        (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0]), "positive"),
        (torch.tensor([float("nan"), 1.0]), torch.tensor([1.0, 2.0]), "finite"),
        (torch.tensor([1.0, 2.0]), torch.tensor([0.0, 1.0]), "positive"),
        (torch.tensor([1.0, 2.0]), torch.tensor([float("nan"), 1.0]), "finite"),
        (torch.tensor([1.0, 2.0]), torch.tensor([float("inf"), 1.0]), "finite"),
    ],
)
def test_compute_density_grid_rejects_invalid_inputs(x, y, pattern):
    with pytest.raises(DensityPlotError, match=pattern):
        compute_density_grid(x, y)
