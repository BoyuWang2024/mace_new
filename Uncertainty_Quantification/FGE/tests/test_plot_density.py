from __future__ import annotations

import torch

from Uncertainty_Quantification.FGE.fge.plot_density import (
    analyze_log_panel,
    deterministic_indices,
)
from Uncertainty_Quantification.FGE.fge.plot_style import PlotConfig


def test_log_analysis_records_excluded_values() -> None:
    result = analyze_log_panel(
        torch.tensor([1.0, 0.0, float("inf"), 2.0, -1.0, float("nan")]),
        torch.tensor([1.0, 3.0, 4.0, 2.0, 5.0, 6.0]),
        PlotConfig(grid_size=16),
    )

    assert result.valid_count == 2
    assert result.excluded == {"zero": 1, "negative": 1, "nan": 1, "inf": 1}
    assert result.density.shape == (16, 16)


def test_sampling_is_stable_unique_and_bounded() -> None:
    first = deterministic_indices(100_000, 20_000, 20260714)
    second = deterministic_indices(100_000, 20_000, 20260714)

    assert first.numel() == 20_000
    assert torch.equal(first, second)
    assert torch.unique(first).numel() == 20_000
    assert int(first.min()) >= 0
    assert int(first.max()) < 100_000


def test_sampling_keeps_all_points_when_under_limit() -> None:
    assert torch.equal(deterministic_indices(4, 20_000, 20260714), torch.arange(4))
