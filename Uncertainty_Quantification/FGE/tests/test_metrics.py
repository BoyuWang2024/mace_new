from __future__ import annotations

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.metrics import (
    compute_correlations,
    compute_errors,
    compute_risk_coverage,
)


def test_constant_input_correlation_is_null_warning() -> None:
    result = compute_correlations(
        torch.ones(3, dtype=torch.float64),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
    )
    assert result["pearson"] is None
    assert result["spearman"] is None
    assert result["warnings"] == ["correlation undefined for constant input"]


def test_spearman_uses_average_ranks_for_ties() -> None:
    result = compute_correlations(
        torch.tensor([1.0, 1.0, 3.0, 4.0], dtype=torch.float64),
        torch.tensor([1.0, 2.0, 2.0, 4.0], dtype=torch.float64),
    )
    assert result["pearson"] is not None
    assert result["spearman"] == pytest.approx(5.0 / 6.0)


def test_risk_coverage_rejects_highest_uncertainty_first_stably() -> None:
    rows = compute_risk_coverage(
        uncertainty=torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float64),
        error=torch.tensor([1.0, 9.0, 2.0, 8.0], dtype=torch.float64),
        coverages=(1.0, 0.5),
    )
    assert rows == [
        {"coverage": 1.0, "risk": 5.0, "count": 4},
        {"coverage": 0.5, "risk": 1.5, "count": 2},
    ]

    tied = compute_risk_coverage(
        uncertainty=torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64),
        error=torch.tensor([10.0, 20.0, 1.0], dtype=torch.float64),
        coverages=(2.0 / 3.0,),
    )
    assert tied[0]["risk"] == pytest.approx(5.5)


def test_compute_errors_preserves_energy_force_granularities() -> None:
    result = compute_errors(
        energy_prediction=torch.tensor([4.0, 8.0], dtype=torch.float64),
        forces_prediction=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]],
            dtype=torch.float64,
        ),
        energy_reference=torch.tensor([2.0, 10.0], dtype=torch.float64),
        forces_reference=torch.zeros((3, 3), dtype=torch.float64),
        n_atoms=torch.tensor([2, 1], dtype=torch.int64),
        atom_to_structure=torch.tensor([0, 0, 1], dtype=torch.int64),
    )
    assert torch.equal(result["energy_total"], torch.tensor([2.0, 2.0]))
    assert torch.equal(result["energy_per_atom"], torch.tensor([1.0, 2.0]))
    assert torch.equal(result["force_atom_vector"], torch.tensor([1.0, 2.0, 4.0]))
    assert torch.equal(result["force_structure_mean"], torch.tensor([1.5, 4.0]))
    assert torch.equal(result["force_structure_max"], torch.tensor([2.0, 4.0]))


def test_metrics_reject_nonfinite_or_misaligned_inputs() -> None:
    with pytest.raises(HardFailure):
        compute_correlations(
            torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 2.0])
        )
    with pytest.raises(HardFailure):
        compute_risk_coverage(
            torch.tensor([1.0]), torch.tensor([1.0, 2.0]), (1.0,)
        )
