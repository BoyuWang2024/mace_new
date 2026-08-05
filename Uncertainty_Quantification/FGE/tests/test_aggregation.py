from __future__ import annotations

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.aggregation import (
    normalized_positive_weights,
    validation_error_weights,
    weighted_mean,
)
from Uncertainty_Quantification.FGE.fge.errors import HardFailure


def test_validation_error_weights_use_base_scaled_epsilon() -> None:
    rmse = torch.tensor([1.0, 3.0], dtype=torch.float64)
    weights = validation_error_weights(rmse, base_rmse=2.0, eps_ratio=0.5)
    assert torch.allclose(weights, torch.tensor([2.0 / 3.0, 1.0 / 3.0], dtype=torch.float64))


def test_weighted_mean_broadcasts_only_over_member_axis() -> None:
    members = torch.tensor(
        [[[0.0, 2.0]], [[4.0, 6.0]]], dtype=torch.float64
    )
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    assert torch.equal(
        weighted_mean(members, weights),
        torch.tensor([[3.0, 5.0]], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("weights", "size"),
    [
        (torch.tensor([1.0], dtype=torch.float64), 2),
        (torch.tensor([0.0, 1.0], dtype=torch.float64), 2),
        (torch.tensor([float("nan"), 1.0], dtype=torch.float64), 2),
        (torch.tensor([1, 2]), 2),
    ],
)
def test_normalized_weights_reject_invalid_inputs(
    weights: torch.Tensor, size: int
) -> None:
    with pytest.raises(HardFailure):
        normalized_positive_weights(weights, size)


def test_validation_error_weights_reject_nonpositive_epsilon() -> None:
    with pytest.raises(HardFailure):
        validation_error_weights(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            base_rmse=1.0,
            eps_ratio=0.0,
        )


def test_weighted_mean_rejects_nonfloating_and_nonfinite_members() -> None:
    with pytest.raises(HardFailure):
        weighted_mean(torch.tensor([[1], [2]]), None)
    with pytest.raises(HardFailure):
        weighted_mean(
            torch.tensor([[1.0], [float("inf")]], dtype=torch.float64), None
        )
