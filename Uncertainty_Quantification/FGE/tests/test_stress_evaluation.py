from __future__ import annotations

import torch

from Uncertainty_Quantification.FGE.fge.evaluation import _stress_outputs
from Uncertainty_Quantification.FGE.fge.uncertainty import unbiased_std


def _prediction_with_stress() -> dict[str, torch.Tensor]:
    return {
        "stress_members": torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                    [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
                ],
                [
                    [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
                    [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                ],
            ],
            dtype=torch.float64,
        ),
        "stress_reference": torch.zeros((2, 3, 3), dtype=torch.float64),
    }


def test_stress_equal_weight_uses_k_minus_one() -> None:
    members = torch.tensor([[[[1.0]]], [[[3.0]]]], dtype=torch.float64)

    assert torch.allclose(
        unbiased_std(members), torch.tensor([[[2.0**0.5]]], dtype=torch.float64)
    )


def test_stress_validation_weighted_uses_force_weights() -> None:
    force_weights = torch.tensor([0.25, 0.75], dtype=torch.float64)

    ensemble, uncertainty, errors = _stress_outputs(
        _prediction_with_stress(), force_weights
    )

    assert torch.equal(ensemble["stress_weights"], force_weights)
    assert uncertainty["stress_component_std"].shape == (2, 3, 3)
    assert errors["stress_component"].shape == (2, 3, 3)
    assert torch.equal(
        ensemble["stress"],
        torch.tensor(
            [
                [[2.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 4.5]],
                [[3.5, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 5.5]],
            ],
            dtype=torch.float64,
        ),
    )
