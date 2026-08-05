from __future__ import annotations

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.uncertainty import (
    reduce_atoms_by_structure,
    scalar_gmd,
    unbiased_std,
    vector_gmd,
    vector_std,
)


def test_equal_weight_std_uses_k_minus_one() -> None:
    members = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    assert torch.equal(unbiased_std(members), torch.tensor([1.0]))


def test_weighted_std_uses_one_minus_sum_squared_weights() -> None:
    members = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    expected = torch.tensor([2.0**0.5], dtype=torch.float64)
    assert torch.allclose(unbiased_std(members, weights), expected)


def test_vector_std_uses_l2_scatter() -> None:
    members = torch.tensor(
        [[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=torch.float64
    )
    assert torch.allclose(vector_std(members), torch.tensor([2.0**0.5], dtype=torch.float64))


def test_equal_and_weighted_scalar_gmd_use_unordered_pairs() -> None:
    members = torch.tensor([[0.0], [2.0], [5.0]], dtype=torch.float64)
    assert torch.allclose(scalar_gmd(members), torch.tensor([10.0 / 3.0], dtype=torch.float64))

    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    expected = (0.2 * 0.3 * 2.0 + 0.2 * 0.5 * 5.0 + 0.3 * 0.5 * 3.0) / (
        0.2 * 0.3 + 0.2 * 0.5 + 0.3 * 0.5
    )
    assert torch.allclose(
        scalar_gmd(members, weights), torch.tensor([expected], dtype=torch.float64)
    )


def test_vector_gmd_uses_pairwise_l2_distance() -> None:
    members = torch.tensor(
        [[[0.0, 0.0, 0.0]], [[3.0, 4.0, 0.0]]], dtype=torch.float64
    )
    assert torch.equal(vector_gmd(members), torch.tensor([5.0]))


def test_reduce_atoms_by_structure_reports_mean_max_and_linear_q95() -> None:
    values = torch.tensor([1.0, 3.0, 2.0, 6.0], dtype=torch.float64)
    atom_to_structure = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    reduced = reduce_atoms_by_structure(values, atom_to_structure, 2)
    assert torch.equal(reduced["mean"], torch.tensor([2.0, 4.0]))
    assert torch.equal(reduced["max"], torch.tensor([3.0, 6.0]))
    assert torch.allclose(reduced["q95"], torch.tensor([2.9, 5.8], dtype=torch.float64))


@pytest.mark.parametrize(
    "members",
    [
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.empty((2, 0), dtype=torch.float64),
        torch.tensor([[1], [2]]),
        torch.tensor([[1.0], [float("nan")]], dtype=torch.float64),
    ],
)
def test_uncertainty_rejects_invalid_members(members: torch.Tensor) -> None:
    with pytest.raises(HardFailure):
        unbiased_std(members)


def test_structure_reduction_rejects_empty_or_misaligned_partitions() -> None:
    values = torch.tensor([1.0, 2.0], dtype=torch.float64)
    with pytest.raises(HardFailure):
        reduce_atoms_by_structure(values, torch.tensor([0]), 1)
    with pytest.raises(HardFailure):
        reduce_atoms_by_structure(values, torch.tensor([0, 0]), 2)
    with pytest.raises(HardFailure):
        reduce_atoms_by_structure(values, torch.tensor([0, 2]), 2)
