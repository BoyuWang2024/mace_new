"""Tests for publication evaluation metrics and tie-aware correlations."""

from __future__ import annotations

import math

import pytest
import torch

from confidence_head.metrics import (
    branch_metrics,
    expected_errors,
    pearson_correlation,
    spearman_correlation,
    tie_aware_ranks,
)


def test_expected_errors_uses_probability_weighted_representatives() -> None:
    logits = torch.tensor([[0.0, 0.0], [math.log(3.0), 0.0]])
    representatives = torch.tensor([1.0, 5.0])

    result = expected_errors(logits, representatives)

    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    assert result.tolist() == pytest.approx([3.0, 2.0])


def test_branch_metrics_match_hand_derived_values() -> None:
    logits = torch.tensor([[0.0, 0.0], [math.log(3.0), 0.0]])
    labels = torch.tensor([1, 0])
    errors = torch.tensor([4.0, 1.0])
    representatives = torch.tensor([1.0, 5.0])

    result = branch_metrics(logits, labels, errors, representatives)

    assert result == {
        "sample_count": 2,
        "accuracy": pytest.approx(0.5),
        "brier": pytest.approx(0.3125),
        "mean_expected_error": pytest.approx(2.5),
        "mean_observed_error": pytest.approx(2.5),
        "mae_expected_vs_error": pytest.approx(1.0),
        "spearman_expected_vs_error": pytest.approx(1.0),
    }


def test_tie_aware_ranks_use_stable_average_one_based_ranks() -> None:
    values = torch.tensor([4.0, 1.0, 1.0, 3.0], dtype=torch.float32)

    result = tie_aware_ranks(values)

    assert result.dtype == torch.float64
    assert result.tolist() == [4.0, 1.5, 1.5, 3.0]


def test_tie_aware_spearman_is_pearson_of_average_ranks() -> None:
    left = torch.tensor([1.0, 1.0, 3.0, 4.0])
    right = torch.tensor([1.0, 2.0, 2.0, 4.0])

    assert spearman_correlation(left, right) == pytest.approx(5.0 / 6.0)


def test_pearson_correlation_matches_hand_derived_negative_relation() -> None:
    left = torch.tensor([1.0, 2.0, 3.0])
    right = torch.tensor([6.0, 4.0, 2.0])

    assert pearson_correlation(left, right) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: expected_errors(
                torch.tensor([[0.0, float("nan")]]), torch.tensor([1.0, 2.0])
            ),
            "finite",
        ),
        (
            lambda: expected_errors(
                torch.tensor([[0.0, 1.0]]), torch.tensor([1.0])
            ),
            "bins",
        ),
        (
            lambda: branch_metrics(
                torch.tensor([[0.0, 1.0]]),
                torch.tensor([2]),
                torch.tensor([1.0]),
                torch.tensor([1.0, 2.0]),
            ),
            "labels",
        ),
        (
            lambda: branch_metrics(
                torch.empty((0, 2)),
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,)),
                torch.tensor([1.0, 2.0]),
            ),
            "non-empty",
        ),
        (
            lambda: pearson_correlation(
                torch.tensor([1.0, 1.0]), torch.tensor([1.0, 2.0])
            ),
            "variance",
        ),
    ],
)
def test_metrics_reject_malformed_or_undefined_inputs(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
