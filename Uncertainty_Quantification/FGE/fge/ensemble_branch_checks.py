"""Validation of both persisted weighting branches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

from .ensemble_checks import assert_ensemble_weights


def assert_stored_weights(
    ensemble: Mapping[str, Any],
    energy_weights: Tensor | None,
    force_weights: Tensor | None,
    member_count: int,
    branch: str,
) -> None:
    assert_ensemble_weights(
        ensemble.get("energy_weights"), energy_weights, member_count, f"{branch}.energy_weights"
    )
    assert_ensemble_weights(
        ensemble.get("force_weights"), force_weights, member_count, f"{branch}.force_weights"
    )
