"""Canonical projections of runtime metric dictionaries."""

from __future__ import annotations

from collections.abc import Mapping


def rmse_only(metrics: Mapping[str, float]) -> dict[str, float]:
    """Keep only the two RMSE values accepted by the training manifest."""
    return {
        "energy_rmse": float(metrics["energy_rmse"]),
        "forces_rmse": float(metrics["forces_rmse"]),
    }
