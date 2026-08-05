"""Deterministic error, correlation, and risk-coverage metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from .errors import HardFailure
from .uncertainty import reduce_atoms_by_structure


def _finite_vector(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise HardFailure(f"{name} must be a floating torch.Tensor")
    if value.ndim != 1 or value.numel() == 0:
        raise HardFailure(f"{name} must be a nonempty vector")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(result).all().item():
        raise HardFailure(f"{name} contains NaN or Inf")
    return result


def _average_ranks(values: Tensor) -> Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end].item() == sorted_values[start].item():
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _pearson(left: Tensor, right: Tensor) -> float | None:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(
        right_centered
    )
    if denominator.item() == 0.0:
        return None
    result = (left_centered * right_centered).sum() / denominator
    value = float(result.item())
    if not math.isfinite(value):
        raise HardFailure("correlation produced NaN or Inf")
    return max(-1.0, min(1.0, value))


def compute_correlations(uncertainty: Tensor, error: Tensor) -> dict[str, Any]:
    """Compute Pearson and tie-aware Spearman at one shared granularity."""
    uncertainty = _finite_vector(uncertainty, "uncertainty")
    error = _finite_vector(error, "error")
    if uncertainty.shape != error.shape:
        raise HardFailure("uncertainty and error must be aligned")
    pearson = _pearson(uncertainty, error)
    spearman = _pearson(_average_ranks(uncertainty), _average_ranks(error))
    warnings: list[str] = []
    if pearson is None or spearman is None:
        pearson = None
        spearman = None
        warnings.append("correlation undefined for constant input")
    return {"pearson": pearson, "spearman": spearman, "warnings": warnings}


def compute_risk_coverage(
    uncertainty: Tensor, error: Tensor, coverages: Sequence[float]
) -> list[dict[str, float | int]]:
    """Keep the least-uncertain samples and report their mean absolute error."""
    uncertainty = _finite_vector(uncertainty, "uncertainty")
    error = _finite_vector(error, "error")
    if uncertainty.shape != error.shape:
        raise HardFailure("uncertainty and error must be aligned")
    if not isinstance(coverages, Sequence) or len(coverages) == 0:
        raise HardFailure("coverages must be a nonempty sequence")
    order = torch.argsort(uncertainty, stable=True)
    rows: list[dict[str, float | int]] = []
    for coverage in coverages:
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
            raise HardFailure("coverage must be numeric")
        coverage_value = float(coverage)
        if not math.isfinite(coverage_value) or not 0.0 < coverage_value <= 1.0:
            raise HardFailure("coverage must be finite and in (0, 1]")
        count = max(1, math.ceil(coverage_value * error.numel()))
        risk = float(error[order[:count]].mean().item())
        rows.append({"coverage": coverage_value, "risk": risk, "count": count})
    return rows


def _aligned_prediction_tensor(value: Tensor, shape: tuple[int, ...], name: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise HardFailure(f"{name} must be a floating torch.Tensor")
    if tuple(value.shape) != shape:
        raise HardFailure(f"{name} has an invalid shape")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(result).all().item():
        raise HardFailure(f"{name} contains NaN or Inf")
    return result


def compute_errors(
    *,
    energy_prediction: Tensor,
    forces_prediction: Tensor,
    energy_reference: Tensor,
    forces_reference: Tensor,
    n_atoms: Tensor,
    atom_to_structure: Tensor,
    force_structure_quantile: float = 0.95,
) -> dict[str, Tensor]:
    """Compute absolute errors at every canonical energy/force granularity."""
    if not isinstance(n_atoms, Tensor) or n_atoms.dtype != torch.int64 or n_atoms.ndim != 1:
        raise HardFailure("n_atoms must be a one-dimensional int64 tensor")
    if n_atoms.numel() == 0 or (n_atoms <= 0).any().item():
        raise HardFailure("n_atoms must contain positive counts")
    structure_count = n_atoms.numel()
    atom_count = int(n_atoms.sum().item())
    if not isinstance(atom_to_structure, Tensor):
        raise HardFailure("atom_to_structure must be a tensor")
    energy_prediction = _aligned_prediction_tensor(
        energy_prediction, (structure_count,), "energy_prediction"
    )
    energy_reference = _aligned_prediction_tensor(
        energy_reference, (structure_count,), "energy_reference"
    )
    forces_prediction = _aligned_prediction_tensor(
        forces_prediction, (atom_count, 3), "forces_prediction"
    )
    forces_reference = _aligned_prediction_tensor(
        forces_reference, (atom_count, 3), "forces_reference"
    )
    if not 0.0 < float(force_structure_quantile) < 1.0:
        raise HardFailure("force structure quantile must be in (0, 1)")

    energy_total = (energy_prediction - energy_reference).abs()
    energy_per_atom = energy_total / n_atoms.to(dtype=torch.float64)
    force_component = (forces_prediction - forces_reference).abs()
    force_atom_vector = torch.linalg.vector_norm(
        forces_prediction - forces_reference, dim=-1
    )
    reduced = reduce_atoms_by_structure(
        force_atom_vector, atom_to_structure.to(device="cpu"), structure_count
    )
    if force_structure_quantile != 0.95:
        q_values = []
        for index in range(structure_count):
            selected = force_atom_vector[atom_to_structure == index]
            q_values.append(
                torch.quantile(
                    selected,
                    float(force_structure_quantile),
                    interpolation="linear",
                )
            )
        reduced["q95"] = torch.stack(q_values)
    return {
        "energy_total": energy_total,
        "energy_per_atom": energy_per_atom,
        "force_component": force_component,
        "force_atom_vector": force_atom_vector,
        "force_structure_mean": reduced["mean"],
        "force_structure_max": reduced["max"],
        "force_structure_q95": reduced["q95"],
    }
