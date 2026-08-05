"""Unbiased ensemble spread, GMD, and structure-level reductions."""

from __future__ import annotations

import torch
from torch import Tensor

from .aggregation import normalized_positive_weights, require_members
from .errors import HardFailure


def _weights_and_denominator(
    members: Tensor, weights: Tensor | None
) -> tuple[Tensor | None, Tensor | float]:
    if weights is None:
        return None, members.shape[0] - 1
    normalized = normalized_positive_weights(weights, members.shape[0]).to(
        device=members.device, dtype=members.dtype
    )
    denominator = 1.0 - normalized.square().sum()
    if not torch.isfinite(denominator).item() or denominator.item() <= 0:
        raise HardFailure("weighted variance denominator must be positive")
    return normalized, denominator


def unbiased_std(members: Tensor, weights: Tensor | None = None) -> Tensor:
    """Elementwise standard deviation with K-1 or normalized-weight correction."""
    members = require_members(members)
    normalized, denominator = _weights_and_denominator(members, weights)
    if normalized is None:
        mean = members.mean(dim=0)
        variance = (members - mean).square().sum(dim=0) / denominator
    else:
        view = normalized.reshape((-1,) + (1,) * (members.ndim - 1))
        mean = (view * members).sum(dim=0)
        variance = (view * (members - mean).square()).sum(dim=0) / denominator
    result = torch.sqrt(torch.clamp_min(variance, 0.0))
    if not torch.isfinite(result).all().item():
        raise HardFailure("standard deviation contains NaN or Inf")
    return result


def vector_std(members: Tensor, weights: Tensor | None = None) -> Tensor:
    """L2 member scatter for vectors stored on the final axis."""
    members = require_members(members)
    if members.ndim < 2 or members.shape[-1] != 3:
        raise HardFailure("vector members must have final dimension 3")
    normalized, denominator = _weights_and_denominator(members, weights)
    if normalized is None:
        mean = members.mean(dim=0)
        squared_distance = (members - mean).square().sum(dim=-1)
        variance = squared_distance.sum(dim=0) / denominator
    else:
        view = normalized.reshape((-1,) + (1,) * (members.ndim - 1))
        mean = (view * members).sum(dim=0)
        squared_distance = (members - mean).square().sum(dim=-1)
        scalar_view = normalized.reshape((-1,) + (1,) * (squared_distance.ndim - 1))
        variance = (scalar_view * squared_distance).sum(dim=0) / denominator
    result = torch.sqrt(torch.clamp_min(variance, 0.0))
    if not torch.isfinite(result).all().item():
        raise HardFailure("vector standard deviation contains NaN or Inf")
    return result


def _pair_weights(
    members: Tensor, weights: Tensor | None
) -> tuple[Tensor, Tensor, Tensor | None]:
    pair_indices = torch.triu_indices(
        members.shape[0], members.shape[0], offset=1, device=members.device
    )
    left, right = pair_indices[0], pair_indices[1]
    if weights is None:
        return left, right, None
    normalized = normalized_positive_weights(weights, members.shape[0]).to(
        device=members.device, dtype=members.dtype
    )
    pair_weights = normalized[left] * normalized[right]
    total = pair_weights.sum()
    if not torch.isfinite(total).item() or total.item() <= 0:
        raise HardFailure("GMD pair weight sum must be positive")
    return left, right, pair_weights / total


def scalar_gmd(members: Tensor, weights: Tensor | None = None) -> Tensor:
    """Mean absolute difference over unordered member pairs."""
    members = require_members(members)
    left, right, pair_weights = _pair_weights(members, weights)
    differences = (members[left] - members[right]).abs()
    if pair_weights is None:
        result = differences.mean(dim=0)
    else:
        view = pair_weights.reshape((-1,) + (1,) * (differences.ndim - 1))
        result = (view * differences).sum(dim=0)
    if not torch.isfinite(result).all().item():
        raise HardFailure("GMD contains NaN or Inf")
    return result


def vector_gmd(members: Tensor, weights: Tensor | None = None) -> Tensor:
    """Mean L2 vector difference over unordered member pairs."""
    members = require_members(members)
    if members.ndim < 2 or members.shape[-1] != 3:
        raise HardFailure("vector members must have final dimension 3")
    left, right, pair_weights = _pair_weights(members, weights)
    distances = torch.linalg.vector_norm(members[left] - members[right], dim=-1)
    if pair_weights is None:
        result = distances.mean(dim=0)
    else:
        view = pair_weights.reshape((-1,) + (1,) * (distances.ndim - 1))
        result = (view * distances).sum(dim=0)
    if not torch.isfinite(result).all().item():
        raise HardFailure("vector GMD contains NaN or Inf")
    return result


def reduce_atoms_by_structure(
    values: Tensor, atom_to_structure: Tensor, structure_count: int
) -> dict[str, Tensor]:
    """Reduce atom-first values into per-structure mean, max, and q95 tensors."""
    if not isinstance(values, Tensor) or not values.is_floating_point():
        raise HardFailure("atom values must be a floating torch.Tensor")
    if values.ndim < 1 or values.numel() == 0:
        raise HardFailure("atom values must not be empty")
    if not torch.isfinite(values).all().item():
        raise HardFailure("atom values contain NaN or Inf")
    if not isinstance(atom_to_structure, Tensor):
        raise HardFailure("atom_to_structure must be a torch.Tensor")
    if atom_to_structure.dtype != torch.int64 or atom_to_structure.ndim != 1:
        raise HardFailure("atom_to_structure must be one-dimensional int64")
    if atom_to_structure.shape[0] != values.shape[0]:
        raise HardFailure("atom values and structure mapping are misaligned")
    if type(structure_count) is not int or structure_count <= 0:
        raise HardFailure("structure_count must be a positive integer")
    if (atom_to_structure < 0).any().item() or (
        atom_to_structure >= structure_count
    ).any().item():
        raise HardFailure("atom_to_structure contains an invalid structure index")

    means: list[Tensor] = []
    maxima: list[Tensor] = []
    q95s: list[Tensor] = []
    for structure_index in range(structure_count):
        selected = values[atom_to_structure == structure_index]
        if selected.shape[0] == 0:
            raise HardFailure("every structure must contain at least one atom")
        means.append(selected.mean(dim=0))
        maxima.append(selected.max(dim=0).values)
        q95s.append(torch.quantile(selected, 0.95, dim=0, interpolation="linear"))
    return {
        "mean": torch.stack(means),
        "max": torch.stack(maxima),
        "q95": torch.stack(q95s),
    }
