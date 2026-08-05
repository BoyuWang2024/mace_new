"""Validated member aggregation and validation-error weighting."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .errors import HardFailure


def require_members(members: Tensor) -> Tensor:
    """Return a valid floating member tensor whose first axis is the ensemble."""
    if not isinstance(members, Tensor):
        raise HardFailure("members must be a torch.Tensor")
    if not members.is_floating_point():
        raise HardFailure("members must use a floating dtype")
    if members.ndim < 1 or members.shape[0] < 2:
        raise HardFailure("members must contain at least two ensemble members")
    if members.numel() == 0:
        raise HardFailure("members must not be empty")
    if not torch.isfinite(members).all().item():
        raise HardFailure("members contain NaN or Inf")
    return members


def normalized_positive_weights(weights: Tensor, member_count: int) -> Tensor:
    """Validate strictly positive member weights and normalize them to sum to one."""
    if not isinstance(weights, Tensor):
        raise HardFailure("weights must be a torch.Tensor")
    if not weights.is_floating_point():
        raise HardFailure("weights must use a floating dtype")
    if weights.ndim != 1 or weights.shape[0] != member_count:
        raise HardFailure("weights must have one entry for every member")
    if not torch.isfinite(weights).all().item():
        raise HardFailure("weights contain NaN or Inf")
    if not (weights > 0).all().item():
        raise HardFailure("weights must be strictly positive")
    total = weights.sum()
    if not torch.isfinite(total).item() or total.item() <= 0:
        raise HardFailure("weight sum must be finite and positive")
    normalized = weights / total
    if not torch.isfinite(normalized).all().item():
        raise HardFailure("normalized weights contain NaN or Inf")
    return normalized


def validation_error_weights(
    rmse: Tensor, base_rmse: float, eps_ratio: float
) -> Tensor:
    """Compute normalized inverse-error weights using ``eps_ratio * base_rmse``."""
    if not isinstance(rmse, Tensor) or not rmse.is_floating_point():
        raise HardFailure("rmse must be a floating torch.Tensor")
    if rmse.ndim != 1 or rmse.numel() < 2:
        raise HardFailure("rmse must contain at least two member values")
    if not torch.isfinite(rmse).all().item() or (rmse < 0).any().item():
        raise HardFailure("rmse values must be finite and nonnegative")
    if type(base_rmse) not in (float, int) or not math.isfinite(base_rmse):
        raise HardFailure("base_rmse must be finite")
    if type(eps_ratio) not in (float, int) or not math.isfinite(eps_ratio):
        raise HardFailure("eps_ratio must be finite")
    epsilon = float(base_rmse) * float(eps_ratio)
    if epsilon <= 0:
        raise HardFailure("base-scaled epsilon must be positive")
    inverse = (rmse + epsilon).reciprocal()
    return normalized_positive_weights(inverse, rmse.shape[0])


def weighted_mean(members: Tensor, weights: Tensor | None = None) -> Tensor:
    """Aggregate members along axis zero, equally or with normalized weights."""
    members = require_members(members)
    if weights is None:
        return members.mean(dim=0)
    normalized = normalized_positive_weights(weights, members.shape[0]).to(
        device=members.device, dtype=members.dtype
    )
    view = normalized.reshape((-1,) + (1,) * (members.ndim - 1))
    result = (view * members).sum(dim=0)
    if not torch.isfinite(result).all().item():
        raise HardFailure("weighted mean contains NaN or Inf")
    return result
