"""Trainable local-to-global cumulant feature adapter."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dropout_probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("dropout must be a finite number in [0, 1)")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("dropout must be a finite number in [0, 1)")
    return probability


def _validate_features(features: torch.Tensor, input_dim: int) -> None:
    if not isinstance(features, torch.Tensor):
        raise ValueError("features must be a tensor")
    if features.ndim != 2:
        raise ValueError("features must have rank 2")
    if features.shape[1] != input_dim:
        raise ValueError(f"features dimension must equal {input_dim}")
    if not features.is_floating_point():
        raise ValueError("features must have a floating dtype")
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite")


def _validate_offsets(offsets: torch.Tensor, features: torch.Tensor) -> list[int]:
    if not isinstance(offsets, torch.Tensor):
        raise ValueError("offsets must be a tensor")
    if offsets.ndim != 1:
        raise ValueError("offsets must be one-dimensional")
    if (
        offsets.dtype == torch.bool
        or offsets.is_floating_point()
        or offsets.is_complex()
    ):
        raise ValueError("offsets must contain integral boundaries")
    if offsets.device != features.device:
        raise ValueError("offsets and features must be on the same device")
    if offsets.numel() < 1:
        raise ValueError("offsets must contain structure boundaries")

    boundaries = [int(value) for value in offsets.tolist()]
    if boundaries[0] != 0:
        raise ValueError("offsets must start at 0")
    if boundaries[-1] != features.shape[0]:
        raise ValueError("offsets must end at the atom count")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("offsets must be strictly increasing without empty structures")
    return boundaries


class LocalToGlobalCumulantAdapter(nn.Module):
    """Aggregate atom features into per-structure cumulants and project them."""

    def __init__(
        self,
        input_dim: int,
        order: int,
        projection_dim: int = 512,
        signed_root: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, "input_dim")
        self.order = _positive_integer(order, "order")
        if self.order > 5:
            raise ValueError("order must be in 1..5")
        self.projection_dim = _positive_integer(projection_dim, "projection_dim")
        if not isinstance(signed_root, bool):
            raise ValueError("signed_root must be a boolean")
        self.signed_root = signed_root
        probability = _dropout_probability(dropout)

        self.projection = nn.Linear(self.input_dim * self.order, self.projection_dim)
        self.normalization = nn.LayerNorm(self.projection_dim)
        self.dropout = nn.Dropout(probability) if probability > 0.0 else None

    def cumulants(self, features: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Return orders 1..K, concatenated in order-major layout."""
        _validate_features(features, self.input_dim)
        boundaries = _validate_offsets(offsets, features)
        if len(boundaries) == 1:
            return features.new_empty((0, self.input_dim * self.order))

        moments = [
            torch.stack(
                [
                    torch.mean(features[start:end].pow(rank), dim=0)
                    for start, end in zip(boundaries, boundaries[1:])
                ]
            )
            for rank in range(1, self.order + 1)
        ]
        raw_cumulants: list[torch.Tensor] = []
        for rank, moment in enumerate(moments, start=1):
            correction = torch.zeros_like(moment)
            for index in range(1, rank):
                correction = correction + (
                    math.comb(rank - 1, index - 1)
                    * raw_cumulants[index - 1]
                    * moments[rank - index - 1]
                )
            raw_cumulants.append(moment - correction)

        transformed = [raw_cumulants[0]]
        for rank, cumulant in enumerate(raw_cumulants[1:], start=2):
            if self.signed_root:
                zero = cumulant == 0
                safe_magnitude = torch.where(
                    zero, torch.ones_like(cumulant), torch.abs(cumulant)
                )
                rooted = torch.sign(cumulant) * safe_magnitude.pow(1.0 / rank)
                transformed.append(
                    torch.where(zero, torch.zeros_like(cumulant), rooted)
                )
            else:
                transformed.append(cumulant)
        return torch.cat(transformed, dim=1)

    def forward(self, features: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        projected = self.projection(self.cumulants(features, offsets))
        normalized = self.normalization(projected)
        if self.dropout is not None:
            return self.dropout(normalized)
        return normalized
