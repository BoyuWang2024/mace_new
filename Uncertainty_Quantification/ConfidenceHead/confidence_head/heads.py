"""Fixed-layout classification heads for force and energy confidence."""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

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


class ConfidenceHead(nn.Module):
    """MLP producing unnormalized confidence-bin logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        num_bins: int,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, "input_dim")
        if isinstance(hidden_dims, (str, bytes)) or not isinstance(
            hidden_dims, Sequence
        ):
            raise ValueError("hidden_dims must be a non-empty sequence")
        dimensions = tuple(
            _positive_integer(dimension, "hidden dimension")
            for dimension in hidden_dims
        )
        if not dimensions:
            raise ValueError("hidden_dims must be a non-empty sequence")
        probability = _dropout_probability(dropout)
        self.num_bins = _positive_integer(num_bins, "num_bins")
        if self.num_bins < 3:
            raise ValueError("num_bins must be at least 3")

        modules: list[nn.Module] = []
        previous_dim = self.input_dim
        for hidden_dim in dimensions:
            modules.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            if probability > 0.0:
                modules.append(nn.Dropout(probability))
            previous_dim = hidden_dim
        modules.append(nn.Linear(previous_dim, self.num_bins))
        self.network = nn.Sequential(*modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise ValueError("features must be a tensor")
        if features.ndim != 2:
            raise ValueError("features must have rank 2")
        if features.shape[1] != self.input_dim:
            raise ValueError(f"features dimension must equal {self.input_dim}")
        if not features.is_floating_point():
            raise ValueError("features must have a floating dtype")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        return self.network(features)


class ComponentConfidenceHead(nn.Module):
    """Three parameter-independent confidence heads for force components."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        num_bins: int,
    ) -> None:
        super().__init__()
        self.components = nn.ModuleList(
            [
                ConfidenceHead(input_dim, hidden_dims, dropout, num_bins)
                for _ in range(3)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [component(features) for component in self.components], dim=1
        )
