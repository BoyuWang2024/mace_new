"""Independent checks for persisted ensemble metadata."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .errors import HardFailure


def assert_ensemble_weights(
    actual: Any, expected: Tensor | None, member_count: int, label: str
) -> None:
    if expected is None:
        expected = torch.full(
            (member_count,), 1.0 / member_count, dtype=torch.float64
        )
    else:
        expected = expected.to(device="cpu", dtype=torch.float64)
    if not isinstance(actual, Tensor):
        raise HardFailure(f"{label} is not a tensor")
    if actual.device.type != "cpu" or actual.dtype != torch.float64:
        raise HardFailure(f"{label} must be CPU float64")
    if not torch.equal(actual, expected):
        raise HardFailure(f"{label} value mismatch")
