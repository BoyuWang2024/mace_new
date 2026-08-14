"""Ensemble aggregation for native BootStrapping predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .errors import HardFailure
from .prediction import PredictionArrays


def _stack(members: Sequence[PredictionArrays], field: str) -> np.ndarray:
    if not members:
        raise HardFailure("ensemble must contain at least one member")
    values = [np.asarray(getattr(member, field)) for member in members]
    if any(value.shape != values[0].shape for value in values[1:]):
        raise HardFailure(f"ensemble {field} shapes differ")
    return np.stack(values, axis=0)


def ensemble_mean(members: Sequence[PredictionArrays]) -> PredictionArrays:
    return PredictionArrays(
        energy=np.mean(_stack(members, "energy"), axis=0),
        forces=np.mean(_stack(members, "forces"), axis=0),
        stress=np.mean(_stack(members, "stress"), axis=0),
    )
