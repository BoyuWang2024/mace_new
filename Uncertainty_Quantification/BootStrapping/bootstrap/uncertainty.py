"""Sample STD and distinct-pair GMD for native ensemble predictions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .aggregation import _stack
from .errors import HardFailure
from .prediction import PredictionArrays


@dataclass(frozen=True)
class UncertaintyArrays:
    energy_std: NDArray[np.floating]
    force_std: NDArray[np.floating]
    stress_std: NDArray[np.floating]
    energy_gmd: NDArray[np.floating]
    force_gmd: NDArray[np.floating]
    stress_gmd: NDArray[np.floating]


def _gmd(values: np.ndarray) -> np.ndarray:
    members = values.shape[0]
    pairs = [np.abs(values[left] - values[right]) for left in range(members) for right in range(left + 1, members)]
    if not pairs:
        raise HardFailure("GMD requires at least two members")
    return np.mean(np.stack(pairs, axis=0), axis=0)


def compute_uncertainty(members: Sequence[PredictionArrays], *, ddof: int = 1) -> UncertaintyArrays:
    if ddof != 1:
        raise HardFailure("uncertainty ddof must be 1")
    if len(members) < 2:
        raise HardFailure("sample uncertainty requires at least two members")
    energy = _stack(members, "energy")
    forces = _stack(members, "forces")
    stress = _stack(members, "stress")
    return UncertaintyArrays(
        energy_std=np.std(energy, axis=0, ddof=1),
        force_std=np.std(forces, axis=0, ddof=1),
        stress_std=np.std(stress, axis=0, ddof=1),
        energy_gmd=_gmd(energy),
        force_gmd=_gmd(forces),
        stress_gmd=_gmd(stress),
    )
