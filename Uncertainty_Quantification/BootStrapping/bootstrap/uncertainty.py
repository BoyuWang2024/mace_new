"""Sample STD and distinct-pair GMD for native ensemble predictions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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


def compute_domain_uncertainty(
    members: Sequence[PredictionArrays],
    *,
    domains: Iterable[str] = ("energy", "forces", "stress"),
    num_atoms: np.ndarray | None = None,
    ddof: int = 1,
) -> dict[str, NDArray[np.floating]]:
    """Compute uncertainty only for requested physical domains."""
    selected = tuple(dict.fromkeys(domains))
    allowed = {"energy", "forces", "stress"}
    if not selected or any(domain not in allowed for domain in selected):
        raise HardFailure("domains must be a non-empty subset of energy, forces, stress")
    if ddof != 1:
        raise HardFailure("uncertainty ddof must be 1")
    if len(members) < 2:
        raise HardFailure("sample uncertainty requires at least two members")
    result: dict[str, NDArray[np.floating]] = {}
    if "energy" in selected:
        energy = _stack(members, "energy")
        result["energy_std"] = np.std(energy, axis=0, ddof=ddof)
        result["energy_gmd"] = _gmd(energy)
        counts = np.ones(energy.shape[1:], dtype=float) if num_atoms is None else np.asarray(num_atoms, dtype=float)
        if counts.shape != energy.shape[1:] or np.any(counts <= 0):
            raise HardFailure("num_atoms shape must match energy and contain positive values")
        per_atom = energy / counts
        result["energy_per_atom_std"] = np.std(per_atom, axis=0, ddof=ddof)
        result["energy_per_atom_gmd"] = _gmd(per_atom)
    if "forces" in selected:
        forces = _stack(members, "forces")
        result["force_std"] = np.std(forces, axis=0, ddof=ddof)
        result["force_gmd"] = _gmd(forces)
    if "stress" in selected:
        raw_stress = _stack(members, "stress")
        symmetric = 0.5 * (raw_stress + np.swapaxes(raw_stress, -1, -2))
        stress = symmetric[..., [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        result["stress_std"] = np.std(stress, axis=0, ddof=ddof)
        result["stress_gmd"] = _gmd(stress)
    return result
