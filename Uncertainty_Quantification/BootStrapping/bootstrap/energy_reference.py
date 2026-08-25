"""Pure numerical helpers for post-inference atomic reference corrections."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .errors import HardFailure


@dataclass(frozen=True)
class ReferenceFit:
    values: np.ndarray
    rank: int
    singular_values: np.ndarray
    residual: np.ndarray
    max_abs_residual: float
    rmse_residual: float


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise HardFailure(f"{name} contains NaN or Inf")
    return array


def count_matrix(
    atomic_numbers: Sequence[np.ndarray],
    supported_atomic_numbers: Sequence[int],
) -> np.ndarray:
    """Build a structure-by-element atom-count matrix in a fixed element order."""
    supported = tuple(int(value) for value in supported_atomic_numbers)
    if not supported or len(set(supported)) != len(supported):
        raise HardFailure("supported atomic numbers must be unique and non-empty")
    positions = {number: index for index, number in enumerate(supported)}
    matrix = np.zeros((len(atomic_numbers), len(supported)), dtype=float)
    for row, numbers in enumerate(atomic_numbers):
        values = np.asarray(numbers, dtype=int).reshape(-1)
        unknown = sorted(set(int(value) for value in values) - set(positions))
        if unknown:
            raise HardFailure(f"unsupported atomic numbers: {unknown}")
        for number in values:
            matrix[row, positions[int(number)]] += 1.0
    if not matrix.size or not np.isfinite(matrix).all():
        raise HardFailure("atom-count matrix is empty or non-finite")
    return matrix


def _fit(matrix: np.ndarray, target: np.ndarray, *, name: str) -> ReferenceFit:
    design = _finite("atom-count matrix", matrix)
    values_target = _finite(name, target).reshape(-1)
    if design.ndim != 2 or design.shape[0] != values_target.size:
        raise HardFailure(f"{name} shape does not match atom-count matrix")
    if design.shape[1] == 0:
        raise HardFailure("atom-count matrix has no elements")
    values, _, rank, singular_values = np.linalg.lstsq(design, values_target, rcond=None)
    if int(rank) != design.shape[1]:
        raise HardFailure(f"atom-count matrix is rank deficient: {rank}/{design.shape[1]}")
    residual = design @ values - values_target
    residual = _finite("reference fit residual", residual)
    return ReferenceFit(
        values=np.asarray(values, dtype=float),
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
        residual=residual,
        max_abs_residual=float(np.max(np.abs(residual))),
        rmse_residual=float(np.sqrt(np.mean(residual**2))),
    )


def recover_mad_e0(matrix: np.ndarray, total_energy: np.ndarray, atomization_energy: np.ndarray) -> ReferenceFit:
    """Recover MAD E0 from the identity total energy - atomization energy."""
    total = _finite("total energy", total_energy).reshape(-1)
    atomization = _finite("atomization energy", atomization_energy).reshape(-1)
    if total.shape != atomization.shape:
        raise HardFailure("total and atomization energy shapes differ")
    return _fit(matrix, total - atomization, name="MAD E0 target")


def fit_model_aware_delta(matrix: np.ndarray, reference_energy: np.ndarray, predicted_mean: np.ndarray) -> ReferenceFit:
    """Fit one common element correction from validation ensemble mean predictions."""
    reference = _finite("reference energy", reference_energy).reshape(-1)
    predicted = _finite("predicted validation mean", predicted_mean).reshape(-1)
    if reference.shape != predicted.shape:
        raise HardFailure("reference and predicted validation energy shapes differ")
    return _fit(matrix, reference - predicted, name="model-aware delta target")


def apply_energy_correction(member_energies: np.ndarray, matrix: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a common composition-linear correction to one or more members."""
    raw = _finite("member energies", member_energies)
    design = _finite("atom-count matrix", matrix)
    correction = _finite("energy correction", delta).reshape(-1)
    if design.ndim != 2 or correction.size != design.shape[1]:
        raise HardFailure("energy correction shape does not match atom-count matrix")
    if raw.shape[-1] != design.shape[0]:
        raise HardFailure("member energy structure count does not match atom-count matrix")
    return raw + np.asarray(design @ correction)[None, ...] if raw.ndim == 2 else raw + design @ correction


def validate_common_shift(raw: np.ndarray, corrected: np.ndarray, *, rtol: float = 1e-12, atol: float = 1e-10) -> None:
    """Verify that a common energy shift preserves member STD and GMD."""
    before = np.asarray(raw, dtype=float)
    after = np.asarray(corrected, dtype=float)
    if before.shape != after.shape or before.ndim != 2 or before.shape[0] < 2:
        raise HardFailure("common-shift validation requires matching 2-D member arrays")
    std_before = np.std(before, axis=0, ddof=1)
    std_after = np.std(after, axis=0, ddof=1)
    if not np.allclose(std_before, std_after, rtol=rtol, atol=atol):
        raise HardFailure("energy STD changed after common correction")
    gmd_before = np.mean(np.abs(before[:, None, :] - before[None, :, :]), axis=(0, 1))
    gmd_after = np.mean(np.abs(after[:, None, :] - after[None, :, :]), axis=(0, 1))
    if not np.allclose(gmd_before, gmd_after, rtol=rtol, atol=atol):
        raise HardFailure("energy GMD changed after common correction")


__all__ = [
    "ReferenceFit",
    "apply_energy_correction",
    "count_matrix",
    "fit_model_aware_delta",
    "recover_mad_e0",
    "validate_common_shift",
]
