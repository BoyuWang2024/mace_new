"""Pure float64 kernels for post-inference atomic-reference corrections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


class E0CorrectionError(ValueError):
    """An E0 correction input or numerical result violates its contract."""


@dataclass(frozen=True)
class E0Replacement:
    """Per-structure replacement of the checkpoint atomic baseline."""

    model_baseline: NDArray[np.float64]
    mad_baseline: NDArray[np.float64]
    correction: NDArray[np.float64]
    corrected_total: NDArray[np.float64]


@dataclass(frozen=True)
class E0Application:
    """A composition-linear correction applied to raw total energies."""

    correction: NDArray[np.float64]
    corrected_total: NDArray[np.float64]


@dataclass(frozen=True)
class E0Fit:
    """Diagnostics for validation-only ordinary least squares."""

    delta_e0: NDArray[np.float64]
    corrected_validation_total: NDArray[np.float64]
    rank: int
    singular_values: NDArray[np.float64]
    condition_number: float
    residual_sum_squares: float
    residual_rmse: float
    residual_max_abs: float


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise E0CorrectionError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise E0CorrectionError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise E0CorrectionError(f"{name} must be finite")
    return array


def _composition(values: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise E0CorrectionError("composition must be two-dimensional")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise E0CorrectionError("composition must not be empty")
    if not np.isfinite(matrix).all():
        raise E0CorrectionError("composition must be finite")
    if np.any(matrix < 0.0):
        raise E0CorrectionError("composition counts must be non-negative")
    if not np.equal(matrix, np.floor(matrix)).all():
        raise E0CorrectionError("composition counts must be integers")
    if np.any(matrix.sum(axis=1) <= 0.0):
        raise E0CorrectionError("each composition row must contain atoms")
    return matrix


def _same_rows(matrix: np.ndarray, *vectors: np.ndarray) -> None:
    if any(vector.shape[0] != matrix.shape[0] for vector in vectors):
        raise E0CorrectionError(
            "structure vectors and composition must have the same rows"
        )


def composition_matrix(
    atomic_numbers: Sequence[ArrayLike],
    supported_atomic_numbers: Sequence[int],
) -> NDArray[np.float64]:
    """Build structure-by-element counts in checkpoint element order."""
    supported = tuple(int(number) for number in supported_atomic_numbers)
    if not supported:
        raise E0CorrectionError("supported atomic numbers must not be empty")
    if len(set(supported)) != len(supported) or any(number < 1 for number in supported):
        raise E0CorrectionError(
            "supported atomic numbers must be unique positive integers"
        )
    if not atomic_numbers:
        raise E0CorrectionError("atomic-number structures must not be empty")
    positions = {number: index for index, number in enumerate(supported)}
    matrix = np.zeros((len(atomic_numbers), len(supported)), dtype=np.float64)
    for row, numbers in enumerate(atomic_numbers):
        raw = np.asarray(numbers)
        if raw.ndim != 1 or raw.size == 0:
            raise E0CorrectionError(
                f"atomic numbers for structure {row} must be a non-empty vector"
            )
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise E0CorrectionError(
                f"atomic numbers for structure {row} must be integers"
            ) from error
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise E0CorrectionError(
                f"atomic numbers for structure {row} must be finite integers"
            )
        values = numeric.astype(np.int64)
        unknown = sorted(set(values.tolist()) - set(positions))
        if unknown:
            raise E0CorrectionError(f"unsupported atomic numbers: {unknown}")
        for number in values:
            matrix[row, positions[int(number)]] += 1.0
    return matrix


def apply_e0_replace(
    *,
    raw_total: ArrayLike,
    reference_total: ArrayLike,
    atomization_total: ArrayLike,
    composition: ArrayLike,
    model_e0: ArrayLike,
) -> E0Replacement:
    """Replace checkpoint E0 with each test structure MAD E0 baseline."""
    raw = _vector(raw_total, "raw_total")
    reference = _vector(reference_total, "reference_total")
    atomization = _vector(atomization_total, "atomization_total")
    counts = _composition(composition)
    atomic_energies = _vector(model_e0, "model_e0")
    _same_rows(counts, raw, reference, atomization)
    if counts.shape[1] != atomic_energies.shape[0]:
        raise E0CorrectionError("composition columns must match model_e0")
    model_baseline = counts @ atomic_energies
    mad_baseline = reference - atomization
    correction = mad_baseline - model_baseline
    corrected = raw + correction
    return E0Replacement(
        model_baseline=model_baseline,
        mad_baseline=mad_baseline,
        correction=correction,
        corrected_total=corrected,
    )


def fit_e0_reestimate(
    *,
    composition: ArrayLike,
    reference_total: ArrayLike,
    raw_total: ArrayLike,
) -> E0Fit:
    """Fit unweighted total-energy OLS on validation data only."""
    counts = _composition(composition)
    reference = _vector(reference_total, "reference_total")
    raw = _vector(raw_total, "raw_total")
    _same_rows(counts, reference, raw)
    delta, _, rank, singular_values = np.linalg.lstsq(
        counts, reference - raw, rcond=None
    )
    columns = counts.shape[1]
    if int(rank) != columns:
        raise E0CorrectionError(
            f"validation composition is rank deficient: {int(rank)}/{columns}"
        )
    if singular_values.shape != (columns,) or np.any(singular_values <= 0.0):
        raise E0CorrectionError("validation singular values are invalid")
    corrected = raw + counts @ delta
    residual = reference - corrected
    if not np.isfinite(delta).all() or not np.isfinite(residual).all():
        raise E0CorrectionError("E0 fit produced non-finite values")
    return E0Fit(
        delta_e0=np.asarray(delta, dtype=np.float64),
        corrected_validation_total=np.asarray(corrected, dtype=np.float64),
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=np.float64),
        condition_number=float(singular_values[0] / singular_values[-1]),
        residual_sum_squares=float(residual @ residual),
        residual_rmse=float(np.sqrt(np.mean(np.square(residual)))),
        residual_max_abs=float(np.max(np.abs(residual))),
    )


def apply_e0_reestimate(
    *,
    raw_total: ArrayLike,
    composition: ArrayLike,
    delta_e0: ArrayLike,
) -> E0Application:
    """Apply a fixed validation-fitted E0 correction to test totals."""
    raw = _vector(raw_total, "raw_total")
    counts = _composition(composition)
    delta = _vector(delta_e0, "delta_e0")
    _same_rows(counts, raw)
    if counts.shape[1] != delta.shape[0]:
        raise E0CorrectionError("composition columns must match delta_e0")
    correction = counts @ delta
    return E0Application(
        correction=correction,
        corrected_total=raw + correction,
    )


def energy_errors_per_atom(
    *,
    reference_total: ArrayLike,
    prediction_total: ArrayLike,
    num_atoms: ArrayLike,
) -> NDArray[np.float64]:
    """Return absolute total-energy residual normalized by structure size."""
    reference = _vector(reference_total, "reference_total")
    prediction = _vector(prediction_total, "prediction_total")
    counts = _vector(num_atoms, "num_atoms")
    if not (reference.shape == prediction.shape == counts.shape):
        raise E0CorrectionError("energy vectors must have the same shape")
    if np.any(counts <= 0.0):
        raise E0CorrectionError("num_atoms must be positive")
    return np.abs(reference - prediction) / counts


__all__ = [
    "E0Application",
    "E0CorrectionError",
    "E0Fit",
    "E0Replacement",
    "apply_e0_reestimate",
    "apply_e0_replace",
    "composition_matrix",
    "energy_errors_per_atom",
    "fit_e0_reestimate",
]
