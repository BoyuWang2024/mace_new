"""Pure numerical kernels for LLPR energy-reference postprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_composition(values: ArrayLike) -> NDArray[np.float64]:
    composition = np.asarray(values, dtype=np.float64)
    if composition.ndim != 2:
        raise ValueError("composition must be two-dimensional")
    if composition.shape[0] == 0 or composition.shape[1] == 0:
        raise ValueError("composition must not be empty")
    if not np.isfinite(composition).all():
        raise ValueError("composition must contain only finite values")
    if np.any(composition < 0.0):
        raise ValueError("composition must contain only non-negative counts")
    if not np.equal(composition, np.floor(composition)).all():
        raise ValueError("composition must contain only integer counts")
    if np.any(np.sum(composition, axis=1) == 0.0):
        raise ValueError("each composition row must contain at least one atom")
    return composition


def apply_direct_test_atomic_baseline(
    *,
    raw_total: ArrayLike,
    reference_total: ArrayLike,
    atomization_total: ArrayLike,
    composition: ArrayLike,
    model_e0: ArrayLike,
) -> NDArray[np.float64]:
    """Replace the model atomic baseline with each test structure's MAD baseline."""
    raw = _finite_vector(raw_total, "raw_total")
    reference = _finite_vector(reference_total, "reference_total")
    atomization = _finite_vector(atomization_total, "atomization_total")
    counts = _finite_composition(composition)
    atomic_energies = _finite_vector(model_e0, "model_e0")

    rows = raw.shape[0]
    if not (
        reference.shape[0] == rows
        and atomization.shape[0] == rows
        and counts.shape[0] == rows
    ):
        raise ValueError(
            "structure vectors and composition must have the same number of rows"
        )
    if counts.shape[1] != atomic_energies.shape[0]:
        raise ValueError("composition columns must match model_e0")

    model_baseline = counts @ atomic_energies
    mad_baseline = reference - atomization
    return raw - model_baseline + mad_baseline


@dataclass(frozen=True)
class ModelAwareReestimation:
    """Result of the paper's validation-only atomic-reference reestimation."""

    delta_e0: NDArray[np.float64]
    new_e0: NDArray[np.float64]
    corrected_validation_total: NDArray[np.float64]
    rank: int
    singular_values: NDArray[np.float64]
    residual_norm: float


def apply_model_aware_correction(
    *,
    raw_total: ArrayLike,
    composition: ArrayLike,
    delta_e0: ArrayLike,
) -> NDArray[np.float64]:
    """Apply ``E_raw + A @ DeltaE0`` to one prediction per structure."""
    raw = _finite_vector(raw_total, "raw_total")
    counts = _finite_composition(composition)
    correction = _finite_vector(delta_e0, "delta_e0")
    if counts.shape[0] != raw.shape[0]:
        raise ValueError(
            "structure vectors and composition must have the same number of rows"
        )
    if counts.shape[1] != correction.shape[0]:
        raise ValueError("composition columns must match delta_e0")
    return raw + counts @ correction


def fit_model_aware_reestimation(
    *,
    composition: ArrayLike,
    reference_total: ArrayLike,
    raw_total: ArrayLike,
    model_e0: ArrayLike,
) -> ModelAwareReestimation:
    """Fit ``A @ DeltaE0 = E_reference - E_raw`` with SVD least squares."""
    counts = _finite_composition(composition)
    reference = _finite_vector(reference_total, "reference_total")
    raw = _finite_vector(raw_total, "raw_total")
    atomic_energies = _finite_vector(model_e0, "model_e0")
    if not (counts.shape[0] == reference.shape[0] == raw.shape[0]):
        raise ValueError(
            "structure vectors and composition must have the same number of rows"
        )
    if counts.shape[1] != atomic_energies.shape[0]:
        raise ValueError("composition columns must match model_e0")

    error = reference - raw
    delta, _residuals, rank, singular_values = np.linalg.lstsq(
        counts, error, rcond=None
    )
    corrected = apply_model_aware_correction(
        raw_total=raw,
        composition=counts,
        delta_e0=delta,
    )
    residual_norm = float(np.linalg.norm(reference - corrected))
    return ModelAwareReestimation(
        delta_e0=delta,
        new_e0=atomic_energies + delta,
        corrected_validation_total=corrected,
        rank=int(rank),
        singular_values=singular_values,
        residual_norm=residual_norm,
    )


def calibrate_energy_alpha(
    *,
    reference_total: ArrayLike,
    prediction_total: ArrayLike,
    num_atoms: ArrayLike,
    q: ArrayLike,
    min_q: float = 1.0e-30,
) -> float:
    """Calibrate energy Alpha from corrected total-energy residuals and LLPR q."""
    reference = _finite_vector(reference_total, "reference_total")
    prediction = _finite_vector(prediction_total, "prediction_total")
    atoms = _finite_vector(num_atoms, "num_atoms")
    q_values = _finite_vector(q, "q")
    if not (reference.shape == prediction.shape == atoms.shape == q_values.shape):
        raise ValueError("energy calibration vectors must have the same shape")
    if reference.size == 0:
        raise ValueError("energy calibration vectors must not be empty")
    if np.any(atoms <= 0.0):
        raise ValueError("num_atoms must contain only positive values")
    if np.any(q_values <= 0.0):
        raise ValueError("q must contain only positive values")
    floor = float(min_q)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("min_q must be finite and positive")

    residual_per_atom = (reference - prediction) / atoms
    ratios = np.square(residual_per_atom) / np.maximum(q_values, floor)
    return float(np.sqrt(np.mean(ratios)))
