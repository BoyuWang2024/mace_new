"""Strict NumPy kernels for E0 correction and calibration."""

from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any

import numpy as np

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.postprocessing.e0_correction.models import (
    CalibrationFit,
    WarningRecord,
)


_MEMBER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _fail(path: str, message: str) -> None:
    raise HardFailure(f"{path}: {message}")


def _float_array(value: Any, path: str, *, ndim: int) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HardFailure(f"{path}: must be a regular numeric array") from exc
    if source.dtype.kind not in "iuf":
        _fail(path, "must contain only real numeric values")
    if source.ndim != ndim:
        _fail(path, f"must be {ndim}-dimensional")
    if any(size == 0 for size in source.shape):
        _fail(path, "must not be empty")
    try:
        result = np.array(source, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HardFailure(f"{path}: cannot be represented as float64") from exc
    if not np.all(np.isfinite(result)):
        _fail(path, "must contain only finite values")
    return result


def _energy_vector(value: Any, path: str) -> np.ndarray:
    return _float_array(value, path, ndim=1)


def _composition(value: Any, path: str) -> np.ndarray:
    result = _float_array(value, path, ndim=2)
    if np.any(result < 0.0):
        _fail(path, "must contain only nonnegative counts")
    return result


def _atomic_numbers(
    value: Any, path: str, *, column_count: int
) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        _fail(path, "must be a nonempty tuple")
    if len(value) != column_count:
        _fail(path, "must align one-to-one with the element columns")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in value
    ):
        _fail(path, "must contain only positive integers")
    if len(set(value)) != len(value):
        _fail(path, "must contain unique atomic numbers")
    return value


def _aligned_atomic_numbers(
    left: Any,
    right: Any,
    *,
    left_path: str,
    right_path: str,
    left_columns: int,
    right_columns: int,
) -> tuple[int, ...]:
    left_numbers = _atomic_numbers(
        left, left_path, column_count=left_columns
    )
    right_numbers = _atomic_numbers(
        right, right_path, column_count=right_columns
    )
    if left_numbers != right_numbers:
        _fail(right_path, f"must exactly match {left_path} in column order")
    return left_numbers


def _warning_threshold(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail(path, "must be a finite nonnegative number or None")
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise HardFailure(
            f"{path}: must be a finite nonnegative number or None"
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        _fail(path, "must be a finite nonnegative number or None")
    return result


def _member_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _MEMBER_ID.fullmatch(value) is None
    ):
        _fail("member_id", "must be a safe nonempty identifier")
    return value


def _composition_pair(
    *,
    composition_val: Any,
    composition_test: Any,
    val_atomic_numbers: Any,
    test_atomic_numbers: Any,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    val = _composition(composition_val, "composition_val")
    test = _composition(composition_test, "composition_test")
    atomic_numbers = _aligned_atomic_numbers(
        val_atomic_numbers,
        test_atomic_numbers,
        left_path="val_atomic_numbers",
        right_path="test_atomic_numbers",
        left_columns=val.shape[1],
        right_columns=test.shape[1],
    )
    return val, test, atomic_numbers


def _identifiability(
    composition_val: np.ndarray,
    composition_test: np.ndarray,
    condition_number_warning_threshold: float | None,
) -> tuple[dict[str, Any], tuple[WarningRecord, ...]]:
    uncovered = np.logical_and(
        np.all(composition_val == 0.0, axis=0),
        np.any(composition_test != 0.0, axis=0),
    )
    if np.any(uncovered):
        _fail(
            "composition_test",
            "uses an element column absent from validation",
        )

    try:
        _, singular_values, vh = np.linalg.svd(
            composition_val, full_matrices=True
        )
    except np.linalg.LinAlgError as exc:
        raise HardFailure("composition_val: SVD did not converge") from exc
    if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
        _fail("composition_val", "has no finite identifiable singular direction")

    rank_tol = float(
        np.finfo(np.float64).eps
        * max(composition_val.shape)
        * max(float(singular_values[0]), 1.0)
    )
    rank = int(np.count_nonzero(singular_values > rank_tol))
    if rank == 0:
        _fail("composition_val", "has no identifiable element direction")

    nullspace = vh[rank:].T
    with np.errstate(over="ignore", invalid="ignore"):
        projected_nullspace = composition_test @ nullspace
    if not np.all(np.isfinite(projected_nullspace)):
        _fail("composition_test", "produces nonfinite nullspace impact")
    nullspace_test_impact = (
        float(np.max(np.abs(projected_nullspace)))
        if nullspace.shape[1]
        else 0.0
    )
    try:
        test_norm = float(np.linalg.norm(composition_test, ord=2))
    except np.linalg.LinAlgError as exc:
        raise HardFailure("composition_test: norm computation failed") from exc
    impact_tol = float(rank_tol * max(1.0, test_norm))
    if not math.isfinite(impact_tol):
        _fail("composition_test", "produces a nonfinite impact tolerance")
    if nullspace_test_impact > impact_tol:
        _fail(
            "composition_test",
            "depends on validation-unidentifiable element combinations",
        )

    condition_number = float(
        singular_values[0] / singular_values[rank - 1]
    )
    if not math.isfinite(condition_number):
        _fail("composition_val", "has a nonfinite effective condition number")

    diagnostics: dict[str, Any] = {
        "rank": rank,
        "full_column_count": int(composition_val.shape[1]),
        "singular_values": tuple(float(value) for value in singular_values),
        "condition_number": condition_number,
        "rank_tol": rank_tol,
        "impact_tol": impact_tol,
        "nullspace_test_impact": nullspace_test_impact,
        "rcond": None,
    }
    warnings: list[WarningRecord] = []
    if rank < composition_val.shape[1]:
        warnings.append(
            WarningRecord(
                code="harmless_rank_deficiency",
                message=(
                    "Validation composition is rank deficient, but its "
                    "nullspace has no material test-space impact."
                ),
            )
        )
    if (
        condition_number_warning_threshold is not None
        and condition_number > condition_number_warning_threshold
    ):
        warnings.append(
            WarningRecord(
                code="ill_conditioned_composition",
                message=(
                    "The identifiable validation composition subspace "
                    "exceeds the configured condition-number threshold."
                ),
            )
        )
    return diagnostics, tuple(warnings)


def _diagnostics(
    *,
    composition_val: np.ndarray,
    reference_val: np.ndarray,
    raw_val_member: np.ndarray,
    delta_e0: np.ndarray,
    identifiability: dict[str, Any],
    identifiability_warnings: tuple[WarningRecord, ...],
    residual_rmse_warning_threshold: float | None,
) -> tuple[dict[str, Any], tuple[WarningRecord, ...]]:
    with np.errstate(over="ignore", invalid="ignore"):
        baseline_residual = reference_val - raw_val_member
        residual = reference_val - (
            raw_val_member + composition_val @ delta_e0
        )
    if not np.all(np.isfinite(baseline_residual)) or not np.all(
        np.isfinite(residual)
    ):
        _fail("calibration", "produces nonfinite residuals")

    with np.errstate(over="ignore", invalid="ignore"):
        baseline_rmse = float(
            np.sqrt(np.mean(np.square(baseline_residual)))
        )
        residual_rmse = float(np.sqrt(np.mean(np.square(residual))))
        residual_max_abs = float(np.max(np.abs(residual)))
    if not all(
        math.isfinite(value)
        for value in (baseline_rmse, residual_rmse, residual_max_abs)
    ):
        _fail("calibration", "produces nonfinite residual diagnostics")

    diagnostics = dict(identifiability)
    diagnostics.update(
        {
            "residual_rmse": residual_rmse,
            "residual_max_abs": residual_max_abs,
        }
    )
    warnings = list(identifiability_warnings)
    if (
        residual_rmse_warning_threshold is not None
        and residual_rmse > residual_rmse_warning_threshold
    ):
        warnings.append(
            WarningRecord(
                code="large_calibration_residual",
                message=(
                    "Calibration residual RMSE exceeds the configured "
                    "warning threshold."
                ),
            )
        )
    if not residual_rmse < baseline_rmse:
        warnings.append(
            WarningRecord(
                code="weak_rmse_improvement",
                message=(
                    "Calibration does not strictly improve validation RMSE."
                ),
            )
        )
    return diagnostics, tuple(warnings)


def direct_test_correction(
    *,
    raw_total: Any,
    reference_total: Any,
    atomization_energy: Any,
    composition: Any,
    model_e0: Any,
    composition_atomic_numbers: Any,
    e0_atomic_numbers: Any,
) -> np.ndarray:
    raw = _float_array(raw_total, "raw_total", ndim=2)
    reference = _energy_vector(reference_total, "reference_total")
    atomization = _energy_vector(
        atomization_energy, "atomization_energy"
    )
    counts = _composition(composition, "composition")
    e0 = _float_array(model_e0, "model_e0", ndim=2)
    _aligned_atomic_numbers(
        composition_atomic_numbers,
        e0_atomic_numbers,
        left_path="composition_atomic_numbers",
        right_path="e0_atomic_numbers",
        left_columns=counts.shape[1],
        right_columns=e0.shape[1],
    )
    structure_count = raw.shape[1]
    if (
        reference.shape[0] != structure_count
        or atomization.shape[0] != structure_count
        or counts.shape[0] != structure_count
    ):
        _fail("direct_test_correction", "structure dimensions must align")
    if e0.shape[0] != raw.shape[0]:
        _fail("model_e0", "member dimension must match raw_total")

    with np.errstate(over="ignore", invalid="ignore"):
        mad_baseline = reference - atomization
        model_baseline = counts @ e0.T
        corrected = raw - model_baseline.T + mad_baseline
    if not np.all(np.isfinite(corrected)):
        _fail("direct_test_correction", "produces nonfinite corrected energies")
    return np.array(corrected, dtype=np.float64, order="C", copy=True)


def test_space_identifiability(
    *,
    composition_val: Any,
    composition_test: Any,
    val_atomic_numbers: Any,
    test_atomic_numbers: Any,
    condition_number_warning_threshold: Any = None,
) -> tuple[dict[str, Any], tuple[WarningRecord, ...]]:
    threshold = _warning_threshold(
        condition_number_warning_threshold,
        "condition_number_warning_threshold",
    )
    val, test, _ = _composition_pair(
        composition_val=composition_val,
        composition_test=composition_test,
        val_atomic_numbers=val_atomic_numbers,
        test_atomic_numbers=test_atomic_numbers,
    )
    return _identifiability(val, test, threshold)


def correction_diagnostics(
    *,
    composition_val: Any,
    reference_val: Any,
    raw_val_member: Any,
    delta_e0: Any,
    composition_test: Any,
    val_atomic_numbers: Any,
    test_atomic_numbers: Any,
    condition_number_warning_threshold: Any = None,
    residual_rmse_warning_threshold: Any = None,
) -> tuple[dict[str, Any], tuple[WarningRecord, ...]]:
    condition_threshold = _warning_threshold(
        condition_number_warning_threshold,
        "condition_number_warning_threshold",
    )
    residual_threshold = _warning_threshold(
        residual_rmse_warning_threshold,
        "residual_rmse_warning_threshold",
    )
    val, test, _ = _composition_pair(
        composition_val=composition_val,
        composition_test=composition_test,
        val_atomic_numbers=val_atomic_numbers,
        test_atomic_numbers=test_atomic_numbers,
    )
    reference = _energy_vector(reference_val, "reference_val")
    raw = _energy_vector(raw_val_member, "raw_val_member")
    delta = _energy_vector(delta_e0, "delta_e0")
    if reference.shape[0] != val.shape[0] or raw.shape[0] != val.shape[0]:
        _fail("correction_diagnostics", "validation rows must align")
    if delta.shape[0] != val.shape[1]:
        _fail("delta_e0", "must align with validation element columns")

    identifiability, warnings = _identifiability(
        val, test, condition_threshold
    )
    return _diagnostics(
        composition_val=val,
        reference_val=reference,
        raw_val_member=raw,
        delta_e0=delta,
        identifiability=identifiability,
        identifiability_warnings=warnings,
        residual_rmse_warning_threshold=residual_threshold,
    )


def fit_member_delta_e0(
    *,
    composition_val: Any,
    reference_val: Any,
    raw_val_member: Any,
    composition_test: Any,
    member_id: Any,
    val_atomic_numbers: Any,
    test_atomic_numbers: Any,
    condition_number_warning_threshold: Any = None,
    residual_rmse_warning_threshold: Any = None,
) -> tuple[CalibrationFit, dict[str, Any], tuple[WarningRecord, ...]]:
    condition_threshold = _warning_threshold(
        condition_number_warning_threshold,
        "condition_number_warning_threshold",
    )
    residual_threshold = _warning_threshold(
        residual_rmse_warning_threshold,
        "residual_rmse_warning_threshold",
    )
    identifier = _member_id(member_id)
    val, test, atomic_numbers = _composition_pair(
        composition_val=composition_val,
        composition_test=composition_test,
        val_atomic_numbers=val_atomic_numbers,
        test_atomic_numbers=test_atomic_numbers,
    )
    reference = _energy_vector(reference_val, "reference_val")
    raw = _energy_vector(raw_val_member, "raw_val_member")
    if reference.shape[0] != val.shape[0] or raw.shape[0] != val.shape[0]:
        _fail("fit_member_delta_e0", "validation rows must align")

    identifiability, identifiability_warnings = _identifiability(
        val, test, condition_threshold
    )
    with np.errstate(over="ignore", invalid="ignore"):
        target = reference - raw
    if not np.all(np.isfinite(target)):
        _fail("fit_member_delta_e0", "produces a nonfinite fit target")
    try:
        delta, _, _, _ = np.linalg.lstsq(val, target, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise HardFailure("fit_member_delta_e0: least squares failed") from exc
    delta = np.array(delta, dtype=np.float64, order="C", copy=True)
    if delta.shape != (val.shape[1],) or not np.all(np.isfinite(delta)):
        _fail("fit_member_delta_e0", "produces an invalid E0 offset")

    diagnostics, warnings = _diagnostics(
        composition_val=val,
        reference_val=reference,
        raw_val_member=raw,
        delta_e0=delta,
        identifiability=identifiability,
        identifiability_warnings=identifiability_warnings,
        residual_rmse_warning_threshold=residual_threshold,
    )
    fit = CalibrationFit(
        member_id=identifier,
        atomic_numbers=atomic_numbers,
        delta_e0=tuple(float(value) for value in delta),
        rank=int(diagnostics["rank"]),
        residual_rmse=float(diagnostics["residual_rmse"]),
        condition_number=float(diagnostics["condition_number"]),
    )
    return fit, diagnostics, warnings


def apply_member_delta_e0(
    *,
    raw_test_member: Any,
    composition_test: Any,
    delta_e0: Any,
    composition_atomic_numbers: Any,
    delta_atomic_numbers: Any,
) -> np.ndarray:
    raw = _energy_vector(raw_test_member, "raw_test_member")
    counts = _composition(composition_test, "composition_test")
    delta = _energy_vector(delta_e0, "delta_e0")
    _aligned_atomic_numbers(
        composition_atomic_numbers,
        delta_atomic_numbers,
        left_path="composition_atomic_numbers",
        right_path="delta_atomic_numbers",
        left_columns=counts.shape[1],
        right_columns=delta.shape[0],
    )
    if raw.shape[0] != counts.shape[0]:
        _fail("apply_member_delta_e0", "test structure dimensions must align")

    with np.errstate(over="ignore", invalid="ignore"):
        corrected = raw + counts @ delta
    if not np.all(np.isfinite(corrected)):
        _fail("apply_member_delta_e0", "produces nonfinite corrected energies")
    return np.array(corrected, dtype=np.float64, order="C", copy=True)
