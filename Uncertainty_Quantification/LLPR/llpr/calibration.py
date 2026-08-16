"""Deterministic six-path LLPR calibration with crash-safe resume."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Mapping

from ase.data import chemical_symbols
import torch
from torch import Tensor

from .artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    atomic_json_dump,
    atomic_torch_save,
    canonical_json,
    load_torch_artifact,
    require_identity,
    sha256_file,
)
from .calibration_policy import (
    ZERO_Q_POLICY,
    ForceCalibrationDecision,
    classify_force_calibration_structure,
)
from .checkpoint import CheckpointIdentity, load_checkpoint
from .config import LLPRConfig
from .curvature_source import load_curvature_source
from .curvature import run_root
from .data import DatasetHandle, build_dataset, iter_samples
from .observables import compute_structure_jacobians
from .readout import discover_readout_layout
from .ridge import RidgeRecord, condition_number_ridge


_VARIANTS = ("he", "hf", "hef")
_TARGETS = ("energy", "forces")


class CholeskyQuadraticForm:
    """Evaluate rows of ``g (H + ridge I)^-1 g.T`` via float64 Cholesky."""

    def __init__(self, curvature: Tensor, ridge: float) -> None:
        matrix = curvature.detach().to(device="cpu", dtype=torch.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("curvature must be a square matrix")
        if not torch.isfinite(matrix).all():
            raise ValueError("curvature must be finite")
        ridge_value = float(ridge)
        if not math.isfinite(ridge_value) or ridge_value < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        regularized = matrix + ridge_value * torch.eye(
            matrix.shape[0], dtype=torch.float64, device="cpu"
        )
        self.factor = torch.linalg.cholesky(regularized)
        self.size = matrix.shape[0]
        self.ridge = ridge_value

    def q(self, gradients: Tensor) -> Tensor:
        """Return one quadratic form per gradient row."""
        rows = gradients.detach().to(device="cpu", dtype=torch.float64)
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        if rows.ndim != 2 or rows.shape[1] != self.size:
            raise ValueError(
                f"gradients must have shape (rows, {self.size})"
            )
        if not torch.isfinite(rows).all():
            raise ValueError("gradients must be finite")
        transformed = torch.linalg.solve_triangular(
            self.factor, rows.T, upper=False
        )
        return torch.sum(transformed.square(), dim=0)


@dataclass(frozen=True)
class CalibrationRecord:
    variant: Literal["he", "hf", "hef"]
    target: Literal["energy", "forces"]
    ridge_mode: Literal["fixed", "condition_number"]
    ridge: float
    alpha: float
    rows: int
    mean_residual_squared_over_q: float


def _squared_residual_over_q(
    residuals: Tensor,
    q: Tensor,
    min_q: float,
) -> Tensor:
    residual_values = residuals.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    q_values = q.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if residual_values.numel() == 0:
        raise ValueError("residuals and q must not be empty")
    if residual_values.shape != q_values.shape:
        raise ValueError("residuals and q must have the same number of rows")
    if not torch.isfinite(residual_values).all():
        raise ValueError("residuals must be finite")
    if not torch.isfinite(q_values).all() or torch.any(q_values <= 0.0):
        raise ValueError("q must contain only finite positive values")
    floor = float(min_q)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("min_q must be finite and positive")
    return residual_values.square() / torch.clamp(q_values, min=floor)


def calibrate_alpha(
    residuals: Tensor,
    q: Tensor,
    min_q: float = 1.0e-30,
) -> float:
    """Return ``sqrt(mean(residual**2 / max(q, min_q)))``."""
    ratios = _squared_residual_over_q(residuals, q, min_q)
    return math.sqrt(float(ratios.mean()))


def _checkpoint_metadata(identity: CheckpointIdentity) -> dict[str, Any]:
    return {
        "sha256": identity.sha256,
        "model_class": identity.model_class,
        "heads": list(identity.heads),
        "selected_head": identity.selected_head,
        "r_max": identity.r_max,
        "atomic_numbers": list(identity.atomic_numbers),
        "dtype": str(identity.dtype),
    }


def _dataset_metadata(dataset: DatasetHandle) -> dict[str, Any]:
    return {
        "sha256": dataset.sha256,
        "identity": dataset.identity,
        "size": dataset.size,
        "atomic_numbers": list(dataset.atomic_numbers),
        "r_max": dataset.r_max,
        "head": dataset.head,
    }


def _ridge_identity(config: LLPRConfig, records: Mapping[str, RidgeRecord]) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "mode": config.ridge.mode,
        "selected": {variant: records[variant].value for variant in _VARIANTS},
    }
    if config.ridge.mode == "fixed":
        identity["value"] = config.ridge.value
    else:
        identity["max_condition_number"] = config.ridge.max_condition_number
    return identity


def _calibration_identity(
    config: LLPRConfig,
    checkpoint: CheckpointIdentity,
    dataset: DatasetHandle,
    curvature_identity: Mapping[str, Any],
    ridges: Mapping[str, RidgeRecord],
    curvature_artifact: Mapping[str, str],
) -> dict[str, Any]:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "zero_q_policy": ZERO_Q_POLICY,
        "checkpoint": _checkpoint_metadata(checkpoint),
        "curvature": dict(curvature_identity),
        "calibration": _dataset_metadata(dataset),
        "ridge": _ridge_identity(config, ridges),
        "curvature_artifact": dict(curvature_artifact),
        "min_q": config.curvature.min_q,
        "limits": {
            "max_structures": config.runtime.max_structures,
            "max_force_components_per_structure": (
                config.runtime.max_force_components_per_structure
            ),
        },
    }
    if config.runtime.has_explicit_consumer_limits:
        identity["consumer_limits"] = {
            "max_structures": config.runtime.effective_consumer_max_structures,
            "max_force_components_per_structure": (
                config.runtime.effective_consumer_max_force_components_per_structure
            ),
        }
    return identity




def _empty_accumulators() -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        variant: {
            target: {"sum": 0.0, "rows": 0}
            for target in _TARGETS
        }
        for variant in _VARIANTS
    }


def _empty_counts() -> dict[str, int]:
    return {
        "energy_structures": 0,
        "force_used_structures": 0,
        "force_excluded_structures": 0,
        "force_components_total": 0,
        "force_components_used": 0,
        "force_components_excluded": 0,
    }


def _new_progress(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": dict(identity),
        "status": "in_progress",
        "next_index": 0,
        "structures": 0,
        "counts": _empty_counts(),
        "exclusions": [],
        "accumulators": _empty_accumulators(),
    }


def _validate_progress(progress: Mapping[str, Any]) -> None:
    if progress.get("status") not in ("in_progress", "complete"):
        raise ValueError("calibration progress has an invalid status")
    for field in ("next_index", "structures"):
        value = progress.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"calibration progress {field} must be non-negative")
    accumulators = progress.get("accumulators")
    if not isinstance(accumulators, Mapping) or set(accumulators) != set(_VARIANTS):
        raise ValueError("calibration progress has invalid variants")
    for variant in _VARIANTS:
        targets = accumulators[variant]
        if not isinstance(targets, Mapping) or set(targets) != set(_TARGETS):
            raise ValueError("calibration progress has invalid targets")
        for target in _TARGETS:
            accumulator = targets[target]
            if not isinstance(accumulator, Mapping):
                raise ValueError("calibration progress accumulator must be a mapping")
            total = accumulator.get("sum")
            rows = accumulator.get("rows")
            if (
                isinstance(total, bool)
                or not isinstance(total, (float, int))
                or not math.isfinite(float(total))
                or float(total) < 0.0
            ):
                raise ValueError("calibration progress sum must be finite and non-negative")
            if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
                raise ValueError("calibration progress rows must be non-negative")
    _validate_progress_population(progress)


def _matching_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return canonical_json(actual) == canonical_json(expected)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("complete calibration diagnostics are invalid strict JSON") from error


def _finite_number(value: Any, field: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"complete calibration diagnostics {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (non_negative and result < 0.0):
        raise ValueError(f"complete calibration diagnostics {field} must be finite")
    return result


def _close_float(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-15)


def _validate_exclusion_entry(
    entry: Any,
    *,
    next_index: int,
) -> tuple[int, int]:
    fields = {
        "calibration_index",
        "structure_id",
        "num_atoms",
        "elements",
        "force_components_total",
        "zero_components",
    }
    if not isinstance(entry, Mapping) or set(entry) != fields:
        raise ValueError("calibration progress exclusion schema mismatch")
    calibration_index = entry["calibration_index"]
    if (
        isinstance(calibration_index, bool)
        or not isinstance(calibration_index, int)
        or calibration_index < 0
        or calibration_index >= next_index
    ):
        raise ValueError("calibration progress exclusion index is invalid")
    if not isinstance(entry["structure_id"], str) or not entry["structure_id"]:
        raise ValueError("calibration progress exclusion structure id is invalid")
    num_atoms = entry["num_atoms"]
    if isinstance(num_atoms, bool) or not isinstance(num_atoms, int) or num_atoms <= 0:
        raise ValueError("calibration progress exclusion num_atoms is invalid")
    elements = entry["elements"]
    if (
        not isinstance(elements, list)
        or not elements
        or any(not isinstance(element, str) or not element for element in elements)
        or elements != sorted(set(elements))
    ):
        raise ValueError("calibration progress exclusion elements are invalid")
    component_total = entry["force_components_total"]
    if (
        isinstance(component_total, bool)
        or not isinstance(component_total, int)
        or component_total <= 0
    ):
        raise ValueError("calibration progress exclusion component count is invalid")
    zero_components = entry["zero_components"]
    if not isinstance(zero_components, list) or not zero_components:
        raise ValueError("calibration progress exclusion zero components are invalid")

    component_fields = {
        "flat_index",
        "atom_index",
        "direction",
        "reference",
        "prediction",
        "residual",
        "q",
    }
    flat_indices: list[int] = []
    for component in zero_components:
        if not isinstance(component, Mapping) or set(component) != component_fields:
            raise ValueError("calibration progress exclusion component schema mismatch")
        flat_index = component["flat_index"]
        atom_index = component["atom_index"]
        direction = component["direction"]
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (flat_index, atom_index, direction)
            )
            or flat_index < 0
            or atom_index < 0
            or atom_index >= num_atoms
            or direction not in (0, 1, 2)
            or divmod(flat_index, 3) != (atom_index, direction)
        ):
            raise ValueError("calibration progress exclusion component index is invalid")
        flat_indices.append(flat_index)
        reference = _finite_number(component["reference"], "reference")
        prediction = _finite_number(component["prediction"], "prediction")
        residual = _finite_number(component["residual"], "residual")
        if not _close_float(residual, reference - prediction):
            raise ValueError("calibration progress exclusion residual is inconsistent")
        q_values = component["q"]
        if not isinstance(q_values, Mapping) or set(q_values) != set(_VARIANTS):
            raise ValueError("calibration progress exclusion q variants mismatch")
        if any(
            isinstance(q_values[variant], bool)
            or not isinstance(q_values[variant], (int, float))
            or float(q_values[variant]) != 0.0
            for variant in _VARIANTS
        ):
            raise ValueError("calibration progress exclusion q must be exact zero")
    if (
        flat_indices != sorted(flat_indices)
        or len(set(flat_indices)) != len(flat_indices)
        or len(flat_indices) > component_total
    ):
        raise ValueError("calibration progress exclusion components are duplicated")
    return calibration_index, component_total


def _validate_progress_population(progress: Mapping[str, Any]) -> None:
    counts = progress.get("counts")
    expected_count_fields = set(_empty_counts())
    if not isinstance(counts, Mapping) or set(counts) != expected_count_fields:
        raise ValueError("calibration progress counts schema mismatch")
    for field in expected_count_fields:
        value = counts[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"calibration progress count {field} is invalid")

    structures = progress["structures"]
    next_index = progress["next_index"]
    if next_index != structures:
        raise ValueError("calibration progress next_index and structures mismatch")
    if counts["energy_structures"] != structures:
        raise ValueError("calibration progress energy structure count mismatch")
    if (
        counts["force_used_structures"] + counts["force_excluded_structures"]
        != structures
    ):
        raise ValueError("calibration progress force structure count mismatch")
    if (
        counts["force_components_used"] + counts["force_components_excluded"]
        != counts["force_components_total"]
    ):
        raise ValueError("calibration progress force component count mismatch")

    accumulators = progress["accumulators"]
    energy_rows = {
        accumulators[variant]["energy"]["rows"] for variant in _VARIANTS
    }
    force_rows = {
        accumulators[variant]["forces"]["rows"] for variant in _VARIANTS
    }
    if energy_rows != {counts["energy_structures"]}:
        raise ValueError("calibration progress energy row count mismatch")
    if force_rows != {counts["force_components_used"]}:
        raise ValueError("calibration progress force row count mismatch")

    exclusions = progress.get("exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("calibration progress exclusions must be a list")
    indices: list[int] = []
    excluded_components = 0
    for entry in exclusions:
        calibration_index, component_total = _validate_exclusion_entry(
            entry, next_index=next_index
        )
        indices.append(calibration_index)
        excluded_components += component_total
    if len(set(indices)) != len(indices) or indices != sorted(indices):
        raise ValueError("calibration progress has duplicate exclusion entries")
    if len(exclusions) != counts["force_excluded_structures"]:
        raise ValueError("calibration progress exclusion count mismatch")
    if excluded_components != counts["force_components_excluded"]:
        raise ValueError("calibration progress excluded component count mismatch")


def _complete_identity(
    identity: Mapping[str, Any],
    counts: Mapping[str, int],
    audit_sha256: str,
) -> dict[str, Any]:
    result = dict(identity)
    result["calibration_population"] = dict(counts)
    result["force_exclusions"] = {
        "path": "force_exclusions.json",
        "sha256": audit_sha256,
    }
    return result


def _validate_complete_row_counts(
    progress: Mapping[str, Any], target_structures: int
) -> None:
    structures = progress["structures"]
    if structures != target_structures or structures <= 0:
        raise ValueError("complete calibration row count disagrees with structures")
    accumulators = progress["accumulators"]
    energy_rows = [accumulators[variant]["energy"]["rows"] for variant in _VARIANTS]
    if any(rows != structures for rows in energy_rows):
        raise ValueError("complete calibration energy row count must equal structures")
    force_rows = [accumulators[variant]["forces"]["rows"] for variant in _VARIANTS]
    if any(rows <= 0 for rows in force_rows) or len(set(force_rows)) != 1:
        raise ValueError("complete calibration force row count mismatch across variants")


def _validate_record_keys(records: Any) -> None:
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("complete calibration artifact records must contain six rows")
    keys: list[tuple[Any, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("complete calibration artifact record must be a mapping")
        key = (record.get("variant"), record.get("target"))
        keys.append(key)
        rows = record.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError("complete calibration record row count must be positive")
        for field in ("ridge", "alpha", "mean_residual_squared_over_q"):
            value = record.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"complete calibration record {field} is invalid")
    expected = {(variant, target) for variant in _VARIANTS for target in _TARGETS}
    if len(set(keys)) != 6 or set(keys) != expected:
        raise ValueError("complete calibration record keys must be unique")


def _validate_ridge_diagnostics(
    diagnostics: Any,
    identity: Mapping[str, Any],
    ridges: Mapping[str, RidgeRecord],
    spectra: Mapping[str, tuple[float, float]],
) -> None:
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {
        "identity",
        "status",
        "variants",
    }:
        raise ValueError("complete calibration diagnostics schema mismatch")
    diagnostics_identity = diagnostics.get("identity")
    if not isinstance(diagnostics_identity, Mapping):
        raise ValueError("complete calibration diagnostics identity mismatch")
    require_identity(diagnostics_identity, identity)
    if diagnostics.get("status") != "complete":
        raise ValueError("complete calibration diagnostics status mismatch")
    source_variants = diagnostics.get("variants")
    if not isinstance(source_variants, Mapping) or set(source_variants) != set(_VARIANTS):
        raise ValueError("complete calibration diagnostics variants mismatch")

    fields = {
        "ridge_mode",
        "ridge",
        "eigenvalue_min",
        "eigenvalue_max",
        "regularized_condition_number",
    }
    for variant in _VARIANTS:
        source = source_variants[variant]
        if not isinstance(source, Mapping) or set(source) != fields:
            raise ValueError("complete calibration diagnostics variant schema mismatch")
        ridge = ridges[variant]
        if source.get("ridge_mode") != ridge.mode:
            raise ValueError("complete calibration diagnostics ridge mode mismatch")
        stored_ridge = _finite_number(source.get("ridge"), "ridge", non_negative=True)
        if not _close_float(stored_ridge, ridge.value):
            raise ValueError("complete calibration diagnostics ridge mismatch")
        minimum = _finite_number(source.get("eigenvalue_min"), "eigenvalue_min")
        maximum = _finite_number(source.get("eigenvalue_max"), "eigenvalue_max")
        if maximum < minimum and not _close_float(maximum, minimum):
            raise ValueError("complete calibration diagnostics eigenvalue range mismatch")
        if variant in spectra:
            expected_minimum, expected_maximum = spectra[variant]
            if not _close_float(minimum, expected_minimum) or not _close_float(
                maximum, expected_maximum
            ):
                raise ValueError("complete calibration diagnostics eigenvalues mismatch")
        regularized_minimum = minimum + stored_ridge
        regularized_maximum = maximum + stored_ridge
        expected_condition = (
            regularized_maximum / regularized_minimum
            if regularized_minimum > 0.0
            else None
        )
        condition = source.get("regularized_condition_number")
        if expected_condition is None:
            if condition is not None:
                raise ValueError("complete calibration diagnostics condition mismatch")
        else:
            stored_condition = _finite_number(
                condition, "regularized_condition_number", non_negative=True
            )
            if not _close_float(stored_condition, expected_condition):
                raise ValueError("complete calibration diagnostics condition mismatch")


def _force_exclusion_audit(
    identity: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "zero_q_policy": ZERO_Q_POLICY,
        "identity": dict(identity),
        "status": "complete",
        "counts": dict(progress["counts"]),
        "exclusions": list(progress["exclusions"]),
    }


def _validate_force_exclusion_audit(
    audit_path: Path,
    identity: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    audit = _load_strict_json(audit_path)
    fields = {
        "schema_version",
        "zero_q_policy",
        "identity",
        "status",
        "counts",
        "exclusions",
    }
    if not isinstance(audit, Mapping) or set(audit) != fields:
        raise ValueError("complete calibration force exclusion audit schema mismatch")
    if audit.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("complete calibration force exclusion audit schema mismatch")
    if audit.get("zero_q_policy") != ZERO_Q_POLICY:
        raise ValueError("complete calibration force exclusion audit policy mismatch")
    audit_identity = audit.get("identity")
    if not isinstance(audit_identity, Mapping):
        raise ValueError("complete calibration force exclusion audit identity mismatch")
    require_identity(audit_identity, identity)
    if audit.get("status") != "complete":
        raise ValueError("complete calibration force exclusion audit status mismatch")
    if canonical_json(audit.get("counts")) != canonical_json(progress["counts"]):
        raise ValueError("complete calibration force exclusion audit counts mismatch")
    if canonical_json(audit.get("exclusions")) != canonical_json(
        progress["exclusions"]
    ):
        raise ValueError("complete calibration force exclusion audit entries mismatch")
    return _complete_identity(identity, progress["counts"], sha256_file(audit_path))


def _validate_complete_artifacts(
    artifact_path: Path,
    csv_path: Path,
    diagnostics_path: Path,
    audit_path: Path,
    identity: Mapping[str, Any],
    progress: Mapping[str, Any],
    ridges: Mapping[str, RidgeRecord],
    spectra: Mapping[str, tuple[float, float]],
    target_structures: int,
) -> None:
    for path in (artifact_path, csv_path, diagnostics_path):
        if not path.is_file():
            raise ValueError(f"complete calibration cache is missing {path.name}")
    if not audit_path.is_file():
        raise ValueError("complete calibration cache is missing force_exclusions.json")
    complete_identity = _validate_force_exclusion_audit(
        audit_path, identity, progress
    )
    if (
        progress.get("status") != "complete"
        or progress.get("next_index") != target_structures
        or progress.get("structures") != target_structures
    ):
        raise ValueError("complete calibration progress count mismatch")
    _validate_complete_row_counts(progress, target_structures)

    expected_records = _records_from_progress(progress, ridges)
    expected_rows = [asdict(record) for record in expected_records]
    artifact = load_torch_artifact(artifact_path)
    if not isinstance(artifact, Mapping):
        raise ValueError("complete calibration artifact must be a mapping")
    require_identity(artifact.get("identity", {}), complete_identity)
    if artifact.get("status") != "complete":
        raise ValueError("complete calibration artifact has an invalid status")
    _validate_record_keys(artifact.get("records"))
    if canonical_json(artifact.get("records")) != canonical_json(expected_rows):
        raise ValueError("complete calibration artifact records mismatch")

    fields = list(CalibrationRecord.__dataclass_fields__)
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fields:
                raise ValueError("complete calibration CSV schema mismatch")
            source_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError("complete calibration CSV is invalid") from error
    csv_rows: list[dict[str, Any]] = []
    for row in source_rows:
        if None in row or any(row.get(field) is None for field in fields):
            raise ValueError("complete calibration CSV row width mismatch")
        try:
            csv_rows.append(
                asdict(
                    CalibrationRecord(
                        variant=row["variant"],
                        target=row["target"],
                        ridge_mode=row["ridge_mode"],
                        ridge=float(row["ridge"]),
                        alpha=float(row["alpha"]),
                        rows=int(row["rows"]),
                        mean_residual_squared_over_q=float(
                            row["mean_residual_squared_over_q"]
                        ),
                    )
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError("complete calibration CSV value is invalid") from error
    if canonical_json(csv_rows) != canonical_json(expected_rows):
        raise ValueError("complete calibration CSV records mismatch")

    diagnostics = _load_strict_json(diagnostics_path)
    _validate_ridge_diagnostics(diagnostics, complete_identity, ridges, spectra)


def _atomic_csv_dump(path: Path, records: list[CalibrationRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CalibrationRecord.__dataclass_fields__)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _compute_spectra(
    variants: Mapping[str, Tensor],
) -> dict[str, tuple[float, float]]:
    spectra: dict[str, tuple[float, float]] = {}
    for variant in _VARIANTS:
        eigenvalues = torch.linalg.eigvalsh(variants[variant])
        spectra[variant] = (float(eigenvalues.min()), float(eigenvalues.max()))
    return spectra


def _select_ridges(
    config: LLPRConfig,
    variants: Mapping[str, Tensor],
) -> tuple[dict[str, RidgeRecord], dict[str, tuple[float, float]]]:
    if config.ridge.mode == "fixed":
        return (
            {
                variant: RidgeRecord(mode="fixed", value=float(config.ridge.value))
                for variant in _VARIANTS
            },
            {},
        )
    if config.ridge.mode != "condition_number":
        raise ValueError(f"unsupported ridge mode: {config.ridge.mode}")
    spectra = _compute_spectra(variants)
    ridges = {
        variant: RidgeRecord(
            mode="condition_number",
            value=condition_number_ridge(
                torch.tensor(spectra[variant], dtype=torch.float64),
                config.ridge.max_condition_number,
            ),
        )
        for variant in _VARIANTS
    }
    return ridges, spectra


def _ridge_diagnostics(
    identity: Mapping[str, Any],
    spectra: Mapping[str, tuple[float, float]],
    ridges: Mapping[str, RidgeRecord],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for variant in _VARIANTS:
        minimum, maximum = spectra[variant]
        regularized_minimum = minimum + ridges[variant].value
        regularized_maximum = maximum + ridges[variant].value
        condition_number = (
            regularized_maximum / regularized_minimum
            if regularized_minimum > 0.0
            else None
        )
        diagnostics[variant] = {
            "ridge_mode": ridges[variant].mode,
            "ridge": ridges[variant].value,
            "eigenvalue_min": minimum,
            "eigenvalue_max": maximum,
            "regularized_condition_number": condition_number,
        }
    return {
        "identity": dict(identity),
        "status": "complete",
        "variants": diagnostics,
    }


def _records_from_progress(
    progress: Mapping[str, Any],
    ridges: Mapping[str, RidgeRecord],
) -> list[CalibrationRecord]:
    records = []
    accumulators = progress["accumulators"]
    for variant in _VARIANTS:
        for target in _TARGETS:
            accumulator = accumulators[variant][target]
            rows = int(accumulator["rows"])
            if rows == 0:
                raise ValueError(f"calibration {variant}/{target} has no rows")
            mean = float(accumulator["sum"]) / rows
            records.append(
                CalibrationRecord(
                    variant=variant,
                    target=target,
                    ridge_mode=ridges[variant].mode,
                    ridge=ridges[variant].value,
                    alpha=math.sqrt(mean),
                    rows=rows,
                    mean_residual_squared_over_q=mean,
                )
            )
    return records


def _sample_elements(batch: Any, atomic_numbers: tuple[int, ...]) -> list[str]:
    node_attrs = getattr(batch, "node_attrs", None)
    if (
        isinstance(node_attrs, Tensor)
        and node_attrs.ndim == 2
        and node_attrs.shape[1] == len(atomic_numbers)
        and node_attrs.shape[0] > 0
    ):
        columns = torch.nonzero(
            torch.any(node_attrs.detach().to(device="cpu") != 0, dim=0),
            as_tuple=False,
        ).reshape(-1)
        if columns.numel() > 0:
            return sorted(
                {
                    chemical_symbols[atomic_numbers[int(column)]]
                    for column in columns
                }
            )
    if len(atomic_numbers) == 1:
        return [chemical_symbols[atomic_numbers[0]]]
    raise ValueError("cannot derive calibration structure elements")


def _exclusion_record(
    sample: Any,
    indices: Tensor,
    reference_forces: Tensor,
    predicted_forces: Tensor,
    force_residuals: Tensor,
    force_q_by_variant: Mapping[str, Tensor],
    decision: ForceCalibrationDecision,
    atomic_numbers: tuple[int, ...],
) -> dict[str, Any]:
    if not decision.exclude_structure or not decision.zero_rows:
        raise ValueError("force exclusion record requires exact-zero rows")
    components = []
    for row in decision.zero_rows:
        flat_index = int(indices[row])
        atom_index, direction = divmod(flat_index, 3)
        components.append(
            {
                "flat_index": flat_index,
                "atom_index": atom_index,
                "direction": direction,
                "reference": float(reference_forces[flat_index]),
                "prediction": float(predicted_forces[flat_index]),
                "residual": float(force_residuals[row]),
                "q": {
                    variant: float(force_q_by_variant[variant][row])
                    for variant in _VARIANTS
                },
            }
        )
    components.sort(key=lambda component: component["flat_index"])
    return {
        "calibration_index": int(sample.index),
        "structure_id": str(sample.structure_id),
        "num_atoms": int(sample.num_atoms),
        "elements": _sample_elements(sample.batch, atomic_numbers),
        "force_components_total": int(indices.numel()),
        "zero_components": components,
    }


def run_calibrate(config: LLPRConfig) -> Path:
    """Calibrate energy and force scales for all three curvature variants."""
    loaded = load_checkpoint(
        config.checkpoint,
        torch.device("cpu"),
        selected_head=config.selected_head,
        expected_readout_size=config.expected_readout_size,
    )
    layout = discover_readout_layout(loaded.model)
    dataset = build_dataset(
        config.calibration.path,
        config.calibration.expected_sha256,
        loaded.identity.atomic_numbers,
        loaded.identity.r_max,
        loaded.identity.selected_head,
    )
    curvature = load_curvature_source(config, loaded.identity, layout)
    ridges, spectra = _select_ridges(config, curvature.variants)
    identity = _calibration_identity(
        config,
        loaded.identity,
        dataset,
        curvature.identity,
        ridges,
        {"path": str(curvature.path.resolve()), "sha256": curvature.sha256},
    )

    calibration_dir = (
        run_root(config, loaded.identity.sha256)
        / "calibration"
        / "deterministic"
    )
    progress_path = calibration_dir / "progress.pt"
    artifact_path = calibration_dir / "calibrations.pt"
    csv_path = calibration_dir / "calibrations.csv"
    diagnostics_path = calibration_dir / "ridge_diagnostics.json"
    audit_path = calibration_dir / "force_exclusions.json"
    target_structures = dataset.size
    consumer_max_structures = config.runtime.effective_consumer_max_structures
    if consumer_max_structures is not None:
        target_structures = min(target_structures, consumer_max_structures)

    internal_artifacts = (artifact_path, csv_path, diagnostics_path, audit_path)
    progress: dict[str, Any] | None = None
    if not progress_path.exists() and any(path.exists() for path in internal_artifacts):
        raise ValueError("calibration artifact exists without progress")
    if progress_path.exists():
        candidate = load_torch_artifact(progress_path)
        if not isinstance(candidate, Mapping):
            raise ValueError("calibration progress must be a mapping")
        candidate_identity = candidate.get("identity", {})
        if not isinstance(candidate_identity, Mapping):
            raise ValueError("calibration progress identity mismatch")
        require_identity(candidate_identity, identity)
        _validate_progress(candidate)
        if candidate.get("status") == "complete":
            _validate_complete_artifacts(
                artifact_path,
                csv_path,
                diagnostics_path,
                audit_path,
                identity,
                candidate,
                ridges,
                spectra,
                target_structures,
            )
            return artifact_path
        if config.runtime.resume:
            progress = dict(candidate)

    solvers = {
        variant: CholeskyQuadraticForm(
            curvature.variants[variant], ridge=ridges[variant].value
        )
        for variant in _VARIANTS
    }
    if progress is None:
        progress = _new_progress(identity)
        atomic_torch_save(progress_path, progress)

    if progress["next_index"] > target_structures:
        raise ValueError("calibration progress next_index exceeds the calibration limit")

    requested_device = torch.device(config.runtime.device)
    loaded.model.to(requested_device)
    remaining = target_structures - progress["next_index"]
    if remaining > 0:
        samples = iter_samples(
            dataset,
            device=requested_device,
            dtype=loaded.identity.dtype,
            start_index=progress["next_index"],
            max_structures=remaining,
        )
        for sample in samples:
            jacobians = compute_structure_jacobians(
                model=loaded.model,
                batch=sample.batch,
                layout=layout,
                force_component_chunk_size=config.runtime.force_component_chunk_size,
                max_force_components=(
                    config.runtime.effective_consumer_max_force_components_per_structure
                ),
            )
            energy_residual = torch.tensor(
                [
                    float(sample.reference_energy_per_atom.detach().cpu())
                    - jacobians.energy_per_atom
                ],
                dtype=torch.float64,
            )
            indices = jacobians.force_indices.detach().to(device="cpu", dtype=torch.long)
            reference_forces = sample.reference_forces.detach().to(
                device="cpu", dtype=torch.float64
            ).reshape(-1)
            predicted_forces = jacobians.forces.detach().to(
                device="cpu", dtype=torch.float64
            ).reshape(-1)
            if (
                indices.ndim != 1
                or jacobians.g_forces.shape[0] != indices.numel()
                or (indices.numel() and int(indices.max()) >= reference_forces.numel())
                or (indices.numel() and int(indices.max()) >= predicted_forces.numel())
            ):
                raise ValueError("force Jacobian rows do not match force indices")
            force_residuals = reference_forces[indices] - predicted_forces[indices]

            energy_q_by_variant = {
                variant: solvers[variant].q(jacobians.g_energy)
                for variant in _VARIANTS
            }
            force_q_by_variant = {
                variant: solvers[variant].q(jacobians.g_forces)
                for variant in _VARIANTS
            }
            energy_ratios_by_variant = {
                variant: _squared_residual_over_q(
                    energy_residual,
                    energy_q_by_variant[variant],
                    config.curvature.min_q,
                )
                for variant in _VARIANTS
            }
            decision = classify_force_calibration_structure(
                jacobians.g_forces,
                force_q_by_variant,
                indices,
            )
            force_ratios_by_variant = (
                {}
                if decision.exclude_structure
                else {
                    variant: _squared_residual_over_q(
                        force_residuals,
                        force_q_by_variant[variant],
                        config.curvature.min_q,
                    )
                    for variant in _VARIANTS
                }
            )
            exclusion = (
                _exclusion_record(
                    sample,
                    indices,
                    reference_forces,
                    predicted_forces,
                    force_residuals,
                    force_q_by_variant,
                    decision,
                    loaded.identity.atomic_numbers,
                )
                if decision.exclude_structure
                else None
            )

            for variant in _VARIANTS:
                energy_ratios = energy_ratios_by_variant[variant]
                energy_accumulator = progress["accumulators"][variant]["energy"]
                energy_accumulator["sum"] += float(energy_ratios.sum())
                energy_accumulator["rows"] += int(energy_ratios.numel())
                if not decision.exclude_structure:
                    force_ratios = force_ratios_by_variant[variant]
                    force_accumulator = progress["accumulators"][variant]["forces"]
                    force_accumulator["sum"] += float(force_ratios.sum())
                    force_accumulator["rows"] += int(force_ratios.numel())

            component_count = int(indices.numel())
            counts = progress["counts"]
            counts["energy_structures"] += 1
            counts["force_components_total"] += component_count
            if decision.exclude_structure:
                counts["force_excluded_structures"] += 1
                counts["force_components_excluded"] += component_count
                progress["exclusions"].append(exclusion)
            else:
                counts["force_used_structures"] += 1
                counts["force_components_used"] += component_count

            progress["next_index"] = sample.index + 1
            progress["structures"] += 1
            if progress["structures"] % config.runtime.save_every_structures == 0:
                atomic_torch_save(progress_path, progress)

    _validate_progress(progress)
    audit = _force_exclusion_audit(identity, progress)
    atomic_json_dump(audit_path, audit)
    complete_identity = _complete_identity(
        identity, progress["counts"], sha256_file(audit_path)
    )
    records = _records_from_progress(progress, ridges)
    artifact = {
        "identity": complete_identity,
        "status": "complete",
        "records": [asdict(record) for record in records],
    }
    atomic_torch_save(artifact_path, artifact)
    _atomic_csv_dump(csv_path, records)
    if not spectra:
        spectra = _compute_spectra(curvature.variants)
    atomic_json_dump(
        diagnostics_path,
        _ridge_diagnostics(complete_identity, spectra, ridges),
    )
    progress["status"] = "complete"
    _validate_progress(progress)
    atomic_torch_save(progress_path, progress)
    return artifact_path
