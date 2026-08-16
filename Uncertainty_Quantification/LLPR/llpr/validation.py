"""Validation and deterministic manifests for LLPR publication results."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Iterator, Mapping, Sequence

import torch

from .artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    sha256_file,
    stable_id,
)
from .calibration_policy import (
    CALIBRATION_POPULATION_FIELDS,
    ZERO_Q_POLICY,
)
from .config import LLPRConfig
from .curvature import run_root


RELATIVE_TOLERANCE = 1.0e-12
ABSOLUTE_TOLERANCE = 1.0e-12

ENERGY_FIELDS = (
    "structure_id",
    "num_atoms",
    "reference",
    "prediction",
    "residual",
    "q",
    "variance",
    "std",
    "variant",
    "target",
)
FORCE_FIELDS = (
    "structure_id",
    "num_atoms",
    "atom_index",
    "direction",
    "reference",
    "prediction",
    "residual",
    "q",
    "variance",
    "std",
    "variant",
    "target",
)
FORCE_STRUCTURE_FIELDS = (
    "structure_id",
    "num_atoms",
    "components",
    "mae",
    "rmse",
    "mean_q",
    "mean_variance",
    "variant",
    "target",
)

_VARIANTS = ("he", "hf", "hef")
_CANONICAL_FILES = (
    "energy.csv",
    "force_components.csv",
    "force_structure.csv",
    "summary.json",
)
_SUMMARY_FIELDS = {
    "schema_version",
    "formula_version",
    "variant",
    "ridge",
    "alpha",
    "counts",
    "energy",
    "forces",
    "force_structure",
    "cholesky_diagnostics",
}
_TARGET_SUMMARY_FIELDS = {
    "rows",
    "mae",
    "rmse",
    "q",
    "variance",
    "std",
    "coverage",
    "standardized_residual",
}
_ZERO_Q_SUMMARY_FIELDS = {
    "zero_q_rows",
    "zero_q_zero_residual_rows",
    "zero_q_nonzero_residual_rows",
}
_PROGRESS_BASE_FIELDS = {
    "identity",
    "status",
    "next_index",
    "structures",
    "csv_offsets",
}
_DISTRIBUTION_FIELDS = {"rows", "mean", "std", "min", "max"}
_CHOLESKY_FIELDS = {
    "ridge_mode",
    "ridge",
    "eigenvalue_min",
    "eigenvalue_max",
    "regularized_condition_number",
}


def _same(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=RELATIVE_TOLERANCE,
        abs_tol=ABSOLUTE_TOLERANCE,
    )


def _scale_aware_same(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 1.0e-30)
    return abs(left - right) <= RELATIVE_TOLERANCE * scale


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains non-finite value {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file {path}") from error


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_strict_json_dump(path: Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_strict_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_bytes_dump(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _require_keys(value: Any, expected: set[str], source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(
            f"{source} schema mismatch: expected={sorted(expected)}, actual={actual}"
        )
    return value


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"publication file is missing: {path}")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(fields):
                raise ValueError(f"CSV schema mismatch for {path}")
            rows = list(reader)
            for row_index, row in enumerate(rows):
                if None in row:
                    raise ValueError(
                        f"{path} row {row_index} contains surplus CSV values"
                    )
                if any(row[field] is None for field in fields):
                    raise ValueError(
                        f"{path} row {row_index} contains missing CSV values"
                    )
            return rows
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"invalid CSV file {path}") from error


def _integer(row: Mapping[str, str], field: str, source: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source} {field} must be an integer") from error
    return value


def _finite(row: Mapping[str, str], field: str, source: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source} {field} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{source} {field} must be finite")
    return value


def _validate_observation(
    row: Mapping[str, str], source: str
) -> tuple[float, float, float, float, float, float]:
    reference = _finite(row, "reference", source)
    prediction = _finite(row, "prediction", source)
    residual = _finite(row, "residual", source)
    q_value = _finite(row, "q", source)
    variance = _finite(row, "variance", source)
    std = _finite(row, "std", source)
    if not _same(residual, reference - prediction):
        raise ValueError(f"{source} residual != reference - prediction")
    if q_value < 0.0:
        raise ValueError(f"{source} q must be non-negative")
    if variance < 0.0:
        raise ValueError(f"{source} variance must be non-negative")
    if std < 0.0:
        raise ValueError(f"{source} std must be non-negative")
    if not _same(std * std, variance):
        raise ValueError(f"{source} std squared does not match variance")
    return reference, prediction, residual, q_value, variance, std


def _validate_energy(
    root: Path, variant: str
) -> tuple[list[tuple[str, int, float, float, float]], list[float], list[float]]:
    source = f"{variant}/energy.csv"
    rows = _read_csv(root / source, ENERGY_FIELDS)
    if not rows:
        raise ValueError(f"{source} must contain at least one structure")
    keys: set[str] = set()
    observations = []
    residuals = []
    std_values = []
    for row_index, row in enumerate(rows):
        row_source = f"{source} row {row_index}"
        if row.get("variant") != variant or row.get("target") != "energy":
            raise ValueError(f"{row_source} variant or target mismatch")
        structure_id = row.get("structure_id", "")
        if not structure_id:
            raise ValueError(f"{row_source} structure_id must be non-empty")
        if structure_id in keys:
            raise ValueError(f"{source} contains duplicate structure_id {structure_id}")
        keys.add(structure_id)
        num_atoms = _integer(row, "num_atoms", row_source)
        if num_atoms <= 0:
            raise ValueError(f"{row_source} num_atoms must be positive")
        reference, prediction, residual, q_value, _, std = _validate_observation(
            row, row_source
        )
        if q_value <= 0.0:
            raise ValueError(f"{row_source} energy q must be positive")
        observations.append(
            (structure_id, num_atoms, reference, prediction, residual)
        )
        residuals.append(residual)
        std_values.append(std)
    return observations, residuals, std_values


def _validate_forces(
    root: Path,
    variant: str,
    energy: Sequence[tuple[str, int, float, float, float]],
    max_force_components_per_structure: int | None = None,
    allow_zero_q: bool = False,
) -> tuple[
    list[tuple[str, int, int, int, float, float, float]],
    list[float],
    list[float],
    list[dict[str, str]],
]:
    source = f"{variant}/force_components.csv"
    rows = _read_csv(root / source, FORCE_FIELDS)
    expected_keys = []
    for structure_id, num_atoms, _, _, _ in energy:
        structure_keys = [
            (structure_id, num_atoms, atom_index, direction)
            for atom_index in range(num_atoms)
            for direction in range(3)
        ]
        expected_keys.extend(structure_keys[:max_force_components_per_structure])
    actual_keys = []
    observations = []
    residuals = []
    std_values = []
    seen: set[tuple[str, int, int, int]] = set()
    for row_index, row in enumerate(rows):
        row_source = f"{source} row {row_index}"
        if row.get("variant") != variant or row.get("target") != "forces":
            raise ValueError(f"{row_source} variant or target mismatch")
        structure_id = row.get("structure_id", "")
        num_atoms = _integer(row, "num_atoms", row_source)
        atom_index = _integer(row, "atom_index", row_source)
        direction = _integer(row, "direction", row_source)
        if num_atoms <= 0:
            raise ValueError(f"{row_source} num_atoms must be positive")
        if direction not in (0, 1, 2):
            raise ValueError(f"{row_source} direction must be 0, 1, or 2")
        if atom_index < 0 or atom_index >= num_atoms:
            raise ValueError(f"{row_source} atom_index must be within num_atoms")
        key = (structure_id, num_atoms, atom_index, direction)
        if key in seen:
            raise ValueError(f"{source} contains duplicate force component key {key}")
        seen.add(key)
        actual_keys.append(key)
        reference, prediction, residual, q_value, variance, std = _validate_observation(
            row, row_source
        )
        if q_value == 0.0:
            if not allow_zero_q:
                raise ValueError(f"{row_source} legacy force q must be positive")
            if variance != 0.0 or std != 0.0:
                raise ValueError(
                    f"{row_source} zero q requires variance and std literal zero"
                )
        observations.append((*key, reference, prediction, residual))
        residuals.append(residual)
        std_values.append(std)
    if actual_keys != expected_keys:
        raise ValueError(
            f"variant alignment mismatch for {variant} force component order; "
            "every atom must have directions 0, 1, and 2 exactly once"
        )
    return observations, residuals, std_values, rows


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"rows": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return {
        "rows": len(values),
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "min": min(values),
        "max": max(values),
    }


def _target_summary(
    rows: Sequence[Mapping[str, str]], *, include_zero_q_counts: bool
) -> dict[str, Any]:
    residuals = [_finite(row, "residual", "summary source") for row in rows]
    q_values = [_finite(row, "q", "summary source") for row in rows]
    variances = [_finite(row, "variance", "summary source") for row in rows]
    std_values = [_finite(row, "std", "summary source") for row in rows]
    standardized = []
    coverage = {1: 0, 2: 0, 3: 0}
    zero_q_rows = 0
    zero_q_zero_residual_rows = 0
    for residual, q_value, std in zip(residuals, q_values, std_values):
        if q_value == 0.0:
            zero_q_rows += 1
            if residual == 0.0:
                zero_q_zero_residual_rows += 1
        for multiplier in coverage:
            if abs(residual) <= multiplier * std:
                coverage[multiplier] += 1
        if std > 0.0:
            standardized.append(residual / std)
    count = len(rows)
    summary = {
        "rows": count,
        "mae": math.fsum(abs(value) for value in residuals) / count,
        "rmse": math.sqrt(math.fsum(value * value for value in residuals) / count),
        "q": _distribution(q_values),
        "variance": _distribution(variances),
        "std": _distribution(std_values),
        "coverage": {
            f"{multiplier}sigma": coverage[multiplier] / count
            for multiplier in coverage
        },
        "standardized_residual": _distribution(standardized),
    }
    if include_zero_q_counts:
        summary.update(
            {
                "zero_q_rows": zero_q_rows,
                "zero_q_zero_residual_rows": zero_q_zero_residual_rows,
                "zero_q_nonzero_residual_rows": (
                    zero_q_rows - zero_q_zero_residual_rows
                ),
            }
        )
    return summary


def _require_equivalent(actual: Any, expected: Any, source: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{source} summary schema mismatch")
        for key in expected:
            _require_equivalent(actual[key], expected[key], f"{source}.{key}")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{source} summary value mismatch")
        if not math.isfinite(float(actual)) or not _same(float(actual), float(expected)):
            raise ValueError(f"{source} summary value mismatch")
        return
    if actual != expected:
        raise ValueError(f"{source} summary value mismatch")


def _validate_force_structures(
    root: Path,
    variant: str,
    energy: Sequence[tuple[str, int, float, float, float]],
    force_rows: Sequence[Mapping[str, str]],
    max_force_components_per_structure: int | None = None,
) -> dict[str, Any]:
    source = f"{variant}/force_structure.csv"
    rows = _read_csv(root / source, FORCE_STRUCTURE_FIELDS)
    if len(rows) != len(energy):
        raise ValueError(f"{variant} energy/force-structure alignment mismatch")
    offset = 0
    metrics = {name: [] for name in ("mae", "rmse", "mean_q", "mean_variance")}
    seen: set[str] = set()
    for row_index, (row, energy_row) in enumerate(zip(rows, energy)):
        row_source = f"{source} row {row_index}"
        structure_id, num_atoms = energy_row[:2]
        if row.get("variant") != variant or row.get("target") != "forces":
            raise ValueError(f"{row_source} variant or target mismatch")
        if row.get("structure_id") in seen:
            raise ValueError(f"{source} contains duplicate structure_id")
        seen.add(row.get("structure_id", ""))
        if (
            row.get("structure_id") != structure_id
            or _integer(row, "num_atoms", row_source) != num_atoms
        ):
            raise ValueError(f"{variant} energy/force-structure alignment mismatch")
        components = _integer(row, "components", row_source)
        expected_components = num_atoms * 3
        if max_force_components_per_structure is not None:
            expected_components = min(expected_components, max_force_components_per_structure)
        if components != expected_components:
            raise ValueError(f"{row_source} force component count mismatch")
        group = force_rows[offset : offset + components]
        offset += components
        residuals = [_finite(item, "residual", row_source) for item in group]
        q_values = [_finite(item, "q", row_source) for item in group]
        variances = [_finite(item, "variance", row_source) for item in group]
        expected = {
            "mae": math.fsum(abs(value) for value in residuals) / components,
            "rmse": math.sqrt(
                math.fsum(value * value for value in residuals) / components
            ),
            "mean_q": math.fsum(q_values) / components,
            "mean_variance": math.fsum(variances) / components,
        }
        for field, expected_value in expected.items():
            actual = _finite(row, field, row_source)
            if actual < 0.0 or not _same(actual, expected_value):
                raise ValueError(
                    f"variant alignment or {row_source} {field} mismatch "
                    "with force components"
                )
            metrics[field].append(actual)
    return {
        "rows": len(rows),
        "components": len(force_rows),
        **{field: _distribution(values) for field, values in metrics.items()},
    }


def _validate_summary(
    root: Path,
    require_zero_q_counts: bool,
    variant: str,
    energy_rows: Sequence[Mapping[str, str]],
    force_rows: Sequence[Mapping[str, str]],
    force_structure_summary: Mapping[str, Any],
) -> dict[str, Any]:
    source = f"{variant}/summary.json"
    document = _require_keys(_strict_json_load(root / source), _SUMMARY_FIELDS, source)
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{source} summary schema version mismatch")
    if document["formula_version"] != FORMULA_VERSION:
        raise ValueError(f"{source} summary formula version mismatch")
    if document["variant"] != variant:
        raise ValueError(f"{source} summary variant mismatch")
    ridge = _require_keys(document["ridge"], {"mode", "value"}, f"{source}.ridge")
    if not isinstance(ridge["mode"], str) or not ridge["mode"]:
        raise ValueError(f"{source} summary ridge mode mismatch")
    if not isinstance(ridge["value"], (int, float)) or not math.isfinite(
        float(ridge["value"])
    ) or float(ridge["value"]) < 0.0:
        raise ValueError(f"{source} summary ridge value mismatch")
    alpha = _require_keys(document["alpha"], {"energy", "forces"}, f"{source}.alpha")
    for target, value in alpha.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{source} summary {target} alpha must be finite and positive")
    expected_counts = {
        "structures": len(energy_rows),
        "force_components": len(force_rows),
        "force_structures": len(energy_rows),
    }
    _require_equivalent(document["counts"], expected_counts, f"{source}.counts")
    force_zero_q_counts: dict[str, int] | None = None
    for target, rows in (("energy", energy_rows), ("forces", force_rows)):
        target_fields = (
            set(document[target]) if isinstance(document[target], Mapping) else set()
        )
        if target_fields == _TARGET_SUMMARY_FIELDS:
            if require_zero_q_counts:
                raise ValueError(
                    f"{source}.{target} zero-q summary schema mismatch"
                )
            include_zero_q_counts = False
        elif target_fields == _TARGET_SUMMARY_FIELDS | _ZERO_Q_SUMMARY_FIELDS:
            include_zero_q_counts = True
        else:
            _require_keys(
                document[target], _TARGET_SUMMARY_FIELDS, f"{source}.{target}"
            )
            raise AssertionError("unreachable")
        expected_target = _target_summary(
            rows, include_zero_q_counts=include_zero_q_counts
        )
        _require_equivalent(
            document[target],
            expected_target,
            f"{source}.{target}",
        )
        if target == "forces" and include_zero_q_counts:
            force_zero_q_counts = {
                field: int(expected_target[field])
                for field in _ZERO_Q_SUMMARY_FIELDS
            }
    _require_equivalent(
        document["force_structure"],
        force_structure_summary,
        f"{source}.force_structure",
    )
    diagnostics = _require_keys(
        document["cholesky_diagnostics"],
        _CHOLESKY_FIELDS,
        f"{source}.cholesky_diagnostics",
    )
    for key, value in diagnostics.items():
        if value is None:
            continue
        if key == "ridge_mode":
            if not isinstance(value, str):
                raise ValueError(f"{source} summary cholesky diagnostics mismatch")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{source} summary cholesky diagnostics mismatch")
    return {
        "alpha": {target: float(value) for target, value in alpha.items()},
        "ridge": {"mode": ridge["mode"], "value": float(ridge["value"])},
        "force_zero_q_counts": force_zero_q_counts,
    }




def _validate_variance_formula(
    rows: Sequence[Mapping[str, str]], alpha: float, source: str
) -> None:
    for row_index, row in enumerate(rows):
        row_source = f"{source} row {row_index}"
        q_value = _finite(row, "q", row_source)
        variance = _finite(row, "variance", row_source)
        expected = alpha * alpha * q_value
        if not math.isfinite(expected) or (
            alpha > 0.0 and q_value > 0.0 and expected == 0.0
        ):
            raise ValueError(
                f"{row_source} variance alpha squared q result must be finite and representable"
            )
        scale = max(abs(variance), abs(expected), 1.0e-30)
        if abs(variance - expected) > RELATIVE_TOLERANCE * scale:
            raise ValueError(
                f"{row_source} variance does not match alpha squared times q"
            )




def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_delta)
        * math.fsum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return None
    result = math.fsum(
        first * second for first, second in zip(left_delta, right_delta)
    ) / denominator
    return max(-1.0, min(1.0, result))


def _quality_diagnostics(
    residuals: Sequence[float], std_values: Sequence[float]
) -> dict[str, Any]:
    standardized = []
    undefined = 0
    zero_std = 0
    coverage = {1: 0, 2: 0, 3: 0}
    for residual, std in zip(residuals, std_values):
        for multiplier in coverage:
            if abs(residual) <= multiplier * std:
                coverage[multiplier] += 1
        if std > 0.0:
            standardized.append(residual / std)
        else:
            zero_std += 1
            if residual == 0.0:
                standardized.append(0.0)
            else:
                undefined += 1
    count = len(residuals)
    return {
        "rows": count,
        "coverage_1sigma": coverage[1] / count,
        "coverage_2sigma": coverage[2] / count,
        "coverage_3sigma": coverage[3] / count,
        "uncertainty_residual_correlation": _correlation(
            std_values, [abs(value) for value in residuals]
        ),
        "standardized_residual": _distribution(standardized),
        "zero_std_rows": zero_std,
        "undefined_standardized_residual_rows": undefined,
    }


def _normalise_identities(identity: Mapping[str, Any]) -> dict[str, Any]:
    curvature = identity.get("curvature")
    calibration = identity.get("calibration")
    curvature_identity = (
        curvature.get("identity") if isinstance(curvature, Mapping) else None
    )
    calibration_identity = (
        calibration.get("identity") if isinstance(calibration, Mapping) else None
    )
    data = identity.get("data")
    if data is None:
        data = {
            "build": (
                curvature_identity.get("build")
                if isinstance(curvature_identity, Mapping)
                else None
            ),
            "calibration": (
                calibration_identity.get("calibration")
                if isinstance(calibration_identity, Mapping)
                else None
            ),
            "test": identity.get("test"),
        }
    config = identity.get("config")
    if config is None:
        config = {
            key: identity.get(key)
            for key in (
                "ridge",
                "min_q",
                "limits",
                "observables",
                "curvature",
                "calibration",
            )
        }
    return {
        "checkpoint": identity.get("checkpoint"),
        "data": data,
        "readout": identity.get("readout"),
        "config": config,
    }




def _identity_mapping(
    value: Any, fields: set[str], source: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"trusted evaluation progress identity {source} is invalid")
    return value


def _identity_string(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"trusted evaluation progress identity {source} is invalid")
    return value


def _identity_sha256(value: Any, source: str) -> str:
    text = _identity_string(value, source)
    if len(text) != 64 or any(character not in "0123456789abcdefABCDEF" for character in text):
        raise ValueError(f"trusted evaluation progress identity {source} is invalid")
    return text


def _identity_number(
    value: Any,
    source: str,
    *,
    minimum: float,
    strict: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"trusted evaluation progress identity {source} is invalid")
    result = float(value)
    if not math.isfinite(result) or (result <= minimum if strict else result < minimum):
        raise ValueError(f"trusted evaluation progress identity {source} is invalid")
    return result


def _identity_atomic_numbers(value: Any, source: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"trusted evaluation progress identity {source} is invalid")
    return value


def _validate_checkpoint_identity(value: Any, source: str) -> Mapping[str, Any]:
    checkpoint = _identity_mapping(
        value,
        {
            "sha256",
            "model_class",
            "heads",
            "selected_head",
            "r_max",
            "atomic_numbers",
            "dtype",
        },
        source,
    )
    _identity_sha256(checkpoint["sha256"], f"{source}.sha256")
    _identity_string(checkpoint["model_class"], f"{source}.model_class")
    heads = checkpoint["heads"]
    if (
        not isinstance(heads, list)
        or not heads
        or any(not isinstance(head, str) or not head.strip() for head in heads)
        or len(set(heads)) != len(heads)
    ):
        raise ValueError(f"trusted evaluation progress identity {source}.heads is invalid")
    selected_head = _identity_string(
        checkpoint["selected_head"], f"{source}.selected_head"
    )
    if selected_head not in heads:
        raise ValueError(
            f"trusted evaluation progress identity {source}.selected_head is invalid"
        )
    _identity_number(checkpoint["r_max"], f"{source}.r_max", minimum=0.0, strict=True)
    _identity_atomic_numbers(
        checkpoint["atomic_numbers"], f"{source}.atomic_numbers"
    )
    _identity_string(checkpoint["dtype"], f"{source}.dtype")
    return checkpoint


def _validate_readout_identity(value: Any, source: str) -> Mapping[str, Any]:
    readout = _identity_mapping(value, {"names", "shapes", "size"}, source)
    names = readout["names"]
    shapes = readout["shapes"]
    size = readout["size"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
        or not isinstance(shapes, list)
        or len(shapes) != len(names)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise ValueError(f"trusted evaluation progress identity {source} is invalid")
    computed_size = 0
    for shape in shapes:
        if not isinstance(shape, list) or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        ):
            raise ValueError(f"trusted evaluation progress identity {source}.shapes is invalid")
        computed_size += math.prod(shape)
    if computed_size != size:
        raise ValueError(f"trusted evaluation progress identity {source}.size is invalid")
    return readout


def _validate_dataset_identity(value: Any, source: str) -> Mapping[str, Any]:
    dataset = _identity_mapping(
        value,
        {"sha256", "identity", "size", "atomic_numbers", "r_max", "head"},
        source,
    )
    _identity_sha256(dataset["sha256"], f"{source}.sha256")
    _identity_string(dataset["identity"], f"{source}.identity")
    size = dataset["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"trusted evaluation progress identity {source}.size is invalid")
    _identity_atomic_numbers(dataset["atomic_numbers"], f"{source}.atomic_numbers")
    _identity_number(dataset["r_max"], f"{source}.r_max", minimum=0.0, strict=True)
    _identity_string(dataset["head"], f"{source}.head")
    expected_identity = stable_id(
        {
            "sha256": dataset["sha256"],
            "atomic_numbers": tuple(dataset["atomic_numbers"]),
            "r_max": dataset["r_max"],
            "head": dataset["head"],
        }
    )
    if dataset["identity"] != expected_identity:
        raise ValueError(f"trusted evaluation progress identity {source}.identity is invalid")
    return dataset


def _validate_dataset_checkpoint_compatibility(
    dataset: Mapping[str, Any], checkpoint: Mapping[str, Any], source: str
) -> None:
    if dataset["head"] != checkpoint["selected_head"]:
        raise ValueError(f"trusted evaluation progress identity {source} head mismatch")
    if dataset["atomic_numbers"] != checkpoint["atomic_numbers"]:
        raise ValueError(f"trusted evaluation progress identity {source} elements mismatch")
    if float(dataset["r_max"]) != float(checkpoint["r_max"]):
        raise ValueError(f"trusted evaluation progress identity {source} cutoff mismatch")


def _validate_limits_identity(value: Any, source: str) -> Mapping[str, Any]:
    limits = _identity_mapping(
        value,
        {"max_structures", "max_force_components_per_structure"},
        source,
    )
    for field, field_value in limits.items():
        if field_value is not None and (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value <= 0
        ):
            raise ValueError(
                f"trusted evaluation progress identity {source}.{field} is invalid"
            )
    return limits


def _validate_versions(value: Mapping[str, Any], source: str) -> None:
    if value["schema_version"] != SCHEMA_VERSION or value["formula_version"] != FORMULA_VERSION:
        raise ValueError(f"trusted evaluation progress identity {source} version mismatch")


def _validate_ridge_identity(
    value: Any, source: str, *, selected: bool
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"trusted evaluation progress identity {source} is invalid")
    mode = value.get("mode")
    active_field = "value" if mode == "fixed" else "max_condition_number"
    fields = {"mode", active_field}
    if selected:
        fields.add("selected")
    ridge = _identity_mapping(value, fields, source)
    if mode not in {"fixed", "condition_number"}:
        raise ValueError(f"trusted evaluation progress identity {source}.mode is invalid")
    threshold = _identity_number(
        ridge[active_field],
        f"{source}.{active_field}",
        minimum=0.0 if mode == "fixed" else 1.0,
        strict=mode == "condition_number",
    )
    if not math.isfinite(threshold):
        raise ValueError(f"trusted evaluation progress identity {source} is invalid")
    if selected:
        selected_values = _identity_mapping(
            ridge["selected"], set(_VARIANTS), f"{source}.selected"
        )
        for variant, selected_value in selected_values.items():
            _identity_number(
                selected_value,
                f"{source}.selected.{variant}",
                minimum=0.0,
            )
    return ridge


def _validate_curvature_source_identity(
    value: Any, source: str
) -> Mapping[str, Any]:
    identity = _identity_mapping(
        value,
        {
            "schema_version",
            "formula_version",
            "checkpoint",
            "build",
            "readout",
            "curvature",
            "limits",
        },
        source,
    )
    _validate_versions(identity, source)
    _validate_checkpoint_identity(identity["checkpoint"], f"{source}.checkpoint")
    _validate_dataset_identity(identity["build"], f"{source}.build")
    _validate_readout_identity(identity["readout"], f"{source}.readout")
    curvature = _identity_mapping(
        identity["curvature"], {"energy", "forces", "variants"}, f"{source}.curvature"
    )
    if curvature != {
        "energy": "outer(d(E/N)/dtheta, d(E/N)/dtheta)",
        "forces": "G_F.T @ G_F (unweighted)",
        "variants": list(_VARIANTS),
    }:
        raise ValueError(f"trusted evaluation progress identity {source}.curvature is invalid")
    _validate_limits_identity(identity["limits"], f"{source}.limits")
    return identity


def _validate_population_identity(value: Any, source: str) -> Mapping[str, int]:
    population = _identity_mapping(
        value, set(CALIBRATION_POPULATION_FIELDS), source
    )
    for field in CALIBRATION_POPULATION_FIELDS:
        count = population[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"trusted evaluation progress identity {source} population count is invalid"
            )
    if (
        population["force_used_structures"]
        + population["force_excluded_structures"]
        != population["energy_structures"]
        or population["force_components_used"]
        + population["force_components_excluded"]
        != population["force_components_total"]
    ):
        raise ValueError(
            f"trusted evaluation progress identity {source} population arithmetic mismatch"
        )
    return population


def _calibration_zero_q_binding(
    identity: Mapping[str, Any], source: str
) -> tuple[Mapping[str, int], str] | None:
    fields = {"zero_q_policy", "calibration_population", "force_exclusions"}
    present = fields & set(identity)
    if not present:
        return None
    if present != fields:
        raise ValueError(
            f"trusted evaluation progress identity {source} zero-q provenance is incomplete"
        )
    if identity["zero_q_policy"] != ZERO_Q_POLICY:
        raise ValueError(
            f"trusted evaluation progress identity {source} zero-q policy mismatch"
        )
    population = _validate_population_identity(
        identity["calibration_population"], f"{source}.calibration_population"
    )
    audit = _identity_mapping(
        identity["force_exclusions"],
        {"path", "sha256"},
        f"{source}.force_exclusions",
    )
    if audit["path"] != "force_exclusions.json":
        raise ValueError(
            f"trusted evaluation progress identity {source} zero-q audit path mismatch"
        )
    audit_sha256 = _identity_sha256(
        audit["sha256"], f"{source}.force_exclusions.sha256"
    )
    if audit_sha256 != audit_sha256.lower():
        raise ValueError(
            f"trusted evaluation progress identity {source} zero-q audit SHA is invalid"
        )
    return population, audit_sha256


def _validate_evaluation_zero_q_identity(
    value: Any,
    calibration_identity: Mapping[str, Any],
    records: Mapping[str, Any],
) -> None:
    calibration_binding = _calibration_zero_q_binding(
        calibration_identity, "calibration.identity"
    )
    if value is None:
        if calibration_binding is not None:
            raise ValueError("trusted evaluation progress identity zero-q binding is missing")
        return
    zero_q = _identity_mapping(
        value,
        {"policy", "calibration_population", "force_exclusions_sha256"},
        "zero_q",
    )
    if zero_q["policy"] != ZERO_Q_POLICY or calibration_binding is None:
        raise ValueError("trusted evaluation progress identity zero-q policy mismatch")
    population = _validate_population_identity(
        zero_q["calibration_population"], "zero_q.calibration_population"
    )
    audit_sha256 = _identity_sha256(
        zero_q["force_exclusions_sha256"], "zero_q.force_exclusions_sha256"
    )
    if audit_sha256 != audit_sha256.lower():
        raise ValueError("trusted evaluation progress identity zero-q audit SHA is invalid")
    calibration_population, calibration_audit_sha256 = calibration_binding
    if dict(population) != dict(calibration_population):
        raise ValueError("trusted evaluation progress identity zero-q population mismatch")
    if audit_sha256 != calibration_audit_sha256:
        raise ValueError("trusted evaluation progress identity zero-q audit SHA mismatch")
    for variant in _VARIANTS:
        if (
            records[variant]["energy"]["rows"] != population["energy_structures"]
            or records[variant]["forces"]["rows"]
            != population["force_components_used"]
        ):
            raise ValueError("trusted evaluation progress identity zero-q record population mismatch")


def _validate_calibration_source_identity(
    value: Any, source: str
) -> Mapping[str, Any]:
    fields = {
        "schema_version",
        "formula_version",
        "checkpoint",
        "curvature",
        "calibration",
        "ridge",
        "min_q",
        "limits",
    }
    optional_fields = {
        "curvature_artifact",
        "consumer_limits",
        "zero_q_policy",
        "calibration_population",
        "force_exclusions",
    }
    if (
        not isinstance(value, Mapping)
        or not fields.issubset(value)
        or not set(value).issubset(fields | optional_fields)
    ):
        raise ValueError(f"trusted evaluation progress identity {source} is invalid")
    identity = value
    if "curvature_artifact" in identity:
        artifact = _identity_mapping(
            identity["curvature_artifact"],
            {"path", "sha256"},
            f"{source}.curvature_artifact",
        )
        artifact_path = _identity_string(
            artifact["path"], f"{source}.curvature_artifact.path"
        )
        if not Path(artifact_path).is_absolute():
            raise ValueError(
                f"trusted evaluation progress identity "
                f"{source}.curvature_artifact.path is invalid"
            )
        _identity_sha256(
            artifact["sha256"], f"{source}.curvature_artifact.sha256"
        )
    _validate_versions(identity, source)
    _validate_checkpoint_identity(identity["checkpoint"], f"{source}.checkpoint")
    _validate_curvature_source_identity(identity["curvature"], f"{source}.curvature")
    _validate_dataset_identity(identity["calibration"], f"{source}.calibration")
    _validate_ridge_identity(identity["ridge"], f"{source}.ridge", selected=True)
    min_q = _identity_number(
        identity["min_q"], f"{source}.min_q", minimum=0.0, strict=True
    )
    if min_q != 1.0e-30:
        raise ValueError(f"trusted evaluation progress identity {source}.min_q is invalid")
    _validate_limits_identity(identity["limits"], f"{source}.limits")
    if "consumer_limits" in identity:
        _validate_limits_identity(
            identity["consumer_limits"], f"{source}.consumer_limits"
        )
    _calibration_zero_q_binding(identity, source)
    return identity


def _validate_calibration_records(
    value: Any, ridge_identity: Mapping[str, Any], source: str
) -> Mapping[str, Any]:
    variants = _identity_mapping(value, set(_VARIANTS), source)
    for variant in _VARIANTS:
        targets = _identity_mapping(
            variants[variant], {"energy", "forces"}, f"{source}.{variant}"
        )
        for target in ("energy", "forces"):
            record_source = f"{source}.{variant}.{target}"
            record = _identity_mapping(
                targets[target], {"ridge_mode", "ridge", "alpha", "rows"}, record_source
            )
            if record["ridge_mode"] not in {"fixed", "condition_number"}:
                raise ValueError(
                    f"trusted evaluation progress identity {record_source}.ridge_mode is invalid"
                )
            record_ridge = _identity_number(
                record["ridge"], f"{record_source}.ridge", minimum=0.0
            )
            _identity_number(
                record["alpha"],
                f"{record_source}.alpha",
                minimum=0.0,
                strict=True,
            )
            if record["ridge_mode"] != ridge_identity["mode"] or not _scale_aware_same(
                record_ridge, float(ridge_identity["selected"][variant])
            ):
                raise ValueError(
                    f"trusted evaluation progress identity {record_source} ridge mismatch"
                )
            rows = record["rows"]
            if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
                raise ValueError(
                    f"trusted evaluation progress identity {record_source}.rows is invalid"
                )
    return variants


def _regular_file_snapshot(path: Path, source: str) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    non_blocking = getattr(os, "O_NONBLOCK", None)
    if no_follow is None or non_blocking is None:
        raise ValueError(f"{source} requires safe nonblocking no-follow open")
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow | non_blocking)
    except OSError as error:
        raise ValueError(f"{source} is unavailable") from error
    try:
        handle = os.fdopen(descriptor, "rb")
    except OSError as error:
        os.close(descriptor)
        raise ValueError(f"{source} is invalid") from error
    except BaseException:
        os.close(descriptor)
        raise
    try:
        with handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"{source} must be regular")
            return handle.read()
    except OSError as error:
        raise ValueError(f"{source} is invalid") from error


def _load_progress_snapshot(path: Path) -> tuple[Any, str]:
    payload = _regular_file_snapshot(path, "trusted evaluation progress")
    try:
        progress = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True
        )
    except TypeError:
        progress = torch.load(io.BytesIO(payload), map_location="cpu")
    return progress, hashlib.sha256(payload).hexdigest()


def _validate_complete_progress(
    progress: Mapping[str, Any], *, policy_bound: bool
) -> Mapping[str, int] | None:
    fields = set(progress)
    modern_fields = _PROGRESS_BASE_FIELDS | _ZERO_Q_SUMMARY_FIELDS
    valid_schemas = (modern_fields,) if policy_bound else (
        _PROGRESS_BASE_FIELDS,
        modern_fields,
    )
    if fields not in valid_schemas:
        raise ValueError("trusted evaluation progress schema mismatch")
    for field in ("next_index", "structures"):
        value = progress[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"trusted evaluation progress {field} is invalid")
    if progress["next_index"] != progress["structures"]:
        raise ValueError("trusted evaluation progress structure counters mismatch")
    offsets = progress["csv_offsets"]
    expected_offsets = {
        f"{variant}/{filename}"
        for variant in _VARIANTS
        for filename in (
            "energy.csv",
            "force_components.csv",
            "force_structure.csv",
        )
    }
    if not isinstance(offsets, Mapping) or set(offsets) != expected_offsets:
        raise ValueError("trusted evaluation progress CSV offsets schema mismatch")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in offsets.values()
    ):
        raise ValueError("trusted evaluation progress CSV offset is invalid")
    if not _ZERO_Q_SUMMARY_FIELDS.issubset(progress):
        return None
    counts: dict[str, int] = {}
    for field in _ZERO_Q_SUMMARY_FIELDS:
        value = progress[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("trusted evaluation progress zero-q counter is invalid")
        counts[field] = value
    if counts["zero_q_rows"] != (
        counts["zero_q_zero_residual_rows"]
        + counts["zero_q_nonzero_residual_rows"]
    ):
        raise ValueError("trusted evaluation progress zero-q counter mismatch")
    return counts


def _trusted_progress_identity(
    root: Path, explicit_identity: Mapping[str, Any] | None
) -> tuple[Mapping[str, Any], Mapping[str, int] | None, str, Mapping[str, Any]]:
    progress_path = root / "progress.pt"
    if not progress_path.is_file():
        raise ValueError("trusted evaluation progress identity is missing")
    progress, progress_sha256 = _load_progress_snapshot(progress_path)
    if not isinstance(progress, Mapping) or progress.get("status") != "complete":
        raise ValueError("trusted evaluation progress identity requires complete progress")
    identity = progress.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("trusted evaluation progress identity must be a mapping")
    if explicit_identity is not None and _strict_json_bytes(identity) != _strict_json_bytes(
        explicit_identity
    ):
        raise ValueError("trusted evaluation progress identity mismatch")
    progress_zero_q_counts = _validate_complete_progress(
        progress, policy_bound="zero_q" in identity
    )

    required = {
        "schema_version",
        "formula_version",
        "checkpoint",
        "readout",
        "curvature",
        "calibration",
        "test",
        "ridge",
        "min_q",
        "limits",
        "observables",
    }
    if not required.issubset(identity):
        raise ValueError("trusted evaluation progress identity is missing required fields")
    optional = {"consumer_limits", "zero_q"}
    if not set(identity).issubset(required | optional):
        raise ValueError("trusted evaluation progress identity has unexpected fields")
    _validate_versions(identity, "root")
    checkpoint = _validate_checkpoint_identity(identity["checkpoint"], "checkpoint")
    readout = _validate_readout_identity(identity["readout"], "readout")

    curvature_value = identity["curvature"]
    if not isinstance(curvature_value, Mapping) or set(curvature_value) not in (
        {"sha256", "identity"},
        {"path", "sha256", "identity"},
    ):
        raise ValueError("trusted evaluation progress identity curvature is invalid")
    curvature = curvature_value
    if "path" in curvature:
        curvature_path = _identity_string(curvature["path"], "curvature.path")
        if not Path(curvature_path).is_absolute():
            raise ValueError(
                "trusted evaluation progress identity curvature.path is invalid"
            )
    _identity_sha256(curvature["sha256"], "curvature.sha256")
    curvature_identity = _validate_curvature_source_identity(
        curvature["identity"], "curvature.identity"
    )

    calibration = _identity_mapping(
        identity["calibration"],
        {"sha256", "diagnostics_sha256", "identity", "records"},
        "calibration",
    )
    _identity_sha256(calibration["sha256"], "calibration.sha256")
    _identity_sha256(
        calibration["diagnostics_sha256"], "calibration.diagnostics_sha256"
    )
    calibration_identity = _validate_calibration_source_identity(
        calibration["identity"], "calibration.identity"
    )
    calibration_records = _validate_calibration_records(
        calibration["records"], calibration_identity["ridge"], "calibration.records"
    )
    _validate_evaluation_zero_q_identity(
        identity.get("zero_q"), calibration_identity, calibration_records
    )
    test = _validate_dataset_identity(identity["test"], "test")
    ridge = _validate_ridge_identity(identity["ridge"], "ridge", selected=False)
    min_q = _identity_number(
        identity["min_q"], "min_q", minimum=0.0, strict=True
    )
    if min_q != 1.0e-30:
        raise ValueError("trusted evaluation progress identity min_q is invalid")
    limits = _validate_limits_identity(identity["limits"], "limits")
    consumer_limits = (
        _validate_limits_identity(identity["consumer_limits"], "consumer_limits")
        if "consumer_limits" in identity
        else None
    )
    observables = _identity_mapping(
        identity["observables"],
        {"energy", "forces", "residual", "variance"},
        "observables",
    )
    if observables != {
        "energy": "per_atom",
        "forces": "per_cartesian_component",
        "residual": "reference_minus_prediction",
        "variance": "alpha_squared_times_q",
    }:
        raise ValueError("trusted evaluation progress identity observables is invalid")

    relationships = (
        (checkpoint, curvature_identity["checkpoint"], "curvature checkpoint"),
        (checkpoint, calibration_identity["checkpoint"], "calibration checkpoint"),
        (readout, curvature_identity["readout"], "curvature readout"),
        (curvature_identity, calibration_identity["curvature"], "calibration curvature"),
        (limits, curvature_identity["limits"], "curvature limits"),
        (limits, calibration_identity["limits"], "calibration limits"),
    )
    for expected, actual, source in relationships:
        if _strict_json_bytes(expected) != _strict_json_bytes(actual):
            raise ValueError(
                f"trusted evaluation progress identity {source} mismatch"
            )

    calibration_consumer_limits = calibration_identity.get("consumer_limits")
    if _strict_json_bytes(consumer_limits) != _strict_json_bytes(
        calibration_consumer_limits
    ):
        raise ValueError(
            "trusted evaluation progress identity calibration consumer limits mismatch"
        )

    calibration_curvature_artifact = calibration_identity.get(
        "curvature_artifact"
    )
    if calibration_curvature_artifact is not None:
        evaluation_curvature_artifact = {
            key: curvature[key] for key in ("path", "sha256") if key in curvature
        }
        if _strict_json_bytes(calibration_curvature_artifact) != _strict_json_bytes(
            evaluation_curvature_artifact
        ):
            raise ValueError(
                "trusted evaluation progress identity curvature artifact mismatch"
            )
    _validate_dataset_checkpoint_compatibility(
        curvature_identity["build"], checkpoint, "build dataset"
    )
    _validate_dataset_checkpoint_compatibility(
        calibration_identity["calibration"], checkpoint, "calibration dataset"
    )
    _validate_dataset_checkpoint_compatibility(test, checkpoint, "test dataset")
    calibration_ridge = calibration_identity["ridge"]
    active_field = "value" if ridge["mode"] == "fixed" else "max_condition_number"
    if (
        calibration_ridge["mode"] != ridge["mode"]
        or float(calibration_ridge[active_field]) != float(ridge[active_field])
    ):
        raise ValueError("trusted evaluation progress identity ridge mismatch")
    if float(calibration_identity["min_q"]) != min_q:
        raise ValueError("trusted evaluation progress identity min_q mismatch")
    return identity, progress_zero_q_counts, progress_sha256, progress


def _validate_summary_calibration_contract(
    variant: str,
    summary: Mapping[str, Any],
    records: Mapping[str, Any],
) -> None:
    for target in ("energy", "forces"):
        record = records[variant][target]
        if (
            summary["ridge"]["mode"] != record["ridge_mode"]
            or not _scale_aware_same(
                float(summary["ridge"]["value"]), float(record["ridge"])
            )
        ):
            raise ValueError(
                f"{variant}/summary.json ridge mismatch with calibration record"
            )
        if not _scale_aware_same(
            float(summary["alpha"][target]), float(record["alpha"])
        ):
            raise ValueError(
                f"{variant}/summary.json {target} alpha mismatch with calibration record"
            )


def _publication_identities(
    root: Path, explicit_identity: Mapping[str, Any] | None
) -> tuple[
    dict[str, Any],
    Mapping[str, Any],
    Mapping[str, int] | None,
    str,
    Mapping[str, Any],
]:
    source, progress_zero_q_counts, progress_sha256, progress = (
        _trusted_progress_identity(root, explicit_identity)
    )

    existing_path = root / "manifest.json"
    existing: Mapping[str, Any] | None = None
    if existing_path.is_file():
        value = _strict_json_load(existing_path)
        if not isinstance(value, Mapping) or not isinstance(
            value.get("identities"), Mapping
        ):
            raise ValueError("manifest identity is invalid")
        if value.get("schema_version") != SCHEMA_VERSION or value.get(
            "formula_version"
        ) != FORMULA_VERSION:
            raise ValueError("manifest schema or formula identity mismatch")
        existing = value["identities"]

    normalized = _normalise_identities(source)
    if existing is not None and _strict_json_bytes(existing) != _strict_json_bytes(
        normalized
    ):
        raise ValueError("manifest identity mismatch")
    return normalized, source, progress_zero_q_counts, progress_sha256, progress


def _references_aligned(
    actual: Sequence[tuple[Any, ...]],
    expected: Sequence[tuple[Any, ...]],
    *,
    key_fields: int,
) -> bool:
    """Require shared keys and reference observations, not shared predictions."""
    if len(actual) != len(expected):
        return False
    return all(
        left[:key_fields] == right[:key_fields]
        and _scale_aware_same(float(left[key_fields]), float(right[key_fields]))
        for left, right in zip(actual, expected)
    )


def _canonical_file_hashes(root: Path) -> dict[str, str]:
    return {
        f"{variant}/{filename}": sha256_file(root / variant / filename)
        for variant in _VARIANTS
        for filename in _CANONICAL_FILES
    }


def _validation_relative_paths() -> tuple[str, ...]:
    return ("progress.pt",) + tuple(
        f"{variant}/{filename}"
        for variant in _VARIANTS
        for filename in _CANONICAL_FILES
    )


def _stream_regular_file(
    path: Path,
    source: str,
    destination: Path | None = None,
) -> tuple[str, int]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    non_blocking = getattr(os, "O_NONBLOCK", None)
    if no_follow is None or non_blocking is None:
        raise ValueError(f"{source} requires safe nonblocking no-follow open")
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow | non_blocking)
    except OSError as error:
        raise ValueError(f"{source} is unavailable") from error
    destination_handle = None
    destination_descriptor = -1
    try:
        metadata_before = os.fstat(descriptor)
        if not stat.S_ISREG(metadata_before.st_mode):
            raise ValueError(f"{source} must be regular")
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination_descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
                0o600,
            )
            destination_handle = os.fdopen(destination_descriptor, "wb")
            destination_descriptor = -1
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                if destination_handle is not None:
                    destination_handle.write(chunk)
            metadata_after = os.fstat(handle.fileno())
        if destination_handle is not None:
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
            destination_handle.close()
            destination_handle = None
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(metadata_before, field) != getattr(metadata_after, field)
            for field in stable_fields
        ) or size != metadata_before.st_size:
            raise ValueError(f"{source} changed while being read")
        return digest.hexdigest(), size
    except (OSError, ValueError) as error:
        if destination is not None:
            destination.unlink(missing_ok=True)
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"{source} is invalid") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if destination_handle is not None:
            destination_handle.close()
        if destination_descriptor >= 0:
            os.close(destination_descriptor)


def _stable_snapshot_file(live: Path, snapshot: Path, source: str) -> str:
    before, _ = _stream_regular_file(live, source)
    copied, copied_size = _stream_regular_file(live, source, snapshot)
    after, _ = _stream_regular_file(live, source)
    snapshot_sha256, snapshot_size = _stream_regular_file(
        snapshot, f"private snapshot {source}"
    )
    if (
        before != copied
        or copied != after
        or after != snapshot_sha256
        or copied_size != snapshot_size
    ):
        snapshot.unlink(missing_ok=True)
        raise ValueError("publication input changed during snapshot")
    return snapshot_sha256


def _create_validation_snapshot(root: Path, snapshot: Path) -> dict[str, str]:
    hashes = {
        relative_path: _stable_snapshot_file(
            root / relative_path,
            snapshot / relative_path,
            f"validation input {relative_path}",
        )
        for relative_path in _validation_relative_paths()
    }
    for relative_path in _validation_relative_paths():
        if not relative_path.endswith(".csv"):
            continue
        csv_path = snapshot / relative_path
        try:
            with csv_path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                complete = handle.read(1) == b"\n"
        except (OSError, ValueError):
            complete = False
        if not complete:
            raise ValueError(
                "trusted evaluation progress CSV offset requires complete final row: "
                f"{relative_path}"
            )
    manifest_path = root / "manifest.json"
    if os.path.lexists(manifest_path):
        _stable_snapshot_file(
            manifest_path,
            snapshot / "manifest.json",
            "validation identity manifest.json",
        )
    _assert_validation_inputs(
        root, hashes, "publication input changed during snapshot"
    )
    return hashes


def _assert_validation_inputs(
    root: Path, expected_hashes: Mapping[str, str], message: str
) -> None:
    if set(expected_hashes) != set(_validation_relative_paths()):
        raise ValueError("validation input hash schema mismatch")
    try:
        current = {
            relative_path: _stream_regular_file(
                root / relative_path, f"validation input {relative_path}"
            )[0]
            for relative_path in _validation_relative_paths()
        }
    except ValueError as error:
        raise ValueError(message) from error
    if current != expected_hashes:
        raise ValueError(message)


_VALIDATION_LOCK_ROOT = Path("/tmp")


def _validation_effective_user_id() -> int:
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise ValueError("validation output locking requires POSIX effective-user IDs")
    return os.geteuid()


def _canonical_validation_root(root: Path) -> Path:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError("publication root requires safe POSIX directory open")
    try:
        canonical_root = Path(root).resolve(strict=True)
    except OSError as error:
        raise ValueError("publication root cannot be resolved") from error
    descriptor = -1
    try:
        descriptor = os.open(
            canonical_root,
            os.O_RDONLY | no_follow | directory_flag,
        )
        descriptor_state = os.fstat(descriptor)
        path_state = os.stat(canonical_root, follow_symlinks=False)
        effective_user_id = _validation_effective_user_id()
        if (
            not stat.S_ISDIR(descriptor_state.st_mode)
            or not stat.S_ISDIR(path_state.st_mode)
            or descriptor_state.st_uid != effective_user_id
            or path_state.st_uid != effective_user_id
            or (descriptor_state.st_dev, descriptor_state.st_ino)
            != (path_state.st_dev, path_state.st_ino)
        ):
            raise ValueError(
                "publication root must be a stable directory owned by the "
                "current effective user; multi-user publication writers are "
                "not supported"
            )
    except OSError as error:
        raise ValueError("publication root is unavailable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return canonical_root


def _validation_lock_directory_path() -> Path:
    return (
        _VALIDATION_LOCK_ROOT
        / f"mace-llpr-validation-locks-{_validation_effective_user_id()}"
    )


def _validation_lock_name(root: Path) -> str:
    canonical_root = _canonical_validation_root(root)
    digest = hashlib.sha256(os.fsencode(os.fspath(canonical_root))).hexdigest()
    return f"{digest}.lock"


def _open_validation_lock_directory() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError("validation lock directory requires safe no-follow open")
    flags = os.O_RDONLY | no_follow | directory_flag
    temporary_root = _VALIDATION_LOCK_ROOT
    parent_descriptor = -1
    directory_descriptor = -1
    try:
        try:
            parent_descriptor = os.open(temporary_root, flags)
            parent_state = os.fstat(parent_descriptor)
            parent_mode = stat.S_IMODE(parent_state.st_mode)
            effective_user_id = _validation_effective_user_id()
            if (
                not stat.S_ISDIR(parent_state.st_mode)
                or parent_state.st_uid not in {0, effective_user_id}
                or (
                    parent_mode & 0o022
                    and not (parent_state.st_mode & stat.S_ISVTX)
                )
            ):
                raise ValueError("validation lock temporary directory is unsafe")
            directory_name = _validation_lock_directory_path().name
            try:
                os.mkdir(directory_name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                pass
            except OSError as error:
                raise ValueError(
                    "validation lock directory is unavailable"
                ) from error
            try:
                directory_descriptor = os.open(
                    directory_name,
                    flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as error:
                raise ValueError(
                    "validation lock directory is unavailable"
                ) from error
            directory_state = os.fstat(directory_descriptor)
            path_state = os.stat(
                directory_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(directory_state.st_mode)
                or not stat.S_ISDIR(path_state.st_mode)
                or directory_state.st_uid != effective_user_id
                or path_state.st_uid != effective_user_id
                or stat.S_IMODE(directory_state.st_mode) != 0o700
                or stat.S_IMODE(path_state.st_mode) != 0o700
                or (directory_state.st_dev, directory_state.st_ino)
                != (path_state.st_dev, path_state.st_ino)
            ):
                raise ValueError(
                    "validation lock directory must be private and owned by "
                    "this user"
                )
        except OSError as error:
            raise ValueError("validation lock directory is invalid") from error
        finally:
            if parent_descriptor >= 0:
                os.close(parent_descriptor)
    except BaseException:
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        raise
    return directory_descriptor


def _require_private_validation_lock(descriptor: int) -> os.stat_result:
    try:
        descriptor_state = os.fstat(descriptor)
    except OSError as error:
        raise ValueError("validation output lock is invalid") from error
    if (
        not stat.S_ISREG(descriptor_state.st_mode)
        or descriptor_state.st_uid != _validation_effective_user_id()
        or stat.S_IMODE(descriptor_state.st_mode) != 0o600
        or descriptor_state.st_nlink != 1
        or descriptor_state.st_size != 0
    ):
        raise ValueError(
            "validation output lock must be an empty private regular file "
            "owned by this user"
        )
    return descriptor_state


def _lock_entry_matches_descriptor(
    directory_descriptor: int,
    lock_name: str,
    descriptor_state: os.stat_result,
) -> bool:
    try:
        path_state = os.stat(
            lock_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return False
    return (
        stat.S_ISREG(path_state.st_mode)
        and path_state.st_uid == _validation_effective_user_id()
        and stat.S_IMODE(path_state.st_mode) == 0o600
        and path_state.st_nlink == 1
        and path_state.st_size == 0
        and (descriptor_state.st_dev, descriptor_state.st_ino)
        == (path_state.st_dev, path_state.st_ino)
    )


def _open_validation_lock(directory_descriptor: int, lock_name: str) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    non_blocking = getattr(os, "O_NONBLOCK", None)
    if no_follow is None or non_blocking is None:
        raise ValueError("validation output lock requires safe no-follow open")
    try:
        return os.open(
            lock_name,
            os.O_RDWR | os.O_CREAT | no_follow | non_blocking,
            0o600,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise ValueError("validation output lock is unavailable") from error


@contextmanager
def _validation_output_lock(root: Path) -> Iterator[None]:
    canonical_root = _canonical_validation_root(root)
    directory_descriptor = _open_validation_lock_directory()
    descriptor = -1
    locked = False
    try:
        lock_name = _validation_lock_name(canonical_root)
        descriptor = _open_validation_lock(directory_descriptor, lock_name)
        _require_private_validation_lock(descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        descriptor_state = _require_private_validation_lock(descriptor)
        if not _lock_entry_matches_descriptor(
            directory_descriptor,
            lock_name,
            descriptor_state,
        ):
            raise ValueError("validation output lock changed while being acquired")
        yield
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            try:
                if descriptor >= 0:
                    os.close(descriptor)
            finally:
                os.close(directory_descriptor)


def _restore_validation_reports(
    root: Path, previous: Mapping[str, bytes | None]
) -> None:
    for name, payload in previous.items():
        path = root / name
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_bytes_dump(path, payload)


def _publish_validation_reports(
    root: Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> None:
    report_names = ("manifest.json", "validation.json")
    previous = {
        name: (
            _regular_file_snapshot(root / name, f"existing {name}")
            if (root / name).is_file()
            else None
        )
        for name in report_names
    }
    promoted = False
    with TemporaryDirectory(prefix=".validation-staging-", dir=root) as directory:
        staging = Path(directory)
        _atomic_strict_json_dump(staging / "manifest.json", manifest)
        _atomic_strict_json_dump(staging / "validation.json", report)
        _assert_validation_inputs(
            root,
            input_hashes,
            "publication input changed during report publication",
        )
        try:
            promoted = True
            for name in report_names:
                os.replace(staging / name, root / name)
            _assert_validation_inputs(
                root,
                input_hashes,
                "publication input changed during report publication",
            )
        except BaseException:
            if promoted:
                _restore_validation_reports(root, previous)
            raise


def _validate_existing_manifest(
    root: Path, identities: Mapping[str, Any]
) -> None:
    source = "manifest.json"
    document = _require_keys(
        _strict_json_load(root / source),
        {
            "schema_version",
            "formula_version",
            "identities",
            "units",
            "conventions",
            "schemas",
            "files",
        },
        source,
    )
    if document["schema_version"] != SCHEMA_VERSION or document[
        "formula_version"
    ] != FORMULA_VERSION:
        raise ValueError("manifest schema or formula identity mismatch")
    if _strict_json_bytes(document["identities"]) != _strict_json_bytes(
        identities
    ):
        raise ValueError("manifest identity mismatch")
    if document["units"] != {"energy": "eV/atom", "forces": "eV/" + chr(197)}:
        raise ValueError("manifest units mismatch")
    expected_schemas = {
        "energy.csv": list(ENERGY_FIELDS),
        "force_components.csv": list(FORCE_FIELDS),
        "force_structure.csv": list(FORCE_STRUCTURE_FIELDS),
        "summary.json": sorted(_SUMMARY_FIELDS),
    }
    if document["schemas"] != expected_schemas:
        raise ValueError("manifest schemas mismatch")
    files = document["files"]
    expected_hashes = _canonical_file_hashes(root)
    if not isinstance(files, Mapping) or set(files) != set(expected_hashes):
        raise ValueError("manifest files schema mismatch")
    for relative_path, expected_hash in expected_hashes.items():
        actual_hash = files[relative_path]
        if not isinstance(actual_hash, str) or actual_hash != expected_hash:
            raise ValueError(f"manifest SHA256 mismatch for {relative_path}")


def _validate_publication_root_locked(
    snapshot_root: Path,
    live_root: Path,
    input_hashes: Mapping[str, str],
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate canonical LLPR results and atomically refresh their reports.

    Statistical quality metrics are diagnostics only. Structural, identity,
    hash, and numerical-integrity violations raise ``ValueError``.
    """
    root = Path(snapshot_root)
    if not root.is_dir():
        raise ValueError(f"publication root is not a directory: {root}")
    (
        identities,
        trusted_identity,
        progress_zero_q_counts,
        progress_sha256,
        progress,
    ) = _publication_identities(root, identity)
    calibration_records = trusted_identity["calibration"]["records"]
    policy_bound = "zero_q" in trusted_identity
    consumer_limits = trusted_identity.get(
        "consumer_limits", trusted_identity["limits"]
    )
    max_force_components = consumer_limits.get(
        "max_force_components_per_structure"
    )
    manifest_path = root / "manifest.json"
    manifest_exists = manifest_path.is_file()
    if manifest_exists:
        _validate_existing_manifest(root, identities)

    diagnostics: dict[str, Any] = {}
    energy_baseline: list[tuple[str, int, float, float, float]] | None = None
    force_baseline: list[tuple[str, int, int, int, float, float, float]] | None = None
    zero_q_baseline: tuple[tuple[str, str, str, str], ...] | None = None
    summary_zero_q_counts: dict[str, Mapping[str, int] | None] = {}
    structure_counts: dict[str, tuple[int, int]] = {}
    for variant in _VARIANTS:
        energy, energy_residuals, energy_std = _validate_energy(root, variant)
        forces, force_residuals, force_std, force_rows = _validate_forces(
            root, variant, energy, max_force_components, policy_bound
        )
        energy_rows = _read_csv(root / variant / "energy.csv", ENERGY_FIELDS)
        force_structure = _validate_force_structures(
            root, variant, energy, force_rows, max_force_components
        )
        structure_counts[variant] = (
            len(energy_rows),
            int(force_structure["rows"]),
        )
        summary = _validate_summary(
            root, policy_bound, variant, energy_rows, force_rows, force_structure
        )
        summary_zero_q_counts[variant] = summary["force_zero_q_counts"]
        _validate_summary_calibration_contract(
            variant, summary, calibration_records
        )
        _validate_variance_formula(
            energy_rows, summary["alpha"]["energy"], f"{variant}/energy.csv"
        )
        _validate_variance_formula(
            force_rows, summary["alpha"]["forces"], f"{variant}/force_components.csv"
        )
        if energy_baseline is None:
            energy_baseline = energy
            force_baseline = forces
        elif not _references_aligned(
            energy, energy_baseline, key_fields=2
        ) or not _references_aligned(
            forces,
            force_baseline if force_baseline is not None else (),
            key_fields=4,
        ):
            raise ValueError("variant alignment mismatch for canonical observations")
        zero_q_keys = tuple(
            (
                row["structure_id"],
                row["num_atoms"],
                row["atom_index"],
                row["direction"],
            )
            for row in force_rows
            if float(row["q"]) == 0.0
        )
        if zero_q_baseline is None:
            zero_q_baseline = zero_q_keys
        elif zero_q_keys != zero_q_baseline:
            raise ValueError("zero-q force keys mismatch across variants")
        diagnostics[variant] = {
            "energy": _quality_diagnostics(energy_residuals, energy_std),
            "forces": _quality_diagnostics(force_residuals, force_std),
        }

    if policy_bound:
        target_structures = trusted_identity["test"]["size"]
        max_structures = consumer_limits.get("max_structures")
        if max_structures is not None:
            target_structures = min(target_structures, max_structures)
        if (
            progress["next_index"] != target_structures
            or progress["structures"] != target_structures
            or any(
                structure_counts[variant]
                != (target_structures, target_structures)
                for variant in _VARIANTS
            )
        ):
            raise ValueError(
                "trusted evaluation progress structure counters mismatch with "
                "canonical CSV and identity"
            )
        for relative_path, offset in progress["csv_offsets"].items():
            if offset != (root / relative_path).stat().st_size:
                raise ValueError(
                    f"trusted evaluation progress CSV offset mismatch: {relative_path}"
                )
        if any(
            summary_zero_q_counts[variant] != progress_zero_q_counts
            for variant in _VARIANTS
        ):
            raise ValueError("trusted evaluation progress zero-q counters mismatch")
    file_hashes = {
        relative_path: input_hashes[relative_path]
        for relative_path in input_hashes
        if relative_path != "progress.pt"
    }
    if input_hashes["progress.pt"] != progress_sha256:
        raise ValueError("trusted evaluation progress snapshot mismatch")
    _assert_validation_inputs(
        live_root,
        input_hashes,
        "trusted evaluation progress changed during validation",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "identities": identities,
        "units": {"energy": "eV/atom", "forces": "eV/Å"},
        "conventions": {
            "energy": "per_atom",
            "forces": "per_cartesian_component",
            "residual": "reference_minus_prediction",
            "variance": "alpha_squared_times_q",
            "std": "square_root_of_variance",
            "relative_tolerance": RELATIVE_TOLERANCE,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
        },
        "schemas": {
            "energy.csv": list(ENERGY_FIELDS),
            "force_components.csv": list(FORCE_FIELDS),
            "force_structure.csv": list(FORCE_STRUCTURE_FIELDS),
            "summary.json": sorted(_SUMMARY_FIELDS),
        },
        "files": file_hashes,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "status": "valid",
        "manifest_sha256": hashlib.sha256(_strict_json_bytes(manifest)).hexdigest(),
        "tolerances": {
            "relative": RELATIVE_TOLERANCE,
            "absolute": ABSOLUTE_TOLERANCE,
        },
        "diagnostics": diagnostics,
    }
    _publish_validation_reports(live_root, manifest, report, input_hashes)
    return report


def validate_publication_root(
    publication_root: Path,
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _canonical_validation_root(Path(publication_root))
    if not root.is_dir():
        raise ValueError(f"publication root is not a directory: {root}")
    if not (root / "progress.pt").is_file():
        raise ValueError("trusted evaluation progress identity is missing")
    with _validation_output_lock(root):
        with TemporaryDirectory(
            prefix=f".{root.name}.validation-snapshot-",
            dir=root.parent,
        ) as directory:
            snapshot_root = Path(directory)
            input_hashes = _create_validation_snapshot(root, snapshot_root)
            return _validate_publication_root_locked(
                snapshot_root,
                root,
                input_hashes,
                identity=identity,
            )


def run_validate(config: LLPRConfig) -> dict[str, Any]:
    """Locate and validate one workflow's deterministic evaluation output."""
    checkpoint_sha256 = sha256_file(config.checkpoint.path)
    expected_sha256 = config.checkpoint.expected_sha256
    if (
        expected_sha256 is not None
        and checkpoint_sha256.lower() != expected_sha256.lower()
    ):
        raise ValueError(
            "checkpoint SHA256 mismatch: "
            f"expected {expected_sha256}, actual {checkpoint_sha256}"
        )
    publication_root = (
        run_root(config, checkpoint_sha256) / "evaluation" / "deterministic"
    )
    return validate_publication_root(publication_root)
