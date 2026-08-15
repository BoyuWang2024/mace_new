"""Validation and deterministic manifests for LLPR publication results."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

from .artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    load_torch_artifact,
    sha256_file,
    stable_id,
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
        reference, prediction, residual, _, _, std = _validate_observation(
            row, row_source
        )
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
) -> tuple[
    list[tuple[str, int, int, int, float, float, float]],
    list[float],
    list[float],
    list[dict[str, str]],
]:
    source = f"{variant}/force_components.csv"
    rows = _read_csv(root / source, FORCE_FIELDS)
    expected_keys = [
        (structure_id, num_atoms, atom_index, direction)
        for structure_id, num_atoms, _, _, _ in energy
        for atom_index in range(num_atoms)
        for direction in range(3)
    ]
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
        reference, prediction, residual, _, _, std = _validate_observation(
            row, row_source
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


def _target_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    residuals = [_finite(row, "residual", "summary source") for row in rows]
    q_values = [_finite(row, "q", "summary source") for row in rows]
    variances = [_finite(row, "variance", "summary source") for row in rows]
    std_values = [_finite(row, "std", "summary source") for row in rows]
    standardized = []
    coverage = {1: 0, 2: 0, 3: 0}
    for residual, std in zip(residuals, std_values):
        for multiplier in coverage:
            if abs(residual) <= multiplier * std:
                coverage[multiplier] += 1
        if std > 0.0:
            standardized.append(residual / std)
        elif residual == 0.0:
            standardized.append(0.0)
    count = len(rows)
    return {
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
        if components != num_atoms * 3:
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
    for target, rows in (("energy", energy_rows), ("forces", force_rows)):
        _require_keys(
            document[target], _TARGET_SUMMARY_FIELDS, f"{source}.{target}"
        )
        _require_equivalent(
            document[target], _target_summary(rows), f"{source}.{target}"
        )
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
    if not isinstance(value, Mapping) or set(value) not in (
        fields,
        fields | {"curvature_artifact"},
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


def _trusted_progress_identity(
    root: Path, explicit_identity: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    progress_path = root / "progress.pt"
    if not progress_path.is_file():
        raise ValueError("trusted evaluation progress identity is missing")
    progress = load_torch_artifact(progress_path)
    if not isinstance(progress, Mapping) or progress.get("status") != "complete":
        raise ValueError("trusted evaluation progress identity requires complete progress")
    identity = progress.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("trusted evaluation progress identity must be a mapping")
    if explicit_identity is not None and _strict_json_bytes(identity) != _strict_json_bytes(
        explicit_identity
    ):
        raise ValueError("trusted evaluation progress identity mismatch")

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
    if set(identity) != required:
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
    _validate_calibration_records(
        calibration["records"], calibration_identity["ridge"], "calibration.records"
    )
    test = _validate_dataset_identity(identity["test"], "test")
    ridge = _validate_ridge_identity(identity["ridge"], "ridge", selected=False)
    min_q = _identity_number(
        identity["min_q"], "min_q", minimum=0.0, strict=True
    )
    if min_q != 1.0e-30:
        raise ValueError("trusted evaluation progress identity min_q is invalid")
    limits = _validate_limits_identity(identity["limits"], "limits")
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
    return identity


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
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    source = _trusted_progress_identity(root, explicit_identity)

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

    if source is None:
        return (
            dict(existing)
            if existing is not None
            else {
                "checkpoint": None,
                "data": None,
                "readout": None,
                "config": None,
            }
        )
    normalized = _normalise_identities(source)
    if existing is not None and _strict_json_bytes(existing) != _strict_json_bytes(
        normalized
    ):
        raise ValueError("manifest identity mismatch")
    return normalized, source


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


def validate_publication_root(
    publication_root: Path,
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate canonical LLPR results and atomically refresh their reports.

    Statistical quality metrics are diagnostics only. Structural, identity,
    hash, and numerical-integrity violations raise ``ValueError``.
    """
    root = Path(publication_root)
    if not root.is_dir():
        raise ValueError(f"publication root is not a directory: {root}")
    identities, trusted_identity = _publication_identities(root, identity)
    calibration_records = trusted_identity["calibration"]["records"]
    manifest_path = root / "manifest.json"
    manifest_exists = manifest_path.is_file()
    if manifest_exists:
        _validate_existing_manifest(root, identities)

    diagnostics: dict[str, Any] = {}
    energy_baseline: list[tuple[str, int, float, float, float]] | None = None
    force_baseline: list[tuple[str, int, int, int, float, float, float]] | None = None
    for variant in _VARIANTS:
        energy, energy_residuals, energy_std = _validate_energy(root, variant)
        forces, force_residuals, force_std, force_rows = _validate_forces(
            root, variant, energy
        )
        energy_rows = _read_csv(root / variant / "energy.csv", ENERGY_FIELDS)
        force_structure = _validate_force_structures(
            root, variant, energy, force_rows
        )
        summary = _validate_summary(
            root, variant, energy_rows, force_rows, force_structure
        )
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
        diagnostics[variant] = {
            "energy": _quality_diagnostics(energy_residuals, energy_std),
            "forces": _quality_diagnostics(force_residuals, force_std),
        }

    file_hashes = _canonical_file_hashes(root)
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
    if not manifest_exists:
        _atomic_strict_json_dump(manifest_path, manifest)
    report = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "status": "valid",
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "tolerances": {
            "relative": RELATIVE_TOLERANCE,
            "absolute": ABSOLUTE_TOLERANCE,
        },
        "diagnostics": diagnostics,
    }
    _atomic_strict_json_dump(root / "validation.json", report)
    return report


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
