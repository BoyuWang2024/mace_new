"""Immutable configuration and schema contracts for E0 postprocessing."""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from Uncertainty_Quantification.FGE.fge.errors import HardFailure


E0_RUN_SCHEMA_VERSION = "fge.e0-run.v1"
CALIBRATION_SCHEMA_VERSION = "fge.e0-calibration.v1"
CORRECTION_SCHEMA_VERSION = "fge.e0-correction.v1"
INTEGRITY_SCHEMA_VERSION = "fge.e0-integrity.v1"

_METHOD_SPECS = {
    "direct_test_e0": (
        "Direct test-informed E0",
        "test",
        True,
        "transductive_diagnostic",
    ),
    "model_aware_val_e0": (
        "Model-aware val-calibrated E0",
        "val",
        False,
        "calibrated_test",
    ),
}
_LOGICAL_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_DATASET_KEYS = frozenset(
    {
        "test_label",
        "test_path",
        "val_label",
        "val_path",
        "energy",
        "forces",
        "atomization_energy",
        "head",
        "stress",
        "test_forward",
    }
)
_EXPERIMENT_KEYS = frozenset({"label", "config"})
_RUN_KEYS = frozenset({"schema_version", "dataset", "experiments", "methods"})


def _fail(path: str, message: str) -> None:
    raise HardFailure(f"{path}: {message}")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be a mapping")
    if any(not isinstance(key, str) for key in value):
        _fail(path, "must contain only string keys")
    return value


def _require_exact_keys(
    value: Any, expected: frozenset[str], path: str
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, path)
    actual = set(mapping)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        details = []
        if unknown:
            details.append(f"unknown keys {unknown}")
        if missing:
            details.append(f"missing keys {missing}")
        _fail(path, "; ".join(details))
    return mapping


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        _fail(path, "must be a nonempty, trimmed string")
    return value


def _require_logical_label(value: Any, path: str) -> str:
    label = _require_nonempty_string(value, path)
    if label in {".", ".."} or _LOGICAL_LABEL.fullmatch(label) is None:
        _fail(path, "must be a safe logical label")
    return label


def _require_logical_path(value: Any, path: str) -> str:
    raw = _require_nonempty_string(value, path)
    if "\\" in raw:
        _fail(path, "must use a safe relative POSIX path")
    segments = raw.split("/")
    posix = PurePosixPath(raw)
    if (
        posix.is_absolute()
        or any(segment in {"", ".", ".."} for segment in segments)
        or (segments and ":" in segments[0])
    ):
        _fail(path, "must use a safe relative POSIX path")
    return raw


def _require_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(path, "must be a sequence")
    return value


def _require_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _require_finite_number(value: Any, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        _fail(path, f"must be finite and >= {minimum}")
    return result


@dataclass(frozen=True)
class CorrectionMethod:
    """One supported correction method with inseparable provenance metadata."""

    machine_name: str
    title: str
    calibration_split: str
    uses_test_reference_labels: bool
    evaluation_role: str

    def __post_init__(self) -> None:
        expected = _METHOD_SPECS.get(self.machine_name)
        actual = (
            self.title,
            self.calibration_split,
            self.uses_test_reference_labels,
            self.evaluation_role,
        )
        if expected is None or actual != expected:
            _fail("method", "unsupported method or metadata")

    @classmethod
    def from_name(cls, name: str) -> "CorrectionMethod":
        if not isinstance(name, str) or name not in _METHOD_SPECS:
            _fail("method", f"unsupported correction method {name!r}")
        return cls(name, *_METHOD_SPECS[name])


@dataclass(frozen=True)
class E0DatasetConfig:
    """Logical MAD dataset inputs and their explicit reference fields."""

    test_label: str
    test_path: str
    val_label: str
    val_path: str
    energy: str
    forces: str
    atomization_energy: str
    head: str
    stress: bool
    test_forward: bool

    def __post_init__(self) -> None:
        _require_logical_label(self.test_label, "dataset.test_label")
        _require_logical_path(self.test_path, "dataset.test_path")
        _require_logical_label(self.val_label, "dataset.val_label")
        _require_logical_path(self.val_path, "dataset.val_path")
        for name in ("energy", "forces", "atomization_energy", "head"):
            _require_nonempty_string(getattr(self, name), f"dataset.{name}")
        if self.test_label == self.val_label:
            _fail("dataset", "test and val logical labels must differ")
        if self.test_path == self.val_path:
            _fail("dataset", "test and val logical paths must differ")
        if self.stress is not False:
            _fail("dataset.stress", "stress must be disabled")
        if self.test_forward is not False:
            _fail("dataset.test_forward", "test forward must be disabled")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "E0DatasetConfig":
        values = _require_exact_keys(raw, _DATASET_KEYS, "dataset")
        return cls(
            test_label=values["test_label"],
            test_path=values["test_path"],
            val_label=values["val_label"],
            val_path=values["val_path"],
            energy=values["energy"],
            forces=values["forces"],
            atomization_energy=values["atomization_energy"],
            head=values["head"],
            stress=values["stress"],
            test_forward=values["test_forward"],
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """One logical experiment label mapped to one relative YAML config."""

    label: str
    config: str

    def __post_init__(self) -> None:
        _require_logical_label(self.label, "experiment.label")
        config = _require_logical_path(self.config, "experiment.config")
        if PurePosixPath(config).suffix.lower() not in {".yaml", ".yml"}:
            _fail("experiment.config", "must identify a YAML configuration")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        values = _require_exact_keys(raw, _EXPERIMENT_KEYS, "experiment")
        return cls(label=values["label"], config=values["config"])


@dataclass(frozen=True)
class E0RunConfig:
    """The fixed four-experiment, two-method E0 postprocessing protocol."""

    schema_version: str
    dataset: E0DatasetConfig
    experiments: tuple[ExperimentConfig, ...]
    methods: tuple[CorrectionMethod, ...]

    def __post_init__(self) -> None:
        if self.schema_version != E0_RUN_SCHEMA_VERSION:
            _fail("schema_version", f"must be {E0_RUN_SCHEMA_VERSION!r}")
        if not isinstance(self.dataset, E0DatasetConfig):
            _fail("dataset", "must be an E0DatasetConfig")
        if not isinstance(self.experiments, tuple) or len(self.experiments) != 4:
            _fail("experiments", "must contain exactly four experiments")
        expected_labels = tuple(
            f"experiment_{index:02d}" for index in range(1, 5)
        )
        labels = tuple(experiment.label for experiment in self.experiments)
        configs = tuple(experiment.config for experiment in self.experiments)
        if labels != expected_labels or len(set(labels)) != 4:
            _fail("experiments", "labels must be unique and contiguous")
        if len(set(configs)) != 4:
            _fail("experiments", "configs must map one-to-one to labels")
        if not isinstance(self.methods, tuple):
            _fail("methods", "must be an immutable sequence")
        names = tuple(method.machine_name for method in self.methods)
        if names != tuple(_METHOD_SPECS):
            _fail("methods", "must contain each supported method exactly once")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "E0RunConfig":
        values = _require_exact_keys(raw, _RUN_KEYS, "run")
        experiments_raw = _require_sequence(values["experiments"], "experiments")
        methods_raw = _require_sequence(values["methods"], "methods")
        experiments = tuple(
            ExperimentConfig.from_dict(item) for item in experiments_raw
        )
        methods = tuple(CorrectionMethod.from_name(item) for item in methods_raw)
        return cls(
            schema_version=values["schema_version"],
            dataset=E0DatasetConfig.from_dict(values["dataset"]),
            experiments=experiments,
            methods=methods,
        )


def _require_any_finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "must be a finite number")
    return result


@dataclass(frozen=True)
class CalibrationFit:
    """Per-member fitted E0 offsets and numerical diagnostics."""

    member_id: str
    atomic_numbers: tuple[int, ...]
    delta_e0: tuple[float, ...]
    rank: int
    residual_rmse: float
    condition_number: float

    def __post_init__(self) -> None:
        _require_logical_label(self.member_id, "fit.member_id")
        if (
            not isinstance(self.atomic_numbers, tuple)
            or not isinstance(self.delta_e0, tuple)
            or not self.atomic_numbers
            or len(self.atomic_numbers) != len(self.delta_e0)
        ):
            _fail("fit", "atomic numbers and delta_e0 must be aligned tuples")
        for index, atomic_number in enumerate(self.atomic_numbers):
            _require_int(atomic_number, f"fit.atomic_numbers[{index}]", minimum=1)
        if len(set(self.atomic_numbers)) != len(self.atomic_numbers):
            _fail("fit.atomic_numbers", "must be unique")
        for index, value in enumerate(self.delta_e0):
            _require_any_finite_number(value, f"fit.delta_e0[{index}]")
        rank = _require_int(self.rank, "fit.rank", minimum=1)
        if rank > len(self.atomic_numbers):
            _fail("fit.rank", "cannot exceed the number of element columns")
        _require_finite_number(self.residual_rmse, "fit.residual_rmse")
        _require_finite_number(self.condition_number, "fit.condition_number")


@dataclass(frozen=True)
class CorrectionSummary:
    """Before/after energy accuracy for one corrected experiment."""

    structure_count: int
    atom_count: int
    energy_rmse_before: float
    energy_rmse_after: float

    def __post_init__(self) -> None:
        _require_int(self.structure_count, "summary.structure_count", minimum=1)
        _require_int(self.atom_count, "summary.atom_count", minimum=1)
        _require_finite_number(
            self.energy_rmse_before, "summary.energy_rmse_before"
        )
        _require_finite_number(self.energy_rmse_after, "summary.energy_rmse_after")


@dataclass(frozen=True)
class WarningRecord:
    """A stable non-fatal diagnostic."""

    code: str
    message: str

    def __post_init__(self) -> None:
        _require_logical_label(self.code, "warning.code")
        _require_nonempty_string(self.message, "warning.message")


_CALIBRATION_KEYS = frozenset(
    {
        "schema_version",
        "split",
        "member_ids",
        "structure_ids",
        "atomic_numbers",
        "composition",
        "energy_members",
        "energy_reference",
        "n_atoms",
    }
)
_CORRECTION_KEYS = frozenset(
    {
        "schema_version",
        "method",
        "title",
        "calibration_split",
        "application_split",
        "uses_test_reference_labels",
        "evaluation_role",
        "validation_decontamination",
        "overlap_removed_count",
        "experiment_label",
        "dataset_label",
        "member_ids",
        "observables",
        "shape",
        "structure_range",
        "calibration",
        "warnings",
    }
)
_INTEGRITY_KEYS = frozenset(
    {
        "schema_version",
        "excluded_structure_ids",
        "input_sha256",
        "raw_artifact_sha256",
        "output_sha256",
        "remote_paths",
        "runtime",
    }
)
_SHAPE_KEYS = frozenset({"K", "S", "A"})
_RANGE_KEYS = frozenset({"start", "stop"})
_CALIBRATION_SUMMARY_KEYS = frozenset(
    {"rank_min", "rank_max", "residual_rmse_max"}
)
_WARNING_KEYS = frozenset({"code", "message"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.IGNORECASE)


def _require_unique_strings(value: Any, path: str) -> tuple[str, ...]:
    sequence = _require_sequence(value, path)
    items = tuple(
        _require_logical_label(item, f"{path}[{index}]")
        for index, item in enumerate(sequence)
    )
    if not items or len(items) != len(set(items)):
        _fail(path, "must contain unique logical labels")
    return items


def _require_hash_mapping(value: Any, path: str) -> Mapping[str, Any]:
    mapping = _require_mapping(value, path)
    if not mapping:
        _fail(path, "must not be empty")
    for key, digest in mapping.items():
        _require_logical_label(key, f"{path}.key")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            _fail(f"{path}.{key}", "must be a SHA-256 hex digest")
    return mapping


def _reject_public_source_leaks(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        mapping = _require_mapping(value, path)
        for key, item in mapping.items():
            if "sha256" in key.lower():
                _fail(f"{path}.{key}", "published manifests cannot contain hashes")
            _reject_public_source_leaks(item, f"{path}.{key}")
        return
    if isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        segments = tuple(part for part in normalized.split("/") if part)
        is_absolute = normalized.startswith("/") or re.match(
            r"^[a-z]:/", normalized
        )
        is_legacy = "mace/ensemble" in normalized
        is_raw_path = "/" in normalized and any(
            part in {"raw", "checkpoint", "checkpoints"} for part in segments
        )
        if is_absolute or is_legacy or is_raw_path or _SHA256.fullmatch(value):
            _fail(path, "published manifest contains source-specific data")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_public_source_leaks(item, f"{path}[{index}]")


def validate_calibration_manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact val-only energy calibration payload."""

    values = _require_exact_keys(raw, _CALIBRATION_KEYS, "calibration_manifest")
    if values["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        _fail("calibration_manifest.schema_version", "unsupported schema")
    if values["split"] != "val":
        _fail("calibration_manifest.split", "must be 'val'")
    member_ids = _require_unique_strings(values["member_ids"], "member_ids")
    structure_ids = _require_unique_strings(values["structure_ids"], "structure_ids")
    atomic_raw = _require_sequence(values["atomic_numbers"], "atomic_numbers")
    atomic_numbers = tuple(
        _require_int(item, f"atomic_numbers[{index}]", minimum=1)
        for index, item in enumerate(atomic_raw)
    )
    if not atomic_numbers or len(atomic_numbers) != len(set(atomic_numbers)):
        _fail("atomic_numbers", "must contain unique positive integers")
    n_atoms_raw = _require_sequence(values["n_atoms"], "n_atoms")
    n_atoms = tuple(
        _require_int(item, f"n_atoms[{index}]", minimum=1)
        for index, item in enumerate(n_atoms_raw)
    )
    if len(n_atoms) != len(structure_ids):
        _fail("n_atoms", "must align with structure_ids")
    composition = _require_sequence(values["composition"], "composition")
    if len(composition) != len(structure_ids):
        _fail("composition", "must have one row per structure")
    for index, (raw_row, atom_count) in enumerate(zip(composition, n_atoms)):
        row = _require_sequence(raw_row, f"composition[{index}]")
        counts = tuple(
            _require_int(item, f"composition[{index}][{column}]")
            for column, item in enumerate(row)
        )
        if len(counts) != len(atomic_numbers) or sum(counts) != atom_count:
            _fail(f"composition[{index}]", "must align with elements and n_atoms")
    energy_members = _require_sequence(values["energy_members"], "energy_members")
    if len(energy_members) != len(member_ids):
        _fail("energy_members", "must have one row per member")
    for member, raw_row in enumerate(energy_members):
        row = _require_sequence(raw_row, f"energy_members[{member}]")
        if len(row) != len(structure_ids):
            _fail(f"energy_members[{member}]", "must align with structures")
        for index, item in enumerate(row):
            _require_any_finite_number(item, f"energy_members[{member}][{index}]")
    reference = _require_sequence(values["energy_reference"], "energy_reference")
    if len(reference) != len(structure_ids):
        _fail("energy_reference", "must align with structures")
    for index, item in enumerate(reference):
        _require_any_finite_number(item, f"energy_reference[{index}]")
    return deepcopy(dict(values))


def validate_correction_manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one source-neutral published correction manifest."""

    _reject_public_source_leaks(_require_mapping(raw, "correction_manifest"))
    values = _require_exact_keys(raw, _CORRECTION_KEYS, "correction_manifest")
    if values["schema_version"] != CORRECTION_SCHEMA_VERSION:
        _fail("correction_manifest.schema_version", "unsupported schema")
    method = CorrectionMethod.from_name(values["method"])
    metadata = (
        values["title"],
        values["calibration_split"],
        values["evaluation_role"],
    )
    expected = (method.title, method.calibration_split, method.evaluation_role)
    if (
        metadata != expected
        or values["uses_test_reference_labels"]
        is not method.uses_test_reference_labels
    ):
        _fail("correction_manifest.method", "method metadata drifted")
    if values["application_split"] != "test":
        _fail("correction_manifest.application_split", "must be 'test'")
    if (
        values["validation_decontamination"]
        != "exclude_test_identity_overlap_from_val"
    ):
        _fail("correction_manifest.validation_decontamination", "unsupported rule")
    _require_int(values["overlap_removed_count"], "overlap_removed_count")
    _require_logical_label(values["experiment_label"], "experiment_label")
    _require_logical_label(values["dataset_label"], "dataset_label")
    member_ids = _require_unique_strings(values["member_ids"], "member_ids")
    if tuple(_require_sequence(values["observables"], "observables")) != (
        "energy",
        "forces",
    ):
        _fail("observables", "must be exactly energy and forces")
    shape = _require_exact_keys(values["shape"], _SHAPE_KEYS, "shape")
    k = _require_int(shape["K"], "shape.K", minimum=1)
    s = _require_int(shape["S"], "shape.S", minimum=1)
    _require_int(shape["A"], "shape.A", minimum=1)
    if k != len(member_ids):
        _fail("shape.K", "must match member_ids")
    structure_range = _require_exact_keys(
        values["structure_range"], _RANGE_KEYS, "structure_range"
    )
    start = _require_int(structure_range["start"], "structure_range.start")
    stop = _require_int(structure_range["stop"], "structure_range.stop", minimum=1)
    if stop <= start or stop - start != s:
        _fail("structure_range", "must be increasing and match shape.S")
    calibration = _require_exact_keys(
        values["calibration"], _CALIBRATION_SUMMARY_KEYS, "calibration"
    )
    rank_min = _require_int(calibration["rank_min"], "calibration.rank_min", minimum=1)
    rank_max = _require_int(calibration["rank_max"], "calibration.rank_max", minimum=1)
    if rank_min > rank_max:
        _fail("calibration", "rank_min cannot exceed rank_max")
    _require_finite_number(
        calibration["residual_rmse_max"], "calibration.residual_rmse_max"
    )
    warnings = _require_sequence(values["warnings"], "warnings")
    for index, warning in enumerate(warnings):
        item = _require_exact_keys(warning, _WARNING_KEYS, f"warnings[{index}]")
        WarningRecord(code=item["code"], message=item["message"])
    return deepcopy(dict(values))


def validate_integrity_audit(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the separate internal-only integrity audit."""

    values = _require_exact_keys(raw, _INTEGRITY_KEYS, "integrity_audit")
    if values["schema_version"] != INTEGRITY_SCHEMA_VERSION:
        _fail("integrity_audit.schema_version", "unsupported schema")
    excluded_raw = _require_sequence(
        values["excluded_structure_ids"], "excluded_structure_ids"
    )
    excluded = tuple(
        _require_logical_label(item, f"excluded_structure_ids[{index}]")
        for index, item in enumerate(excluded_raw)
    )
    if len(excluded) != len(set(excluded)):
        _fail("excluded_structure_ids", "must be unique")
    for name in ("input_sha256", "raw_artifact_sha256", "output_sha256"):
        _require_hash_mapping(values[name], name)
    remote_paths = _require_mapping(values["remote_paths"], "remote_paths")
    if not remote_paths:
        _fail("remote_paths", "must not be empty")
    for key, path in remote_paths.items():
        _require_logical_label(key, "remote_paths.key")
        _require_nonempty_string(path, f"remote_paths.{key}")
    runtime = _require_mapping(values["runtime"], "runtime")
    if not runtime:
        _fail("runtime", "must not be empty")
    return deepcopy(dict(values))


def load_correction_manifest(path: Path) -> dict[str, Any]:
    """Read only the public correction schema; integrity audits are rejected."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HardFailure(f"unable to read correction manifest: {exc}") from exc
    return validate_correction_manifest(raw)
