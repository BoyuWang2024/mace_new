"""Strict, source-independent configuration for the publishable FGE workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, cast

import yaml

from .errors import ConfigError


RISK_COVERAGES = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1)
_PATH_KEYS = frozenset(
    {"base_checkpoint", "train_data", "val_data", "test_data", "output_root"}
)
_FORBIDDEN_PUBLICATION_ROOTS = frozenset({"internal_migration", "outputs"})
_SCHEMA: dict[str, frozenset[str] | None] = {
    "schema_version": None,
    "project": frozenset({"name", "method", "backend"}),
    "paths": _PATH_KEYS,
    "data": frozenset(
        {"format", "energy_key", "forces_key", "stress_key", "head_name"}
    ),
    "training": frozenset(
        {
            "mode",
            "trainable_scope",
            "expected_readout_parameter_count",
            "seed",
            "member_count",
            "epochs_per_cycle",
            "batch_size",
            "validation_batch_size",
            "weight_decay",
            "energy_weight",
            "forces_weight",
            "stress_weight",
            "huber_delta",
            "beta",
            "amsgrad",
            "max_grad_norm",
            "device",
        }
    ),
    "ema": frozenset({"enabled", "mode", "decay", "main_member_source"}),
    "fge": frozenset({"schedule", "rise_fraction", "lr_min", "lr_max"}),
    "prediction": frozenset(
        {"split", "member_source", "batch_size", "compute_stress"}
    ),
    "ensemble": frozenset(
        {
            "equal_weight",
            "validation_error_weighted",
            "eps_energy_ratio",
            "eps_force_ratio",
        }
    ),
    "evaluation": frozenset({"risk_coverages", "force_structure_quantile"}),
    "quality": frozenset({"weak_rmse_multiplier", "collapsed_rmse_multiplier"}),
    "wandb": frozenset({"enabled", "project", "entity", "mode"}),
    "smoke": frozenset(
        {
            "enabled",
            "allow_identical_splits",
            "max_train_structures",
            "max_validation_structures",
            "max_test_structures",
            "max_train_batches",
            "max_validation_batches",
        }
    ),
}


def _fail(path: str, message: str) -> None:
    raise ConfigError(f"{path}: {message}")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be a mapping")
    return cast(Mapping[str, Any], value)


def _require_number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number")
    result = float(value)
    if positive and result <= 0.0:
        _fail(path, "must be positive")
    return result


def _require_positive_int(value: Any, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _require_literal(value: Any, expected: Any, path: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(path, f"must be {expected!r}")


def _validate_keys(raw: Mapping[str, Any]) -> None:
    expected_top = set(_SCHEMA)
    actual_top = set(raw)
    for name in sorted(actual_top - expected_top):
        _fail(name, "unknown configuration key")
    for name in sorted(expected_top - actual_top):
        _fail(name, "missing configuration key")
    for section, allowed in _SCHEMA.items():
        if allowed is None:
            continue
        values = _require_mapping(raw[section], section)
        for name in sorted(set(values) - set(allowed)):
            _fail(f"{section}.{name}", "unknown configuration key")
        for name in sorted(set(allowed) - set(values)):
            _fail(f"{section}.{name}", "missing configuration key")


def validate_config_dict(raw: Mapping[str, Any]) -> None:
    """Validate every supported key and the fixed public FGE protocol."""

    raw = _require_mapping(raw, "config")
    _validate_keys(raw)
    _require_literal(raw["schema_version"], "fge.v1", "schema_version")

    project = _require_mapping(raw["project"], "project")
    for key in ("name", "method", "backend"):
        if not isinstance(project[key], str) or not project[key].strip():
            _fail(f"project.{key}", "must be a nonempty string")
    _require_literal(project["method"], "FGE", "project.method")
    _require_literal(project["backend"], "mace", "project.backend")

    paths = _require_mapping(raw["paths"], "paths")
    for key in _PATH_KEYS:
        if not isinstance(paths[key], str) or not paths[key].strip():
            _fail(f"paths.{key}", "must be a nonempty path string")

    data = _require_mapping(raw["data"], "data")
    _require_literal(data["format"], "extxyz", "data.format")
    for key in ("energy_key", "forces_key", "stress_key", "head_name"):
        if not isinstance(data[key], str) or not data[key].strip():
            _fail(f"data.{key}", "must be a nonempty string")

    training = _require_mapping(raw["training"], "training")
    _require_literal(
        training["mode"], "readout_only_official_mace", "training.mode"
    )
    _require_literal(training["trainable_scope"], "readouts", "training.trainable_scope")
    _require_literal(
        training["expected_readout_parameter_count"],
        2192,
        "training.expected_readout_parameter_count",
    )
    _require_positive_int(training["member_count"], "training.member_count", minimum=2)
    for key in ("epochs_per_cycle", "batch_size", "validation_batch_size"):
        _require_positive_int(training[key], f"training.{key}")
    if isinstance(training["seed"], bool) or not isinstance(training["seed"], int):
        _fail("training.seed", "must be an integer")
    for key in (
        "energy_weight",
        "forces_weight",
        "stress_weight",
        "huber_delta",
        "beta",
        "max_grad_norm",
    ):
        _require_number(training[key], f"training.{key}", positive=True)
    _require_number(training["weight_decay"], "training.weight_decay")
    if not isinstance(training["amsgrad"], bool):
        _fail("training.amsgrad", "must be boolean")
    if training["device"] not in {"cpu", "cuda"}:
        _fail("training.device", "must be 'cpu' or 'cuda'")

    ema = _require_mapping(raw["ema"], "ema")
    _require_literal(ema["enabled"], True, "ema.enabled")
    _require_literal(ema["mode"], "global", "ema.mode")
    _require_literal(ema["main_member_source"], "raw", "ema.main_member_source")
    decay = _require_number(ema["decay"], "ema.decay")
    if not 0.0 <= decay < 1.0:
        _fail("ema.decay", "must be in [0, 1)")

    fge = _require_mapping(raw["fge"], "fge")
    _require_literal(fge["schedule"], "asymmetric_triangular", "fge.schedule")
    lr_min = _require_number(fge["lr_min"], "fge.lr_min", positive=True)
    lr_max = _require_number(fge["lr_max"], "fge.lr_max", positive=True)
    if lr_min >= lr_max:
        _fail("fge.lr_min", "must be less than fge.lr_max")
    rise_fraction = _require_number(fge["rise_fraction"], "fge.rise_fraction")
    if not 0.0 < rise_fraction < 1.0:
        _fail("fge.rise_fraction", "must be in (0, 1)")

    prediction = _require_mapping(raw["prediction"], "prediction")
    _require_literal(prediction["split"], "test", "prediction.split")
    _require_literal(prediction["member_source"], "raw", "prediction.member_source")
    _require_positive_int(prediction["batch_size"], "prediction.batch_size")
    if not isinstance(prediction["compute_stress"], bool):
        _fail("prediction.compute_stress", "must be boolean")

    ensemble = _require_mapping(raw["ensemble"], "ensemble")
    _require_literal(ensemble["equal_weight"], True, "ensemble.equal_weight")
    _require_literal(
        ensemble["validation_error_weighted"],
        True,
        "ensemble.validation_error_weighted",
    )
    for key in ("eps_energy_ratio", "eps_force_ratio"):
        _require_number(ensemble[key], f"ensemble.{key}", positive=True)

    evaluation = _require_mapping(raw["evaluation"], "evaluation")
    coverages = evaluation["risk_coverages"]
    if not isinstance(coverages, (list, tuple)) or tuple(coverages) != RISK_COVERAGES:
        _fail("evaluation.risk_coverages", f"must be exactly {RISK_COVERAGES}")
    quantile = _require_number(
        evaluation["force_structure_quantile"],
        "evaluation.force_structure_quantile",
    )
    if not 0.0 < quantile < 1.0:
        _fail("evaluation.force_structure_quantile", "must be in (0, 1)")

    quality = _require_mapping(raw["quality"], "quality")
    weak = _require_number(
        quality["weak_rmse_multiplier"], "quality.weak_rmse_multiplier", positive=True
    )
    collapsed = _require_number(
        quality["collapsed_rmse_multiplier"],
        "quality.collapsed_rmse_multiplier",
        positive=True,
    )
    if collapsed < weak:
        _fail(
            "quality.collapsed_rmse_multiplier",
            "must be >= quality.weak_rmse_multiplier",
        )

    wandb = _require_mapping(raw["wandb"], "wandb")
    if not isinstance(wandb["enabled"], bool):
        _fail("wandb.enabled", "must be boolean")
    if wandb["mode"] not in {"online", "offline", "disabled"}:
        _fail("wandb.mode", "must be online, offline, or disabled")

    smoke = _require_mapping(raw["smoke"], "smoke")
    for key in ("enabled", "allow_identical_splits"):
        if not isinstance(smoke[key], bool):
            _fail(f"smoke.{key}", "must be boolean")
    if smoke["allow_identical_splits"] and not smoke["enabled"]:
        _fail("smoke.allow_identical_splits", "requires smoke.enabled")
    for key in (
        "max_train_structures",
        "max_validation_structures",
        "max_test_structures",
        "max_train_batches",
        "max_validation_batches",
    ):
        _require_positive_int(smoke[key], f"smoke.{key}")


def _resolve_paths(raw: Mapping[str, Any], base: Path) -> dict[str, Any]:
    resolved = deepcopy(dict(raw))
    paths = dict(cast(Mapping[str, Any], resolved["paths"]))
    for key in _PATH_KEYS:
        path = Path(paths[key]).expanduser()
        paths[key] = str(path.resolve() if path.is_absolute() else (base / path).resolve())
    resolved["paths"] = paths
    return resolved


@dataclass(frozen=True)
class FGEConfig:
    """Validated configuration with defensive access to resolved values."""

    source_path: Path
    _values: dict[str, Any]

    def section(self, name: str) -> Mapping[str, Any]:
        if name not in self._values or not isinstance(self._values[name], Mapping):
            raise ConfigError(f"unknown configuration section: {name}")
        return deepcopy(cast(dict[str, Any], self._values[name]))

    @property
    def project_name(self) -> str:
        return str(cast(Mapping[str, Any], self._values["project"])["name"])

    @property
    def output_dir(self) -> Path:
        return Path(cast(Mapping[str, Any], self._values["paths"])["output_root"])

    def to_resolved_dict(self) -> dict[str, Any]:
        return deepcopy(self._values)


def load_config(path: Path) -> FGEConfig:
    """Load one UTF-8 YAML file and resolve only its configured filesystem paths."""

    source = Path(path).resolve()
    if source.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigError("configuration source must be a YAML file")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"unable to load configuration YAML: {exc}") from exc
    mapping = _require_mapping(raw, "config")
    validate_config_dict(mapping)
    return FGEConfig(source_path=source, _values=_resolve_paths(mapping, source.parent))


def load_publication_allowlist(path: Path) -> tuple[str, ...]:
    """Parse safe, relative publication roots while excluding internal outputs."""

    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"unable to read publication allowlist: {exc}") from exc
    entries: list[str] = []
    for raw_line in lines:
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue
        normalized = entry[:-1] if entry.endswith("/") else entry
        posix = PurePosixPath(normalized)
        if (
            posix.is_absolute()
            or ".." in posix.parts
            or "\\" in entry
            or not posix.parts
            or posix.parts[0] in _FORBIDDEN_PUBLICATION_ROOTS
        ):
            raise ConfigError(f"unsafe publication allowlist entry: {entry}")
        entries.append(entry)
    if len(entries) != len(set(entries)):
        raise ConfigError("publication allowlist contains duplicate entries")
    return tuple(entries)
