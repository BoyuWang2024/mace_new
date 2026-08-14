"""Strict, path-aware configuration loading for MACE BootStrapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from .errors import HardFailure


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    run_id: str
    output_root: Path


@dataclass(frozen=True)
class CheckpointConfig:
    base_path: Path


@dataclass(frozen=True)
class DataUnits:
    energy: str
    forces: str
    stress: str


@dataclass(frozen=True)
class DataConfig:
    train: Path
    val: Path
    test: Path
    units: DataUnits


@dataclass(frozen=True)
class BootstrapSettings:
    ensemble_size: int
    base_seed: int
    sample_size: int | None
    replacement: bool
    save_indices: bool
    save_oob: bool


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    max_epochs: int
    trainable_head: str
    optimizer: OptimizerConfig
    scheduler: str
    gradient_clip: float
    ema_decay: float
    device: str
    precision: str
    num_workers: int


@dataclass(frozen=True)
class PredictionConfig:
    splits: tuple[str, ...]
    parameter_modes: tuple[str, ...]
    batch_size: int
    structure_chunk_size: int
    device: str
    num_workers: int


@dataclass(frozen=True)
class UncertaintyConfig:
    parameter_modes: tuple[str, ...]
    ddof: int
    compute_std: bool
    compute_gmd: bool
    gmd_pairs: str


@dataclass(frozen=True)
class BootstrapConfig:
    schema_version: int
    experiment: ExperimentConfig
    checkpoint: CheckpointConfig
    data: DataConfig
    bootstrap: BootstrapSettings
    training: TrainingConfig
    prediction: PredictionConfig
    uncertainty: UncertaintyConfig
    source_path: Path


T = TypeVar("T")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HardFailure(f"{location} must be a mapping")
    return value


def _keys(
    value: Any,
    location: str,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    result = _mapping(value, location)
    unknown = set(result) - required - (optional or set())
    if unknown:
        raise HardFailure(f"unknown key {location}.{sorted(unknown)[0]}")
    missing = required - set(result)
    if missing:
        raise HardFailure(f"missing key {location}.{sorted(missing)[0]}")
    return result


def _typed(value: Any, expected: type[T], location: str) -> T:
    if expected is int and isinstance(value, bool):
        raise HardFailure(f"{location} must be an integer")
    if expected is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)  # type: ignore[return-value]
    if not isinstance(value, expected):
        raise HardFailure(f"{location} must be {expected.__name__}")
    return value


def _positive_int(value: Any, location: str) -> int:
    result = _typed(value, int, location)
    if result < 1:
        raise HardFailure(f"{location} must be at least 1")
    return result


def _nonnegative_int(value: Any, location: str) -> int:
    result = _typed(value, int, location)
    if result < 0:
        raise HardFailure(f"{location} must be non-negative")
    return result


def _path(value: Any, base: Path, location: str) -> Path:
    path = Path(_typed(value, str, location)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _choice(value: Any, allowed: set[str], location: str) -> str:
    result = _typed(value, str, location)
    if result not in allowed:
        raise HardFailure(f"{location} must be one of: {', '.join(sorted(allowed))}")
    return result


def _choices(value: Any, allowed: set[str], location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise HardFailure(f"{location} must be a non-empty list")
    result = tuple(_choice(item, allowed, location) for item in value)
    if len(set(result)) != len(result):
        raise HardFailure(f"{location} must not contain duplicates")
    return result


def load_config(source: str | Path) -> BootstrapConfig:
    """Load version-one YAML, resolving paths relative to the YAML file."""

    source_path = Path(source).expanduser().resolve()
    try:
        document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise HardFailure(f"could not load config {source_path}: {error}") from error

    root = _keys(
        document,
        "config",
        {
            "schema_version", "experiment", "checkpoint", "data", "bootstrap",
            "training", "prediction", "uncertainty",
        },
    )
    schema_version = _typed(root["schema_version"], int, "schema_version")
    if schema_version != 1:
        raise HardFailure("schema_version must be 1")
    base = source_path.parent

    experiment_map = _keys(root["experiment"], "experiment", {"name", "run_id", "output_root"})
    experiment = ExperimentConfig(
        name=_typed(experiment_map["name"], str, "experiment.name"),
        run_id=_typed(experiment_map["run_id"], str, "experiment.run_id"),
        output_root=_path(experiment_map["output_root"], base, "experiment.output_root"),
    )
    checkpoint_map = _keys(root["checkpoint"], "checkpoint", {"base_path"})
    checkpoint = CheckpointConfig(
        base_path=_path(checkpoint_map["base_path"], base, "checkpoint.base_path")
    )

    data_map = _keys(root["data"], "data", {"train", "val", "test", "units"})
    units_map = _keys(data_map["units"], "data.units", {"energy", "forces", "stress"})
    units = DataUnits(
        energy=_choice(units_map["energy"], {"eV"}, "data.units.energy"),
        forces=_choice(units_map["forces"], {"eV/Angstrom"}, "data.units.forces"),
        stress=_choice(units_map["stress"], {"eV/Angstrom^3"}, "data.units.stress"),
    )
    data = DataConfig(
        train=_path(data_map["train"], base, "data.train"),
        val=_path(data_map["val"], base, "data.val"),
        test=_path(data_map["test"], base, "data.test"),
        units=units,
    )

    bootstrap_map = _keys(
        root["bootstrap"], "bootstrap",
        {"ensemble_size", "base_seed", "sample_size", "replacement", "save_indices", "save_oob"},
    )
    sample_value = bootstrap_map["sample_size"]
    bootstrap = BootstrapSettings(
        ensemble_size=_positive_int(bootstrap_map["ensemble_size"], "bootstrap.ensemble_size"),
        base_seed=_nonnegative_int(bootstrap_map["base_seed"], "bootstrap.base_seed"),
        sample_size=None if sample_value is None else _positive_int(sample_value, "bootstrap.sample_size"),
        replacement=_typed(bootstrap_map["replacement"], bool, "bootstrap.replacement"),
        save_indices=_typed(bootstrap_map["save_indices"], bool, "bootstrap.save_indices"),
        save_oob=_typed(bootstrap_map["save_oob"], bool, "bootstrap.save_oob"),
    )
    if bootstrap.ensemble_size < 2:
        raise HardFailure("bootstrap.ensemble_size must be at least 2")
    if not bootstrap.replacement:
        raise HardFailure("bootstrap.replacement must be true")

    training_map = _keys(
        root["training"], "training",
        {"batch_size", "max_epochs", "trainable_head", "optimizer", "scheduler", "gradient_clip", "ema_decay", "device", "precision", "num_workers"},
    )
    optimizer_map = _keys(training_map["optimizer"], "training.optimizer", {"name", "learning_rate", "weight_decay"})
    optimizer = OptimizerConfig(
        name=_choice(optimizer_map["name"], {"Adam"}, "training.optimizer.name"),
        learning_rate=_typed(optimizer_map["learning_rate"], float, "training.optimizer.learning_rate"),
        weight_decay=_typed(optimizer_map["weight_decay"], float, "training.optimizer.weight_decay"),
    )
    if optimizer.learning_rate <= 0:
        raise HardFailure("training.optimizer.learning_rate must be positive")
    if optimizer.weight_decay < 0:
        raise HardFailure("training.optimizer.weight_decay must be non-negative")
    training = TrainingConfig(
        batch_size=_positive_int(training_map["batch_size"], "training.batch_size"),
        max_epochs=_positive_int(training_map["max_epochs"], "training.max_epochs"),
        trainable_head=_choice(training_map["trainable_head"], {"mace_readouts"}, "training.trainable_head"),
        optimizer=optimizer,
        scheduler=_choice(training_map["scheduler"], {"none"}, "training.scheduler"),
        gradient_clip=_typed(training_map["gradient_clip"], float, "training.gradient_clip"),
        ema_decay=_typed(training_map["ema_decay"], float, "training.ema_decay"),
        device=_typed(training_map["device"], str, "training.device"),
        precision=_choice(training_map["precision"], {"float32", "float64"}, "training.precision"),
        num_workers=_nonnegative_int(training_map["num_workers"], "training.num_workers"),
    )
    if training.gradient_clip <= 0:
        raise HardFailure("training.gradient_clip must be positive")
    if not 0 < training.ema_decay < 1:
        raise HardFailure("training.ema_decay must be between 0 and 1")

    prediction_map = _keys(
        root["prediction"], "prediction",
        {"splits", "parameter_modes", "batch_size", "structure_chunk_size", "device", "num_workers"},
    )
    prediction = PredictionConfig(
        splits=_choices(prediction_map["splits"], {"val", "test"}, "prediction.splits"),
        parameter_modes=_choices(prediction_map["parameter_modes"], {"raw", "ema"}, "prediction.parameter_modes"),
        batch_size=_positive_int(prediction_map["batch_size"], "prediction.batch_size"),
        structure_chunk_size=_positive_int(prediction_map["structure_chunk_size"], "prediction.structure_chunk_size"),
        device=_typed(prediction_map["device"], str, "prediction.device"),
        num_workers=_nonnegative_int(prediction_map["num_workers"], "prediction.num_workers"),
    )

    uncertainty_map = _keys(
        root["uncertainty"], "uncertainty",
        {"parameter_modes", "ddof", "compute_std", "compute_gmd", "gmd_pairs"},
    )
    uncertainty = UncertaintyConfig(
        parameter_modes=_choices(uncertainty_map["parameter_modes"], {"raw", "ema"}, "uncertainty.parameter_modes"),
        ddof=_typed(uncertainty_map["ddof"], int, "uncertainty.ddof"),
        compute_std=_typed(uncertainty_map["compute_std"], bool, "uncertainty.compute_std"),
        compute_gmd=_typed(uncertainty_map["compute_gmd"], bool, "uncertainty.compute_gmd"),
        gmd_pairs=_choice(uncertainty_map["gmd_pairs"], {"distinct_unordered"}, "uncertainty.gmd_pairs"),
    )
    if uncertainty.ddof != 1:
        raise HardFailure("uncertainty.ddof must be 1")
    if not set(uncertainty.parameter_modes) <= set(prediction.parameter_modes):
        raise HardFailure("uncertainty.parameter_modes must be included in prediction.parameter_modes")

    return BootstrapConfig(
        schema_version=schema_version,
        experiment=experiment,
        checkpoint=checkpoint,
        data=data,
        bootstrap=bootstrap,
        training=training,
        prediction=prediction,
        uncertainty=uncertainty,
        source_path=source_path,
    )
