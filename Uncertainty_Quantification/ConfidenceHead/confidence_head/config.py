"""Strict YAML configuration parsing for ConfidenceHead training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from .errors import ConfigError


@dataclass(frozen=True)
class FeatureModuleConfig:
    name: str
    expected_dim: int


@dataclass(frozen=True)
class CheckpointConfig:
    path: Path
    expected_sha256: str
    feature_modules: tuple[FeatureModuleConfig, ...]


@dataclass(frozen=True)
class DataSplitConfig:
    path: Path
    expected_sha256: str


@dataclass(frozen=True)
class DataConfig:
    train: DataSplitConfig
    validation: DataSplitConfig
    test: DataSplitConfig


@dataclass(frozen=True)
class CacheConfig:
    build_batch_size: int
    shard_max_atoms: int
    num_workers: int
    pin_memory: bool
    resume: bool


@dataclass(frozen=True)
class FixedBinConfig:
    num_bins: int
    max_error: float


@dataclass(frozen=True)
class LogBinConfig:
    num_bins: int


@dataclass(frozen=True)
class FixedLinearBinningConfig:
    algorithm: Literal["fixed_linear_v1"]
    force: FixedBinConfig
    energy: FixedBinConfig


@dataclass(frozen=True)
class TrainQuantileLogBinningConfig:
    algorithm: Literal["train_quantile_log_v1"]
    force: LogBinConfig
    energy: LogBinConfig


BinConfig = FixedBinConfig | LogBinConfig
BinningConfig = FixedLinearBinningConfig | TrainQuantileLogBinningConfig


@dataclass(frozen=True)
class ForceModelConfig:
    target_mode: Literal["atom_mean", "component"]
    hidden_dims: tuple[int, ...]
    dropout: float


@dataclass(frozen=True)
class EnergyModelConfig:
    cumulant_order: int
    projection_dim: Literal[512]
    adapter_dropout: float
    hidden_dims: tuple[int, ...]
    dropout: float
    signed_root: bool


@dataclass(frozen=True)
class ModelConfig:
    force: ForceModelConfig
    energy: EnergyModelConfig


@dataclass(frozen=True)
class LossConfig:
    force_coefficient: float
    energy_coefficient: float


@dataclass(frozen=True)
class OptimizerConfig:
    name: Literal["adamw"]
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class TrainerConfig:
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    resume: bool


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    device: Literal["cpu", "cuda"]
    deterministic: bool


@dataclass(frozen=True)
class LoggingConfig:
    jsonl: bool
    wandb: bool
    wandb_project: str
    wandb_mode: Literal["auto", "disabled", "offline", "online"]


@dataclass(frozen=True)
class RunConfig:
    name_prefix: str
    output_root: Path


@dataclass(frozen=True)
class ConfidenceHeadConfig:
    source_path: Path
    profile: Literal["production", "smoke_test"]
    checkpoint: CheckpointConfig
    data: DataConfig
    cache: CacheConfig
    binning: BinningConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    runtime: RuntimeConfig
    logging: LoggingConfig
    run: RunConfig

    @property
    def force_enabled(self) -> bool:
        return self.loss.force_coefficient > 0.0

    @property
    def energy_enabled(self) -> bool:
        return self.loss.energy_coefficient > 0.0


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _keys(value: Any, allowed: set[str], field: str) -> Mapping[str, Any]:
    mapping = _mapping(value, field)
    unknown = set(mapping) - allowed
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ConfigError(f"unknown config fields in {field}: {names}")
    missing = allowed - set(mapping)
    if missing:
        raise ConfigError(f"missing config fields in {field}: {', '.join(sorted(missing))}")
    return mapping


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean")
    return value


def _int(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    if value < minimum:
        raise ConfigError(f"{field} must be at least {minimum}")
    return value


def _float(value: Any, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field} must be a finite number")
    if result < minimum:
        raise ConfigError(f"{field} must be at least {minimum}")
    return result


def _positive_float(value: Any, field: str) -> float:
    result = _float(value, field)
    if result <= 0.0:
        raise ConfigError(f"{field} must be greater than zero")
    return result


def _dropout(value: Any, field: str) -> float:
    result = _float(value, field)
    if result >= 1.0:
        raise ConfigError(f"{field} dropout must be in [0, 1)")
    return result


def _path(source_dir: Path, value: Any, field: str, *, input_file: bool) -> Path:
    candidate = Path(_string(value, field))
    resolved = (source_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if input_file and not resolved.is_file():
        raise ConfigError(f"{field} must resolve to an existing file")
    return resolved


def _identity(source_dir: Path, value: Any, field: str) -> DataSplitConfig:
    mapping = _keys(value, {"path", "expected_sha256"}, field)
    sha256 = _string(mapping["expected_sha256"], f"{field}.expected_sha256")
    if len(sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in sha256):
        raise ConfigError(f"{field}.expected_sha256 must be a 64-character SHA-256 hash")
    return DataSplitConfig(_path(source_dir, mapping["path"], f"{field}.path", input_file=True), sha256)


def _dims(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty list of positive integers")
    return tuple(_int(item, field, 1) for item in value)


def _fixed_bin_config(value: Any, field: str) -> FixedBinConfig:
    mapping = _keys(value, {"num_bins", "max_error"}, field)
    return FixedBinConfig(_int(mapping["num_bins"], f"{field}.num_bins", 3), _positive_float(mapping["max_error"], f"{field}.max_error"))


def _log_bin_config(value: Any, field: str) -> LogBinConfig:
    mapping = _keys(value, {"num_bins"}, field)
    return LogBinConfig(_int(mapping["num_bins"], f"{field}.num_bins", 3))


def _checkpoint(source_dir: Path, value: Any) -> CheckpointConfig:
    mapping = _keys(value, {"path", "expected_sha256", "feature_modules"}, "checkpoint")
    identity = _identity(source_dir, {"path": mapping["path"], "expected_sha256": mapping["expected_sha256"]}, "checkpoint")
    raw_features = mapping["feature_modules"]
    if not isinstance(raw_features, list):
        raise ConfigError("checkpoint.feature_modules must be a list")
    features = tuple(
        FeatureModuleConfig(
            _string(_keys(item, {"name", "expected_dim"}, f"checkpoint.feature_modules[{index}]")["name"], f"checkpoint.feature_modules[{index}].name"),
            _int(_keys(item, {"name", "expected_dim"}, f"checkpoint.feature_modules[{index}]")["expected_dim"], f"checkpoint.feature_modules[{index}].expected_dim", 1),
        )
        for index, item in enumerate(raw_features)
    )
    expected = (FeatureModuleConfig("products.0", 512), FeatureModuleConfig("products.1", 128))
    if features != expected:
        raise ConfigError("checkpoint.feature_modules must equal [(products.0, 512), (products.1, 128)]")
    return CheckpointConfig(identity.path, identity.expected_sha256, features)


def _data(source_dir: Path, value: Any) -> DataConfig:
    mapping = _keys(value, {"train", "validation", "test"}, "data")
    return DataConfig(*(_identity(source_dir, mapping[name], f"data.{name}") for name in ("train", "validation", "test")))


def _cache(value: Any) -> CacheConfig:
    mapping = _keys(value, {"build_batch_size", "shard_max_atoms", "num_workers", "pin_memory", "resume"}, "cache")
    return CacheConfig(_int(mapping["build_batch_size"], "cache.build_batch_size", 1), _int(mapping["shard_max_atoms"], "cache.shard_max_atoms", 1), _int(mapping["num_workers"], "cache.num_workers"), _bool(mapping["pin_memory"], "cache.pin_memory"), _bool(mapping["resume"], "cache.resume"))


def _binning(value: Any) -> BinningConfig:
    mapping = _keys(value, {"algorithm", "force", "energy"}, "binning")
    algorithm = _string(mapping["algorithm"], "binning.algorithm")
    if algorithm == "fixed_linear_v1":
        return FixedLinearBinningConfig(
            algorithm,
            _fixed_bin_config(mapping["force"], "binning.force"),
            _fixed_bin_config(mapping["energy"], "binning.energy"),
        )
    if algorithm == "train_quantile_log_v1":
        return TrainQuantileLogBinningConfig(
            algorithm,
            _log_bin_config(mapping["force"], "binning.force"),
            _log_bin_config(mapping["energy"], "binning.energy"),
        )
    raise ConfigError("binning.algorithm must be fixed_linear_v1 or train_quantile_log_v1")


def _model(value: Any) -> ModelConfig:
    mapping = _keys(value, {"force", "energy"}, "model")
    force = _keys(mapping["force"], {"target_mode", "hidden_dims", "dropout"}, "model.force")
    target_mode = _string(force["target_mode"], "model.force.target_mode")
    if target_mode not in {"atom_mean", "component"}:
        raise ConfigError("model.force.target_mode must be atom_mean or component")
    energy = _keys(mapping["energy"], {"cumulant_order", "projection_dim", "adapter_dropout", "hidden_dims", "dropout", "signed_root"}, "model.energy")
    order = _int(energy["cumulant_order"], "model.energy.cumulant_order")
    if not 1 <= order <= 8:
        raise ConfigError("model.energy.cumulant_order must be in 1..8")
    projection_dim = _int(energy["projection_dim"], "model.energy.projection_dim")
    if projection_dim != 512:
        raise ConfigError("model.energy.projection_dim must be 512")
    return ModelConfig(ForceModelConfig(target_mode, _dims(force["hidden_dims"], "model.force.hidden_dims"), _dropout(force["dropout"], "model.force.dropout")), EnergyModelConfig(order, 512, _dropout(energy["adapter_dropout"], "model.energy.adapter_dropout"), _dims(energy["hidden_dims"], "model.energy.hidden_dims"), _dropout(energy["dropout"], "model.energy.dropout"), _bool(energy["signed_root"], "model.energy.signed_root")))


def _loss(value: Any) -> LossConfig:
    mapping = _keys(value, {"force_coefficient", "energy_coefficient"}, "loss")
    force = _float(mapping["force_coefficient"], "loss.force_coefficient")
    energy = _float(mapping["energy_coefficient"], "loss.energy_coefficient")
    if force == 0.0 and energy == 0.0:
        raise ConfigError("loss coefficients cannot both be zero")
    return LossConfig(force, energy)


def _optimizer(value: Any) -> OptimizerConfig:
    mapping = _keys(value, {"name", "learning_rate", "weight_decay"}, "optimizer")
    if _string(mapping["name"], "optimizer.name") != "adamw":
        raise ConfigError("optimizer.name must be adamw")
    return OptimizerConfig("adamw", _positive_float(mapping["learning_rate"], "optimizer.learning_rate"), _float(mapping["weight_decay"], "optimizer.weight_decay"))


def _trainer(value: Any) -> TrainerConfig:
    mapping = _keys(value, {"batch_size", "max_epochs", "early_stopping_patience", "resume"}, "trainer")
    return TrainerConfig(_int(mapping["batch_size"], "trainer.batch_size", 1), _int(mapping["max_epochs"], "trainer.max_epochs", 1), _int(mapping["early_stopping_patience"], "trainer.early_stopping_patience"), _bool(mapping["resume"], "trainer.resume"))


def _runtime(value: Any) -> RuntimeConfig:
    mapping = _keys(value, {"seed", "device", "deterministic"}, "runtime")
    device = _string(mapping["device"], "runtime.device")
    if device not in {"cpu", "cuda"}:
        raise ConfigError("runtime.device must be cpu or cuda")
    return RuntimeConfig(_int(mapping["seed"], "runtime.seed"), device, _bool(mapping["deterministic"], "runtime.deterministic"))


def _logging(value: Any) -> LoggingConfig:
    mapping = _keys(value, {"jsonl", "wandb", "wandb_project", "wandb_mode"}, "logging")
    mode = _string(mapping["wandb_mode"], "logging.wandb_mode")
    if mode not in {"auto", "disabled", "offline", "online"}:
        raise ConfigError("logging.wandb_mode is unsupported")
    return LoggingConfig(_bool(mapping["jsonl"], "logging.jsonl"), _bool(mapping["wandb"], "logging.wandb"), _string(mapping["wandb_project"], "logging.wandb_project"), mode)


def _run(source_dir: Path, value: Any) -> RunConfig:
    mapping = _keys(value, {"name_prefix", "output_root"}, "run")
    return RunConfig(_string(mapping["name_prefix"], "run.name_prefix"), _path(source_dir, mapping["output_root"], "run.output_root", input_file=False))


def load_config(path: Path) -> ConfidenceHeadConfig:
    """Load and fully validate a ConfidenceHead training configuration."""
    source_path = Path(path).resolve()
    try:
        with source_path.open(encoding="utf-8") as handle:
            document = _keys(yaml.safe_load(handle), {"profile", "checkpoint", "data", "cache", "binning", "model", "loss", "optimizer", "trainer", "runtime", "logging", "run"}, "config")
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"could not load config {source_path}: {error}") from error
    profile = _string(document["profile"], "profile")
    if profile not in {"production", "smoke_test"}:
        raise ConfigError("profile must be production or smoke_test")
    source_dir = source_path.parent
    data = _data(source_dir, document["data"])
    if profile == "production" and len({data.train.path, data.validation.path, data.test.path}) != 3:
        raise ConfigError("production data split paths must resolve to different files")
    return ConfidenceHeadConfig(source_path, profile, _checkpoint(source_dir, document["checkpoint"]), data, _cache(document["cache"]), _binning(document["binning"]), _model(document["model"]), _loss(document["loss"]), _optimizer(document["optimizer"]), _trainer(document["trainer"]), _runtime(document["runtime"]), _logging(document["logging"]), _run(source_dir, document["run"]))
