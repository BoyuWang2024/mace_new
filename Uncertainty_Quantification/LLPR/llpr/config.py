"""Parsing and validation for LLPR YAML configuration files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


_CANONICAL_CURVATURE_VARIANTS = ("he", "hf", "hef")


@dataclass(frozen=True)
class PathIdentity:
    path: Path
    expected_sha256: str | None = None


@dataclass(frozen=True)
class RidgeConfig:
    mode: Literal["fixed", "condition_number"]
    value: float
    max_condition_number: float


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    force_component_chunk_size: int
    save_every_structures: int
    resume: bool
    max_structures: int | None
    max_force_components_per_structure: int | None


@dataclass(frozen=True)
class CurvatureConfig:
    variants: tuple[Literal["he", "hf", "hef"], ...]
    min_q: float


@dataclass(frozen=True)
class DataConfig:
    build: PathIdentity
    calibration: PathIdentity
    test: PathIdentity


@dataclass(frozen=True)
class ArtifactsConfig:
    curvature: PathIdentity | None = None


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    experiment: str


@dataclass(frozen=True)
class LLPRConfig:
    source_path: Path
    checkpoint: PathIdentity
    build: PathIdentity
    calibration: PathIdentity
    test: PathIdentity
    ridge: RidgeConfig
    runtime: RuntimeConfig
    output_root: Path
    experiment: str
    curvature: CurvatureConfig
    selected_head: str = "default"
    expected_readout_size: int = 2192
    artifacts: ArtifactsConfig = ArtifactsConfig()

    @property
    def data(self) -> DataConfig:
        """Provide the grouped data view used by the YAML schema."""
        return DataConfig(self.build, self.calibration, self.test)

    @property
    def output(self) -> OutputConfig:
        """Provide the grouped output view used by the YAML schema."""
        return OutputConfig(self.output_root, self.experiment)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, field: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{field}.{key} is required")
    return mapping[key]


def _resolve_path(source_dir: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return (source_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _path_identity(source_dir: Path, value: Any, field: str) -> PathIdentity:
    mapping = _mapping(value, field)
    expected_sha256 = mapping.get("expected_sha256")
    if expected_sha256 is not None:
        expected_sha256 = str(expected_sha256)
    if expected_sha256 is not None and (
        len(expected_sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in expected_sha256)
    ):
        raise ValueError(f"{field}.expected_sha256 must be a 64-character SHA-256")
    return PathIdentity(
        path=_resolve_path(source_dir, _required(mapping, "path", field), f"{field}.path"),
        expected_sha256=expected_sha256,
    )


def _checkpoint_config(
    source_dir: Path, value: Any
) -> tuple[PathIdentity, str, int]:
    mapping = _mapping(value, "checkpoint")
    fields = {
        "path",
        "expected_sha256",
        "selected_head",
        "expected_readout_size",
    }
    if set(mapping) != fields:
        raise ValueError(
            "checkpoint must contain exactly path, expected_sha256, "
            "selected_head, expected_readout_size"
        )
    identity = _path_identity(source_dir, mapping, "checkpoint")
    if identity.expected_sha256 is None:
        raise ValueError("checkpoint.expected_sha256 must not be null")
    selected_head = mapping["selected_head"]
    if not isinstance(selected_head, str) or not selected_head.strip():
        raise ValueError("checkpoint.selected_head must be a non-empty string")
    expected_readout_size = mapping["expected_readout_size"]
    if (
        isinstance(expected_readout_size, bool)
        or not isinstance(expected_readout_size, int)
        or expected_readout_size <= 0
    ):
        raise ValueError("checkpoint.expected_readout_size must be a positive integer")
    return identity, selected_head, expected_readout_size


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer or null")
    return value


def load_config(path: Path) -> LLPRConfig:
    """Load an LLPR YAML file, resolving every relative path beside it."""
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8") as handle:
        document = _mapping(yaml.safe_load(handle), "config")
    source_dir = source_path.parent

    checkpoint, selected_head, expected_readout_size = _checkpoint_config(
        source_dir, _required(document, "checkpoint", "config")
    )
    data = _mapping(_required(document, "data", "config"), "data")
    build = _path_identity(source_dir, _required(data, "build", "data"), "data.build")
    calibration = _path_identity(
        source_dir, _required(data, "calibration", "data"), "data.calibration"
    )
    test = _path_identity(source_dir, _required(data, "test", "data"), "data.test")

    ridge_data = _mapping(_required(document, "ridge", "config"), "ridge")
    ridge_mode = _required(ridge_data, "mode", "ridge")
    if ridge_mode not in ("fixed", "condition_number"):
        raise ValueError("ridge.mode must be 'fixed' or 'condition_number'")
    ridge_value = float(_required(ridge_data, "value", "ridge"))
    max_condition_number = float(ridge_data.get("max_condition_number", 1.0e12))
    if ridge_mode == "fixed" and ridge_value < 0:
        raise ValueError("ridge.value must be non-negative when ridge.mode is 'fixed'")
    if ridge_mode == "condition_number" and max_condition_number <= 1:
        raise ValueError("ridge.max_condition_number must be greater than 1")
    ridge = RidgeConfig(ridge_mode, ridge_value, max_condition_number)

    runtime_data = _mapping(_required(document, "runtime", "config"), "runtime")
    chunk_size = runtime_data.get("force_component_chunk_size", 1)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("runtime.force_component_chunk_size must be a positive integer")
    resume = runtime_data.get("resume", False)
    if not isinstance(resume, bool):
        raise ValueError("runtime.resume must be a boolean")
    runtime = RuntimeConfig(
        device=str(runtime_data.get("device", "cpu")),
        force_component_chunk_size=chunk_size,
        save_every_structures=_optional_positive_int(
            runtime_data.get("save_every_structures", 1), "runtime.save_every_structures"
        )
        or 1,
        resume=resume,
        max_structures=_optional_positive_int(
            runtime_data.get("max_structures"), "runtime.max_structures"
        ),
        max_force_components_per_structure=_optional_positive_int(
            runtime_data.get("max_force_components_per_structure"),
            "runtime.max_force_components_per_structure",
        ),
    )

    output_data = _mapping(_required(document, "output", "config"), "output")
    experiment = _required(output_data, "experiment", "output")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("output.experiment must be non-empty")
    output_root = _resolve_path(
        source_dir, _required(output_data, "root", "output"), "output.root"
    )

    curvature_data = _mapping(document.get("curvature", {}), "curvature")
    variants = tuple(curvature_data.get("variants", _CANONICAL_CURVATURE_VARIANTS))
    if variants != _CANONICAL_CURVATURE_VARIANTS:
        raise ValueError("curvature.variants must be [he, hf, hef]")
    try:
        min_q = float(curvature_data.get("min_q", 1.0e-30))
    except (TypeError, ValueError) as error:
        raise ValueError("curvature.min_q must be the fixed positive floor 1e-30") from error
    if not math.isfinite(min_q) or min_q != 1.0e-30:
        raise ValueError("curvature.min_q must be the fixed positive floor 1e-30")
    curvature = CurvatureConfig(
        variants=variants,
        min_q=min_q,
    )

    artifacts_value = document.get("artifacts")
    if artifacts_value is None:
        artifacts = ArtifactsConfig()
    else:
        artifacts_data = _mapping(artifacts_value, "artifacts")
        if set(artifacts_data) != {"curvature"}:
            raise ValueError("artifacts must contain exactly curvature")
        curvature_value = _mapping(
            _required(artifacts_data, "curvature", "artifacts"),
            "artifacts.curvature",
        )
        if "expected_sha256" not in curvature_value:
            raise ValueError("artifacts.curvature.expected_sha256 must not be null")
        if set(curvature_value) != {"path", "expected_sha256"}:
            raise ValueError(
                "artifacts.curvature must contain exactly path, expected_sha256"
            )
        curvature_artifact = _path_identity(
            source_dir, curvature_value, "artifacts.curvature"
        )
        if curvature_artifact.expected_sha256 is None:
            raise ValueError("artifacts.curvature.expected_sha256 must not be null")
        artifacts = ArtifactsConfig(curvature=curvature_artifact)

    return LLPRConfig(
        source_path=source_path,
        checkpoint=checkpoint,
        build=build,
        calibration=calibration,
        test=test,
        ridge=ridge,
        runtime=runtime,
        output_root=output_root,
        experiment=experiment,
        curvature=curvature,
        selected_head=selected_head,
        expected_readout_size=expected_readout_size,
        artifacts=artifacts,
    )
