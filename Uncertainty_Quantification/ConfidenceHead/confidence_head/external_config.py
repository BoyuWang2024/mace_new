"""Strict configuration for shared-cache external ConfidenceHead inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ConfidenceHeadConfig
from .workflows.production_matrix import discover_production_matrix


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_TOP_KEYS = {
    "schema_version",
    "dataset",
    "production_config_dir",
    "output_root",
    "plot_root",
    "cache",
    "runtime",
}
_DATASET_KEYS = {
    "name",
    "path",
    "expected_sha256",
    "expected_structures",
    "expected_atoms",
    "source_index_path",
}
_CACHE_KEYS = {"split", "build_batch_size", "shard_max_atoms", "resume"}
_RUNTIME_KEYS = {"device", "head_batch_size"}
_E0_TOP_KEYS = {
    "schema_version",
    "validation_config",
    "test_config",
    "output_root",
    "plot_root",
    "methods",
    "atomization_energy_key",
    "build_missing_inputs",
}
_E0_METHODS = ("e0_replace", "e0_reestimate")
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ExternalConfigError(ValueError):
    """External inference YAML violates its strict schema."""


@dataclass(frozen=True)
class ExternalDatasetConfig:
    name: str
    path: Path
    expected_sha256: str
    expected_structures: int
    expected_atoms: int
    source_index_path: Path | None


@dataclass(frozen=True)
class ExternalCacheConfig:
    split: str
    build_batch_size: int
    shard_max_atoms: int
    resume: bool


@dataclass(frozen=True)
class ExternalInferenceConfig:
    source_path: Path
    dataset: ExternalDatasetConfig
    config_dir: Path
    output_root: Path
    plot_root: Path
    cache: ExternalCacheConfig
    runtime_device: str
    head_batch_size: int
    force_config: ConfidenceHeadConfig
    energy_configs: dict[int, ConfidenceHeadConfig]


@dataclass(frozen=True)
class E0PostprocessConfig:
    """Two immutable external datasets and their derived E0 outputs."""

    source_path: Path
    validation: ExternalInferenceConfig
    test: ExternalInferenceConfig
    output_root: Path
    plot_root: Path
    methods: tuple[str, ...]
    atomization_energy_key: str
    build_missing_inputs: bool


def _mapping(value: object, keys: set[str], where: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise ExternalConfigError(
            f"{where} keys differ (missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)})"
        )
    return value


def _positive_int(value: object, where: str) -> int:
    if type(value) is not int or value < 1:
        raise ExternalConfigError(f"{where} must be a positive integer")
    return value


def _path(value: object, base: Path, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExternalConfigError(f"{where} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _matrix_compatibility(
    force: ConfidenceHeadConfig,
    energy: dict[int, ConfidenceHeadConfig],
) -> None:
    expected = force.checkpoint
    for order in range(1, 9):
        if energy[order].checkpoint != expected:
            raise ExternalConfigError(
                f"production matrix checkpoint differs at energy order {order}"
            )


def load_external_config(path: Path) -> ExternalInferenceConfig:
    """Load one immutable external dataset and the existing production matrix."""
    source = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise ExternalConfigError(f"could not read external config: {error}") from error
    top = _mapping(document, _TOP_KEYS, "external config")
    if top["schema_version"] != 1:
        raise ExternalConfigError("schema_version must be 1")
    base = source.parent

    raw_dataset = _mapping(top["dataset"], _DATASET_KEYS, "dataset")
    name = raw_dataset["name"]
    if not isinstance(name, str) or _SAFE_NAME.fullmatch(name) is None:
        raise ExternalConfigError("dataset.name must be a safe lowercase name")
    digest = raw_dataset["expected_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ExternalConfigError("dataset.expected_sha256 must be lowercase SHA-256")
    source_index = raw_dataset["source_index_path"]
    if source_index is not None:
        source_index = _path(source_index, base, "dataset.source_index_path")
    dataset = ExternalDatasetConfig(
        name=name,
        path=_path(raw_dataset["path"], base, "dataset.path"),
        expected_sha256=digest,
        expected_structures=_positive_int(
            raw_dataset["expected_structures"], "dataset.expected_structures"
        ),
        expected_atoms=_positive_int(
            raw_dataset["expected_atoms"], "dataset.expected_atoms"
        ),
        source_index_path=source_index,
    )

    raw_cache = _mapping(top["cache"], _CACHE_KEYS, "cache")
    if raw_cache["split"] != "inference":
        raise ExternalConfigError("cache.split must be inference")
    if type(raw_cache["resume"]) is not bool:
        raise ExternalConfigError("cache.resume must be a boolean")
    cache = ExternalCacheConfig(
        split="inference",
        build_batch_size=_positive_int(
            raw_cache["build_batch_size"], "cache.build_batch_size"
        ),
        shard_max_atoms=_positive_int(
            raw_cache["shard_max_atoms"], "cache.shard_max_atoms"
        ),
        resume=raw_cache["resume"],
    )

    raw_runtime = _mapping(top["runtime"], _RUNTIME_KEYS, "runtime")
    device = raw_runtime["device"]
    if device not in {"cpu", "cuda"}:
        raise ExternalConfigError("runtime.device must be cpu or cuda")
    config_dir = _path(
        top["production_config_dir"], base, "production_config_dir"
    )
    force, energy = discover_production_matrix(config_dir)
    _matrix_compatibility(force, energy)
    return ExternalInferenceConfig(
        source_path=source,
        dataset=dataset,
        config_dir=config_dir,
        output_root=_path(top["output_root"], base, "output_root"),
        plot_root=_path(top["plot_root"], base, "plot_root"),
        cache=cache,
        runtime_device=device,
        head_batch_size=_positive_int(
            raw_runtime["head_batch_size"], "runtime.head_batch_size"
        ),
        force_config=force,
        energy_configs=energy,
    )

def load_e0_postprocess_config(path: Path) -> E0PostprocessConfig:
    """Load a strict validation/test E0 postprocessing experiment."""
    source = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise ExternalConfigError(f"could not read E0 config: {error}") from error
    top = _mapping(document, _E0_TOP_KEYS, "E0 postprocess config")
    if top["schema_version"] != 1:
        raise ExternalConfigError("E0 schema_version must be 1")

    base = source.parent
    validation_path = _path(
        top["validation_config"], base, "validation_config"
    )
    test_path = _path(top["test_config"], base, "test_config")
    if validation_path == test_path:
        raise ExternalConfigError("validation and test configs must differ")
    validation = load_external_config(validation_path)
    test = load_external_config(test_path)
    if validation.dataset.name == test.dataset.name:
        raise ExternalConfigError("validation and test dataset names must differ")
    if validation.force_config.checkpoint != test.force_config.checkpoint:
        raise ExternalConfigError("validation and test checkpoint differs")
    if validation.config_dir != test.config_dir:
        raise ExternalConfigError("validation and test production matrix differs")

    raw_methods = top["methods"]
    if type(raw_methods) is not list or tuple(raw_methods) != _E0_METHODS:
        raise ExternalConfigError(f"methods must equal {list(_E0_METHODS)}")
    field = top["atomization_energy_key"]
    if not isinstance(field, str) or _FIELD_NAME.fullmatch(field) is None:
        raise ExternalConfigError("atomization_energy_key is invalid")
    build_missing_inputs = top["build_missing_inputs"]
    if type(build_missing_inputs) is not bool:
        raise ExternalConfigError("build_missing_inputs must be a boolean")

    return E0PostprocessConfig(
        source_path=source,
        validation=validation,
        test=test,
        output_root=_path(top["output_root"], base, "output_root"),
        plot_root=_path(top["plot_root"], base, "plot_root"),
        methods=_E0_METHODS,
        atomization_energy_key=field,
        build_missing_inputs=build_missing_inputs,
    )
