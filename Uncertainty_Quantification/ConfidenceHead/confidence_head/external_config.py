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
