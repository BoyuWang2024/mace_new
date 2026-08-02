"""Fit immutable error bins from complete cache-v2 train tensors only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ..artifacts import atomic_json_dump, atomic_torch_save, load_torch_artifact
from ..binning import (
    BINNING_FORMULA_VERSION,
    BINNING_SCHEMA_VERSION,
    BinningArtifact,
    BranchBinning,
    fit_fixed_linear,
    fit_train_quantile_log,
)
from ..cache import (
    CACHE_SCHEMA_VERSION,
    FEATURE_WIDTH,
    iter_cache_batches,
    load_complete_cache,
)
from ..config import (
    BinConfig,
    ConfidenceHeadConfig,
    FixedBinConfig,
    LogBinConfig,
)
from ..identity import cache_id, code_identity, experiment_id, sha256_file
from ..labels import energy_errors, force_errors
from ..run_naming import make_run_tag

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SPLIT_ORDER = ("train", "validation", "test")
_MANIFEST_KEYS = {
    "schema_version",
    "formula_version",
    "artifact_filename",
    "artifact_sha256",
    "cache_id",
    "experiment_id",
    "binning_id",
    "run_id",
    "algorithm",
    "enabled_branches",
    "train_structure_count",
    "train_atom_count",
    "branches",
}


class BinningIdentityError(ValueError):
    """Existing binning artifacts fail immutable identity validation."""


def _cache_identity(config: ConfidenceHeadConfig) -> str:
    """Reproduce the cache-v2 identity without importing MACE or ASE."""
    return cache_id(
        checkpoint={
            "path": str(config.checkpoint.path.resolve()),
            "sha256": config.checkpoint.expected_sha256.lower(),
        },
        splits={
            name: {
                "path": str(getattr(config.data, name).path.resolve()),
                "sha256": getattr(config.data, name).expected_sha256.lower(),
            }
            for name in SPLIT_ORDER
        },
        feature_schema={
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "feature_width": FEATURE_WIDTH,
            "modules": [
                {"name": module.name, "expected_dim": module.expected_dim}
                for module in config.checkpoint.feature_modules
            ],
        },
        code=code_identity(REPOSITORY_ROOT),
    )


def _cache_root(config: ConfidenceHeadConfig, identity: str) -> Path:
    return config.run.output_root / config.run.name_prefix / "cache" / identity


def _binning_root(config: ConfidenceHeadConfig, experiment_identity: str) -> Path:
    directory = f"{make_run_tag(config)}-{experiment_identity[:12]}"
    return (
        config.run.output_root / config.run.name_prefix / "runs" / directory / "binning"
    )


def _fit_branch(
    values: torch.Tensor,
    *,
    algorithm: str,
    config: BinConfig,
) -> BranchBinning:
    if algorithm == "fixed_linear_v1":
        if not isinstance(config, FixedBinConfig):
            raise ValueError("fixed_linear_v1 binning config differs")
        return fit_fixed_linear(
            values,
            num_bins=config.num_bins,
            max_error=config.max_error,
        )
    if algorithm == "train_quantile_log_v1":
        if not isinstance(config, LogBinConfig):
            raise ValueError("train_quantile_log_v1 binning config differs")
        return fit_train_quantile_log(values, num_bins=config.num_bins)
    raise ValueError("binning algorithm is unsupported")


def _payload_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.device.type == right.device.type == "cpu"
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_payload_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _payload_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    return type(left) is type(right) and left == right


def _branch_diagnostics(branch: BranchBinning) -> dict[str, Any]:
    return {
        key: value.tolist()
        if isinstance(value, torch.Tensor)
        else list(value)
        if isinstance(value, tuple)
        else value
        for key, value in branch.to_payload().items()
    }


def _manifest_payload(
    artifact: BinningArtifact,
    *,
    artifact_sha256: str,
    train_structure_count: int,
    train_atom_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": BINNING_SCHEMA_VERSION,
        "formula_version": BINNING_FORMULA_VERSION,
        "artifact_filename": "binning.pt",
        "artifact_sha256": artifact_sha256,
        "cache_id": artifact.cache_id,
        "experiment_id": artifact.experiment_id,
        "binning_id": artifact.binning_id,
        "run_id": artifact.run_id,
        "algorithm": artifact.algorithm,
        "enabled_branches": sorted(artifact.branches),
        "train_structure_count": train_structure_count,
        "train_atom_count": train_atom_count,
        "branches": {
            name: _branch_diagnostics(branch)
            for name, branch in sorted(artifact.branches.items())
        },
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BinningIdentityError(
            f"existing binning manifest identity differs: {error}"
        ) from error
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise BinningIdentityError("existing binning manifest schema differs")
    return value


def _validate_existing(
    root: Path,
    expected: BinningArtifact,
    *,
    train_structure_count: int,
    train_atom_count: int,
    force_target_mode: str | None,
) -> None:
    artifact_path = root / "binning.pt"
    manifest_path = root / "binning_manifest.json"
    try:
        payload = load_torch_artifact(artifact_path)
        actual = BinningArtifact.from_payload(
            payload, force_target_mode=force_target_mode
        )
    except Exception as error:
        raise BinningIdentityError(
            f"existing binning artifact differs: {error}"
        ) from error
    if not _payload_equal(actual.to_payload(), expected.to_payload()):
        raise BinningIdentityError("existing binning artifact differs")
    manifest = _load_manifest(manifest_path)
    expected_manifest = _manifest_payload(
        actual,
        artifact_sha256=sha256_file(artifact_path),
        train_structure_count=train_structure_count,
        train_atom_count=train_atom_count,
    )
    if manifest != expected_manifest:
        raise BinningIdentityError("existing binning manifest identity differs")


def _persist_or_reuse(
    root: Path,
    artifact: BinningArtifact,
    *,
    train_structure_count: int,
    train_atom_count: int,
    force_target_mode: str | None,
) -> None:
    artifact_path = root / "binning.pt"
    manifest_path = root / "binning_manifest.json"
    artifact_exists = artifact_path.exists()
    manifest_exists = manifest_path.exists()
    if artifact_exists != manifest_exists:
        raise BinningIdentityError(
            "existing binning requires both artifact and manifest"
        )
    if artifact_exists:
        _validate_existing(
            root,
            artifact,
            train_structure_count=train_structure_count,
            train_atom_count=train_atom_count,
            force_target_mode=force_target_mode,
        )
        return
    if root.exists() and any(root.iterdir()):
        raise BinningIdentityError("binning directory contains unknown artifacts")
    root.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(artifact_path, artifact.to_payload())
    try:
        reloaded = BinningArtifact.from_payload(
            load_torch_artifact(artifact_path),
            force_target_mode=force_target_mode,
        )
    except Exception as error:
        raise BinningIdentityError(
            f"new binning artifact failed validation: {error}"
        ) from error
    if not _payload_equal(reloaded.to_payload(), artifact.to_payload()):
        raise BinningIdentityError("new binning artifact content differs")
    atomic_json_dump(
        manifest_path,
        _manifest_payload(
            reloaded,
            artifact_sha256=sha256_file(artifact_path),
            train_structure_count=train_structure_count,
            train_atom_count=train_atom_count,
        ),
    )
    _validate_existing(
        root,
        artifact,
        train_structure_count=train_structure_count,
        train_atom_count=train_atom_count,
        force_target_mode=force_target_mode,
    )


def run_fit_bins(config: ConfidenceHeadConfig) -> Path:
    """Fit enabled branches from the committed train cache and persist once."""
    cache_identity = _cache_identity(config)
    manifest = load_complete_cache(
        _cache_root(config, cache_identity),
        expected_cache_id=cache_identity,
        allow_cross_split_duplicates=config.profile == "smoke_test",
    )
    code = code_identity(REPOSITORY_ROOT)
    experiment_identity = experiment_id(config, cache_identity, code=code)

    force_parts: list[torch.Tensor] = []
    energy_parts: list[torch.Tensor] = []
    train_structure_count = 0
    train_atom_count = 0
    for batch in iter_cache_batches(
        manifest, "train", batch_size=config.trainer.batch_size
    ):
        train_structure_count += len(batch.structure_id)
        train_atom_count += int(batch.atom_offsets[-1].item())
        if config.force_enabled:
            force_parts.append(
                force_errors(
                    batch.force_prediction,
                    batch.force_reference,
                    config.model.force.target_mode,
                ).reshape(-1)
            )
        if config.energy_enabled:
            energy_parts.append(
                energy_errors(
                    batch.energy_prediction,
                    batch.energy_reference,
                    batch.num_atoms,
                )
            )
    branches: dict[str, BranchBinning] = {}
    if config.force_enabled:
        if not force_parts:
            raise ValueError("train cache contains no force errors")
        branches["force"] = _fit_branch(
            torch.cat(force_parts),
            algorithm=config.binning.algorithm,
            config=config.binning.force,
        )
    if config.energy_enabled:
        if not energy_parts:
            raise ValueError("train cache contains no energy errors")
        branches["energy"] = _fit_branch(
            torch.cat(energy_parts),
            algorithm=config.binning.algorithm,
            config=config.binning.energy,
        )
    artifact = BinningArtifact.create(
        cache_id=cache_identity,
        experiment_id=experiment_identity,
        algorithm=config.binning.algorithm,
        branches=branches,
        force_target_mode=(
            config.model.force.target_mode if config.force_enabled else None
        ),
    )
    root = _binning_root(config, experiment_identity)
    _persist_or_reuse(
        root,
        artifact,
        train_structure_count=train_structure_count,
        train_atom_count=train_atom_count,
        force_target_mode=(
            config.model.force.target_mode if config.force_enabled else None
        ),
    )
    return root
