"""Build immutable continuous-feature caches with one frozen MACE load."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import ase.io
import torch

from ..backbone import BackboneIdentity, LoadedBackbone, load_frozen_backbone
from ..cache import (
    CACHE_SCHEMA_VERSION,
    FEATURE_WIDTH,
    CacheCorruptionError,
    CacheIncompleteError,
    CacheWriter,
    load_complete_cache,
)
from ..config import ConfidenceHeadConfig
from ..data import (
    DatasetHandle,
    build_structure_batch,
    load_dataset,
    validate_split_isolation,
)
from ..errors import DataContractError
from ..features import FeatureCapture, to_continuous_batch
from ..identity import cache_id, code_identity


SPLIT_ORDER = ("train", "validation", "test")
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _cache_identity(config: ConfidenceHeadConfig) -> str:
    checkpoint_path = config.checkpoint.path.resolve()
    feature_modules = [
        {"name": module.name, "expected_dim": module.expected_dim}
        for module in config.checkpoint.feature_modules
    ]
    return cache_id(
        checkpoint={
            "path": str(checkpoint_path),
            "sha256": config.checkpoint.expected_sha256.lower(),
        },
        splits={
            name: {
                "path": str(getattr(config.data, name).path.resolve()),
                "sha256": getattr(config.data, name).expected_sha256.lower()
            }
            for name in SPLIT_ORDER
        },
        feature_schema={
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "feature_width": FEATURE_WIDTH,
            "modules": feature_modules,
        },
        code=code_identity(REPOSITORY_ROOT),
    )


def _cache_root(config: ConfidenceHeadConfig, identity: str) -> Path:
    return (
        config.run.output_root
        / config.run.name_prefix
        / "cache"
        / identity
    )


def build_dataset_handles(
    config: ConfidenceHeadConfig, backbone: BackboneIdentity
) -> dict[str, DatasetHandle]:
    """Load and verify all split identities in their fixed workflow order."""
    return {
        name: load_dataset(
            getattr(config.data, name).path,
            getattr(config.data, name).expected_sha256,
            backbone.atomic_numbers,
        )
        for name in SPLIT_ORDER
    }


def read_structures(path: Path) -> Sequence[Any]:
    """Read one ordered extxyz split for deterministic sequential batching."""
    try:
        structures = ase.io.read(Path(path), index=":")
    except Exception as error:
        raise DataContractError(f"could not read dataset {path}: {error}") from error
    if not isinstance(structures, list):
        structures = [structures]
    return structures


def _detach_prediction_tensors(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().item()):
            raise DataContractError("MACE predictions must be finite")
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach_prediction_tensors(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_prediction_tensors(item) for item in value)
    if isinstance(value, list):
        return [_detach_prediction_tensors(item) for item in value]
    return value


def _validate_manifest_splits(manifest: object) -> None:
    splits = getattr(manifest, "splits", None)
    expected = frozenset(SPLIT_ORDER)
    if not isinstance(splits, Mapping) or frozenset(splits) != expected:
        raise CacheCorruptionError(
            "cache manifest split set must be exactly "
            "{train, validation, test}"
        )


def cache_one_split(
    name: str,
    *,
    handle: DatasetHandle,
    loaded: LoadedBackbone,
    capture: FeatureCapture,
    writer: CacheWriter,
    config: ConfidenceHeadConfig,
    next_index: int,
) -> None:
    """Run ordered remaining structures through MACE once per batch."""
    structures = read_structures(handle.path)
    if len(structures) != handle.size:
        raise DataContractError(
            f"{name} changed after validation: expected {handle.size} structures, "
            f"got {len(structures)}"
        )

    batch_size = config.cache.build_batch_size
    for start in range(next_index, handle.size, batch_size):
        stop = min(start + batch_size, handle.size)
        batch = build_structure_batch(
            structures[start:stop],
            indices=range(start, stop),
            backbone=loaded.identity,
            device=config.runtime.device,
        )
        expected_ids = handle.structure_ids[start:stop]
        if batch.structure_ids != expected_ids:
            raise DataContractError(f"{name} changed after identity validation")

        predictions = _detach_prediction_tensors(
            loaded.model(
                batch.mace_batch.to_dict(),
                training=False,
                compute_force=True,
            )
        )
        expected_atoms = int(batch.atom_offsets[-1].item())
        features = capture.take(expected_atoms=expected_atoms).detach()
        writer.append(to_continuous_batch(batch, features), split=name)
        del predictions


def _open_writer(
    root: Path, *, identity: str, config: ConfidenceHeadConfig
) -> CacheWriter:
    progress_path = root / "progress.pt"
    if progress_path.is_file():
        if not config.cache.resume:
            raise CacheIncompleteError(
                "partial cache exists but cache.resume is disabled"
            )
        return CacheWriter.resume(root, expected_cache_id=identity)

    if root.exists() and any(root.iterdir()):
        raise CacheCorruptionError(
            "cache identity directory contains artifacts without valid progress"
        )
    return CacheWriter(
        root,
        cache_id=identity,
        shard_max_atoms=config.cache.shard_max_atoms,
    )


def _next_index(writer: CacheWriter, name: str, size: int) -> tuple[int, bool]:
    state = writer.splits.get(name)
    if state is None:
        return 0, False
    next_index = state["next_index"]
    complete = state["complete"]
    if next_index > size or (complete and next_index != size):
        raise CacheCorruptionError(
            f"{name}: cache progress does not match validated split size"
        )
    return next_index, complete


def run_build_cache(config: ConfidenceHeadConfig) -> Path:
    """Build or validate the cache identified by immutable inputs."""
    identity = _cache_identity(config)
    root = _cache_root(config, identity)
    try:
        manifest = load_complete_cache(root, expected_cache_id=identity)
    except CacheIncompleteError:
        pass
    else:
        _validate_manifest_splits(manifest)
        return root

    loaded = load_frozen_backbone(
        config.checkpoint, device=config.runtime.device
    )
    handles = build_dataset_handles(config, loaded.identity)
    validate_split_isolation(
        handles["train"],
        handles["validation"],
        handles["test"],
        profile=config.profile,
    )
    writer = _open_writer(root, identity=identity, config=config)

    with FeatureCapture(
        loaded.model, loaded.identity.feature_modules
    ) as capture:
        for name in SPLIT_ORDER:
            handle = handles[name]
            next_index, complete = _next_index(writer, name, handle.size)
            if complete:
                continue
            cache_one_split(
                name,
                handle=handle,
                loaded=loaded,
                capture=capture,
                writer=writer,
                config=config,
                next_index=next_index,
            )
            writer.finalize_split(name)

    manifest = writer.finalize()
    _validate_manifest_splits(manifest)
    return root
