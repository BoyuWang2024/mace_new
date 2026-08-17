"""Build one resumable frozen-MACE cache for an external labeled dataset."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ..backbone import load_frozen_backbone
from ..cache import (
    CACHE_SCHEMA_VERSION,
    FEATURE_WIDTH,
    CacheCorruptionError,
    CacheIncompleteError,
    CacheManifest,
    CacheWriter,
    load_complete_cache,
)
from ..data import load_dataset
from ..external_config import ExternalInferenceConfig
from ..features import FeatureCapture
from ..identity import cache_id, code_identity
from .build_cache import REPOSITORY_ROOT, cache_one_split


def external_cache_id(config: ExternalInferenceConfig) -> str:
    """Bind one external dataset to the exact frozen feature definition."""
    checkpoint = config.force_config.checkpoint
    return cache_id(
        checkpoint={
            "path": str(checkpoint.path.resolve()),
            "sha256": checkpoint.expected_sha256.lower(),
        },
        splits={
            config.cache.split: {
                "path": str(config.dataset.path.resolve()),
                "sha256": config.dataset.expected_sha256.lower(),
            }
        },
        feature_schema={
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "feature_width": FEATURE_WIDTH,
            "modules": [
                {"name": module.name, "expected_dim": module.expected_dim}
                for module in checkpoint.feature_modules
            ],
        },
        code=code_identity(REPOSITORY_ROOT),
    )


def external_cache_root(config: ExternalInferenceConfig) -> Path:
    return (
        config.output_root
        / config.dataset.name
        / "cache"
        / external_cache_id(config)
    )


def _validate_counts(
    manifest: CacheManifest, config: ExternalInferenceConfig
) -> None:
    if set(manifest.splits) != {config.cache.split}:
        raise CacheCorruptionError("external cache must contain only inference split")
    shards = manifest.splits[config.cache.split]
    structures = sum(shard.num_structures for shard in shards)
    atoms = sum(shard.num_atoms for shard in shards)
    if structures != config.dataset.expected_structures:
        raise CacheCorruptionError("external cache structure count differs")
    if atoms != config.dataset.expected_atoms:
        raise CacheCorruptionError("external cache atom count differs")


def _writer(
    root: Path, identity: str, config: ExternalInferenceConfig
) -> CacheWriter:
    progress = root / "progress.pt"
    if progress.is_file():
        if not config.cache.resume:
            raise CacheIncompleteError(
                "partial external cache exists but resume is disabled"
            )
        return CacheWriter.resume(root, expected_cache_id=identity)
    if root.exists() and any(root.iterdir()):
        raise CacheCorruptionError(
            "external cache directory contains artifacts without progress"
        )
    return CacheWriter(
        root,
        cache_id=identity,
        shard_max_atoms=config.cache.shard_max_atoms,
    )


def run_build_external_cache(config: ExternalInferenceConfig) -> Path:
    """Run the frozen backbone once and commit one inference cache split."""
    identity = external_cache_id(config)
    root = external_cache_root(config)
    try:
        complete = load_complete_cache(root, expected_cache_id=identity)
    except CacheIncompleteError:
        pass
    else:
        _validate_counts(complete, config)
        return root

    loaded = load_frozen_backbone(
        config.force_config.checkpoint, device=config.runtime_device
    )
    handle = load_dataset(
        config.dataset.path,
        config.dataset.expected_sha256,
        loaded.identity.atomic_numbers,
    )
    if handle.size != config.dataset.expected_structures:
        raise CacheCorruptionError("external dataset structure count differs")
    writer = _writer(root, identity, config)
    state = writer.splits.get(config.cache.split)
    next_index = 0 if state is None else int(state["next_index"])
    complete_split = False if state is None else bool(state["complete"])
    if next_index > handle.size or (complete_split and next_index != handle.size):
        raise CacheCorruptionError("external cache progress differs from dataset")
    adapter = SimpleNamespace(
        cache=SimpleNamespace(build_batch_size=config.cache.build_batch_size),
        runtime=SimpleNamespace(device=config.runtime_device),
    )
    if not complete_split:
        with FeatureCapture(
            loaded.model, loaded.identity.feature_modules
        ) as capture:
            cache_one_split(
                config.cache.split,
                handle=handle,
                loaded=loaded,
                capture=capture,
                writer=writer,
                config=adapter,
                next_index=next_index,
            )
        writer.finalize_split(config.cache.split)
    manifest = writer.finalize()
    _validate_counts(manifest, config)
    return root
