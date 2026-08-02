"""Persistence contracts for smoke-only cache split reuse."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from confidence_head.artifacts import load_torch_artifact
from confidence_head.cache import (
    CacheCorruptionError,
    CacheWriter,
    ContinuousBatch,
    iter_cache_batches,
    load_complete_cache,
)


def _one_structure() -> ContinuousBatch:
    return ContinuousBatch(
        structure_index=torch.tensor([0]),
        structure_id=("same-structure",),
        atomic_numbers=torch.tensor([1]),
        atom_offsets=torch.tensor([0, 1]),
        scalar_features=torch.zeros(1, 640),
        force_prediction=torch.zeros(1, 3),
        force_reference=torch.zeros(1, 3),
        energy_prediction=torch.zeros(1),
        energy_reference=torch.zeros(1),
    )


def test_persisted_cache_policy_is_authoritative_across_resume_and_load(
    tmp_path: Path,
) -> None:
    writer = CacheWriter(
        tmp_path,
        cache_id="cache",
        shard_max_atoms=8,
        allow_cross_split_duplicates=True,
    )
    writer.append(_one_structure(), split="train")
    writer.finalize_split("train")

    progress = load_torch_artifact(tmp_path / "progress.pt")
    assert progress["allow_cross_split_duplicates"] is True
    with pytest.raises(CacheCorruptionError, match="policy"):
        CacheWriter.resume(
            tmp_path,
            expected_cache_id="cache",
            allow_cross_split_duplicates=False,
        )

    writer = CacheWriter.resume(
        tmp_path,
        expected_cache_id="cache",
        allow_cross_split_duplicates=True,
    )
    writer.append(_one_structure(), split="validation")
    writer.finalize_split("validation")
    committed = writer.finalize()
    disk_manifest = json.loads(
        (tmp_path / "cache_manifest.json").read_text(encoding="utf-8")
    )
    assert disk_manifest["allow_cross_split_duplicates"] is True

    with pytest.raises(CacheCorruptionError, match="policy"):
        load_complete_cache(
            tmp_path,
            expected_cache_id="cache",
            expected_allow_cross_split_duplicates=False,
        )
    loaded = load_complete_cache(
        tmp_path,
        expected_cache_id="cache",
        expected_allow_cross_split_duplicates=True,
    )
    assert loaded == committed

    forged = replace(loaded, allow_cross_split_duplicates=False)
    with pytest.raises(CacheCorruptionError, match="policy"):
        list(iter_cache_batches(forged, "train", batch_size=1))


def test_manifest_object_cannot_self_authorize_smoke_policy(tmp_path: Path) -> None:
    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=8)
    writer.append(_one_structure(), split="train")
    writer.finalize_split("train")
    committed = writer.finalize()

    forged = replace(committed, allow_cross_split_duplicates=True)
    with pytest.raises(CacheCorruptionError, match="not bound"):
        list(iter_cache_batches(forged, "train", batch_size=1))
