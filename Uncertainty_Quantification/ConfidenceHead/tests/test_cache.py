"""Contracts for durable, versioned ConfidenceHead feature caches."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from confidence_head.artifacts import load_torch_artifact
from confidence_head.features import ContinuousBatch


def batch_with_atom_counts(counts: list[int], *, dtype: torch.dtype = torch.float64) -> ContinuousBatch:
    num_atoms = torch.tensor(counts, dtype=torch.long)
    atom_offsets = torch.cat((torch.zeros(1, dtype=torch.long), num_atoms.cumsum(0)))
    total_atoms = int(atom_offsets[-1])
    return ContinuousBatch(
        indices=torch.arange(len(counts), dtype=torch.long),
        structure_ids=tuple(f"structure-{index}" for index in range(len(counts))),
        num_atoms=num_atoms,
        atom_offsets=atom_offsets,
        features=torch.arange(total_atoms * 640, dtype=dtype).reshape(total_atoms, 640),
        reference_energy=torch.arange(len(counts), dtype=dtype),
        reference_forces=torch.full((total_atoms, 3), 0.125, dtype=dtype),
    )


def write_partial_cache(root: Path) -> Path:
    from confidence_head.cache import CacheWriter

    writer = CacheWriter(root, cache_id="cache", shard_max_atoms=5)
    writer.append(batch_with_atom_counts([3, 4]))
    writer.finalize_split("train")
    return root


def test_cache_writer_never_splits_one_structure(tmp_path: Path) -> None:
    from confidence_head.cache import CacheWriter

    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=5)
    writer.append(batch_with_atom_counts([3, 4]))
    writer.finalize_split("train")
    assert [load_torch_artifact(path)["num_atoms"].tolist() for path in sorted((tmp_path / "train").glob("shard-*.pt"))] == [[3], [4]]


def test_resume_rejects_modified_committed_shard(tmp_path: Path) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    root = write_partial_cache(tmp_path)
    (root / "train" / "shard-000000.pt").write_bytes(b"corrupt")
    with pytest.raises(CacheCorruptionError, match="shard-000000"):
        CacheWriter.resume(root, expected_cache_id="cache")


def test_incomplete_cache_cannot_be_opened_for_training(tmp_path: Path) -> None:
    from confidence_head.cache import CacheIncompleteError, load_complete_cache

    with pytest.raises(CacheIncompleteError):
        load_complete_cache(write_partial_cache(tmp_path), expected_cache_id="cache")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda batch: replace(batch, indices=torch.tensor([0, 2])), "continuous"),
        (lambda batch: replace(batch, structure_ids=("structure-0", "structure-0")), "duplicate"),
        (lambda batch: replace(batch, reference_energy=batch.reference_energy.float()), "dtype"),
        (lambda batch: replace(batch, features=torch.full_like(batch.features, float("nan"))), "finite"),
    ],
)
def test_writer_rejects_invalid_batch_contract(tmp_path: Path, mutation, message: str) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    with pytest.raises(CacheCorruptionError, match=message):
        CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=10).append(mutation(batch_with_atom_counts([2, 2])))


def test_complete_manifest_rejects_missing_shards_and_ignores_temporary_files(tmp_path: Path) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter, load_complete_cache

    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=2)
    writer.append(batch_with_atom_counts([2, 2]))
    writer.finalize_split("train")
    writer.finalize()
    (tmp_path / "train" / ".shard-999999.pt.stale.tmp").write_bytes(b"partial")
    assert load_complete_cache(tmp_path, expected_cache_id="cache").complete is True
    (tmp_path / "train" / "shard-000001.pt").unlink()
    with pytest.raises(CacheCorruptionError, match="shard-000001"):
        load_complete_cache(tmp_path, expected_cache_id="cache")


def test_iter_cache_batches_repacks_full_structures_and_preserves_dtype(tmp_path: Path) -> None:
    from confidence_head.cache import CacheWriter, iter_cache_batches, load_complete_cache

    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=3)
    writer.append(batch_with_atom_counts([2, 1, 3], dtype=torch.float32))
    writer.finalize_split("train")
    writer.finalize()
    batches = list(iter_cache_batches(load_complete_cache(tmp_path, expected_cache_id="cache"), "train", batch_size=2))
    assert [batch.num_atoms.tolist() for batch in batches] == [[2, 1], [3]]
    assert [batch.atom_offsets.tolist() for batch in batches] == [[0, 2, 3], [0, 3]]
    assert batches[0].features.dtype is torch.float32
    assert batches[0].reference_energy.dtype is torch.float32
    assert batches[0].reference_forces.dtype is torch.float32


def test_complete_cache_rejects_manifest_schema_or_identity_mismatch(tmp_path: Path) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter, load_complete_cache

    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=10)
    writer.append(batch_with_atom_counts([2]))
    writer.finalize_split("train")
    writer.finalize()
    with pytest.raises(CacheCorruptionError, match="cache_id"):
        load_complete_cache(tmp_path, expected_cache_id="other")
    (tmp_path / "cache_manifest.json").write_text('{"schema_version":999}', encoding="utf-8")
    with pytest.raises(CacheCorruptionError, match="schema_version"):
        load_complete_cache(tmp_path, expected_cache_id="cache")
