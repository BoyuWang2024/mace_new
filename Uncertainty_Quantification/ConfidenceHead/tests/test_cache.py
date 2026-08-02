"""Contracts for durable, versioned ConfidenceHead feature caches."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from confidence_head.artifacts import (
    atomic_json_dump,
    atomic_torch_save,
    load_torch_artifact,
)
from confidence_head.cache import ContinuousBatch
from confidence_head.identity import sha256_file


def batch_with_atom_counts(
    counts: list[int],
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    index_start: int = 0,
    structure_prefix: str = "structure",
) -> ContinuousBatch:
    atom_counts = torch.tensor(counts, dtype=torch.long, device=device)
    atom_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            atom_counts.cumsum(0),
        )
    )
    total_atoms = int(atom_offsets[-1])
    return ContinuousBatch(
        structure_index=torch.arange(
            index_start,
            index_start + len(counts),
            dtype=torch.long,
            device=device,
        ),
        structure_id=tuple(
            f"{structure_prefix}-{index}" for index in range(len(counts))
        ),
        atomic_numbers=torch.ones(
            total_atoms, dtype=torch.long, device=device
        ),
        atom_offsets=atom_offsets,
        scalar_features=torch.arange(
            total_atoms * 640, dtype=dtype, device=device
        ).reshape(total_atoms, 640),
        force_prediction=torch.full(
            (total_atoms, 3), 0.25, dtype=dtype, device=device
        ),
        force_reference=torch.full(
            (total_atoms, 3), 0.125, dtype=dtype, device=device
        ),
        energy_prediction=torch.arange(
            len(counts), dtype=dtype, device=device
        ) + 0.5,
        energy_reference=torch.arange(
            len(counts), dtype=dtype, device=device
        ),
    )


def write_partial_cache(root: Path) -> Path:
    from confidence_head.cache import CacheWriter

    writer = CacheWriter(root, cache_id="cache", shard_max_atoms=5)
    writer.append(batch_with_atom_counts([3, 4]))
    writer.finalize_split("train")
    return root


def write_two_split_cache(root: Path, *, complete: bool) -> Path:
    from confidence_head.cache import CacheWriter

    writer = CacheWriter(root, cache_id="cache", shard_max_atoms=5)
    writer.append(batch_with_atom_counts([2]), split="train")
    writer.finalize_split("train")
    writer.append(
        batch_with_atom_counts([2], structure_prefix="validation"),
        split="validation",
    )
    writer.finalize_split("validation")
    if complete:
        writer.finalize()
    return root


def duplicate_validation_id_and_recommit(root: Path) -> None:
    shard_path = root / "validation" / "shard-000000.pt"
    shard = load_torch_artifact(shard_path)
    shard["structure_id"] = ("structure-0",)
    atomic_torch_save(shard_path, shard)
    replacement_hash = sha256_file(shard_path)

    progress = load_torch_artifact(root / "progress.pt")
    progress["splits"]["validation"]["shards"][0][
        "sha256"
    ] = replacement_hash
    atomic_torch_save(root / "progress.pt", progress)

    manifest_path = root / "cache_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["splits"]["validation"][0]["sha256"] = replacement_hash
        atomic_json_dump(manifest_path, manifest)


class UnsafePickle:
    def __reduce__(self):
        return (eval, ("40 + 2",))


def approved_payload_batch(
    counts: list[int],
    *,
    index_start: int = 0,
    structure_prefix: str = "approved",
) -> ContinuousBatch:
    """Build independent, hand-derived source tensors for schema-v2 tests."""
    atom_counts = torch.tensor(counts, dtype=torch.long)
    atom_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), atom_counts.cumsum(0))
    )
    total_atoms = int(atom_offsets[-1].item())
    return ContinuousBatch(
        structure_index=torch.arange(
            index_start, index_start + len(counts), dtype=torch.long
        ),
        structure_id=tuple(
            f"{structure_prefix}-{index}" for index in range(len(counts))
        ),
        atomic_numbers=torch.tensor(
            [1 + atom_index for atom_index in range(total_atoms)],
            dtype=torch.int32,
        ),
        atom_offsets=atom_offsets,
        scalar_features=torch.arange(
            total_atoms * 640, dtype=torch.float32
        ).reshape(total_atoms, 640),
        force_prediction=torch.arange(
            total_atoms * 3, dtype=torch.float64
        ).reshape(total_atoms, 3),
        force_reference=torch.full(
            (total_atoms, 3), 0.125, dtype=torch.float16
        ),
        energy_prediction=(
            torch.arange(len(counts), dtype=torch.float64) + 0.5
        ),
        energy_reference=(
            torch.arange(len(counts), dtype=torch.float32) - 0.25
        ),
    )


@pytest.mark.parametrize(
    "field",
    ("atomic_numbers", "force_prediction", "energy_prediction"),
)
def test_shard_preserves_each_approved_payload_field(
    tmp_path: Path, field: str
) -> None:
    from confidence_head.cache import CacheWriter

    batch = approved_payload_batch([2, 1])
    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=10)
    writer.append(batch)
    writer.finalize_split("train")

    payload = load_torch_artifact(
        tmp_path / "train" / "shard-000000.pt"
    )
    expected = getattr(batch, field)
    actual = payload[field]
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_approved_payload_round_trip_and_repacking_preserve_independent_dtypes(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import CacheWriter, iter_cache_batches

    batch = approved_payload_batch([2, 1, 3])
    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=3)
    writer.append(batch)
    writer.finalize_split("train")
    manifest = writer.finalize()

    repacked = list(iter_cache_batches(manifest, "train", batch_size=2))
    assert [item.atom_offsets.tolist() for item in repacked] == [
        [0, 2, 3],
        [0, 3],
    ]
    assert torch.equal(
        repacked[0].atomic_numbers,
        torch.tensor([1, 2, 3], dtype=torch.int32),
    )
    assert repacked[0].scalar_features.dtype is torch.float32
    assert repacked[0].force_prediction.dtype is torch.float64
    assert repacked[0].force_reference.dtype is torch.float16
    assert repacked[0].energy_prediction.dtype is torch.float64
    assert repacked[0].energy_reference.dtype is torch.float32


_NON_FEATURE_DTYPE_FIELDS = (
    "force_prediction",
    "force_reference",
    "energy_prediction",
    "energy_reference",
)


def batch_with_changed_field_dtype(
    *,
    field: str,
    index_start: int,
    structure_prefix: str,
) -> ContinuousBatch:
    batch = approved_payload_batch(
        [1],
        index_start=index_start,
        structure_prefix=structure_prefix,
    )
    source = getattr(batch, field)
    target_dtype = (
        torch.float32
        if source.dtype is not torch.float32
        else torch.float64
    )
    return replace(batch, **{field: source.to(dtype=target_dtype)})


@pytest.mark.parametrize("field", _NON_FEATURE_DTYPE_FIELDS)
def test_writer_flushes_before_each_non_feature_dtype_transition(
    tmp_path: Path, field: str
) -> None:
    from confidence_head.cache import CacheWriter

    first = approved_payload_batch(
        [1], structure_prefix=f"{field}-first"
    )
    second = batch_with_changed_field_dtype(
        field=field,
        index_start=1,
        structure_prefix=f"{field}-second",
    )
    writer = CacheWriter(
        tmp_path, cache_id="cache", shard_max_atoms=10
    )
    writer.append(first)
    writer.append(second)
    writer.finalize_split("train")

    payloads = [
        load_torch_artifact(path)
        for path in sorted((tmp_path / "train").glob("shard-*.pt"))
    ]
    assert len(payloads) == 2
    assert [payload[field].dtype for payload in payloads] == [
        getattr(first, field).dtype,
        getattr(second, field).dtype,
    ]
    assert all(
        payload["scalar_features"].dtype is torch.float32
        for payload in payloads
    )


@pytest.mark.parametrize("field", _NON_FEATURE_DTYPE_FIELDS)
def test_iterator_yields_before_each_non_feature_dtype_transition(
    tmp_path: Path, field: str
) -> None:
    from confidence_head.cache import CacheWriter, iter_cache_batches

    first = approved_payload_batch(
        [1], structure_prefix=f"{field}-first"
    )
    second = batch_with_changed_field_dtype(
        field=field,
        index_start=1,
        structure_prefix=f"{field}-second",
    )
    writer = CacheWriter(
        tmp_path, cache_id="cache", shard_max_atoms=1
    )
    writer.append(first)
    writer.append(second)
    writer.finalize_split("train")
    manifest = writer.finalize()

    batches = list(iter_cache_batches(manifest, "train", batch_size=10))
    assert len(batches) == 2
    assert [getattr(batch, field).dtype for batch in batches] == [
        getattr(first, field).dtype,
        getattr(second, field).dtype,
    ]
    assert all(
        batch.scalar_features.dtype is torch.float32
        for batch in batches
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "force_prediction",
            torch.tensor([[float("nan"), 0.0, 0.0]], dtype=torch.float64),
        ),
        (
            "energy_prediction",
            torch.tensor([float("inf")], dtype=torch.float64),
        ),
        (
            "force_prediction",
            torch.zeros(1, 2, dtype=torch.float64),
        ),
        (
            "energy_prediction",
            torch.zeros(1, 1, dtype=torch.float64),
        ),
        (
            "atomic_numbers",
            torch.tensor([1.0], dtype=torch.float32),
        ),
        (
            "atomic_numbers",
            torch.tensor([1, 8], dtype=torch.int64),
        ),
    ],
)
def test_writer_rejects_malformed_approved_payload_fields(
    tmp_path: Path,
    field: str,
    replacement: torch.Tensor,
) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    batch = approved_payload_batch([1])
    setattr(batch, field, replacement)
    with pytest.raises(CacheCorruptionError):
        CacheWriter(
            tmp_path, cache_id="cache", shard_max_atoms=10
        ).append(batch)


def test_cache_writer_never_splits_one_structure(tmp_path: Path) -> None:
    from confidence_head.cache import CacheWriter

    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=5)
    writer.append(batch_with_atom_counts([3, 4]))
    writer.finalize_split("train")
    payloads = [
        load_torch_artifact(path)
        for path in sorted((tmp_path / "train").glob("shard-*.pt"))
    ]
    assert [payload["num_atoms"] for payload in payloads] == [3, 4]


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
        (lambda batch: replace(batch, structure_index=torch.tensor([0, 2])), "continuous"),
        (lambda batch: replace(batch, structure_id=("structure-0", "structure-0")), "duplicate"),
        (
            lambda batch: replace(
                batch, atomic_numbers=batch.atomic_numbers.float()
            ),
            "integral",
        ),
        (lambda batch: replace(batch, scalar_features=torch.full_like(batch.scalar_features, float("nan"))), "finite"),
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
    assert batches[0].scalar_features.dtype is torch.float32
    assert batches[0].energy_reference.dtype is torch.float32
    assert batches[0].force_reference.dtype is torch.float32


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


def test_cache_import_does_not_load_ase_or_mace() -> None:
    import os
    import subprocess
    import sys
    package_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import confidence_head.cache; assert 'ase' not in sys.modules; assert not any(name.startswith('mace') for name in sys.modules)"],
        cwd=package_root, env={**os.environ, "PYTHONPATH": str(package_root)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_append_validates_whole_batch_before_slicing(tmp_path: Path) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    batch = batch_with_atom_counts([2, 2])
    broken = replace(batch, scalar_features=torch.cat((batch.scalar_features, batch.scalar_features[:1])))
    with pytest.raises(CacheCorruptionError):
        CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=10).append(broken)


@pytest.mark.parametrize(
    "case",
    [
        "structure_ids_length",
        "indices_length",
        "offsets_length",
        "energy_length",
        "feature_rows",
        "force_rows",
        "offsets_start",
        "offsets_monotonic",
        "offsets_terminal",
        "feature_width",
        "force_width",
        "finite_energy",
        "finite_forces",
        "indices_dtype",
        "offsets_dtype",
        "floating_device",
    ],
)
def test_append_rejects_every_malformed_batch_boundary(
    tmp_path: Path, case: str
) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    batch = batch_with_atom_counts([2, 2])
    if case == "structure_ids_length":
        broken = replace(batch, structure_id=batch.structure_id + ("extra",))
    elif case == "indices_length":
        broken = replace(
            batch,
            structure_index=torch.cat((batch.structure_index, torch.tensor([2]))),
        )
    elif case == "offsets_length":
        broken = replace(
            batch,
            atom_offsets=torch.cat(
                (batch.atom_offsets, batch.atom_offsets[-1:])
            ),
        )
    elif case == "energy_length":
        broken = replace(
            batch,
            energy_reference=torch.cat(
                (batch.energy_reference, batch.energy_reference[:1])
            ),
        )
    elif case == "feature_rows":
        broken = replace(
            batch,
            scalar_features=torch.cat((batch.scalar_features, batch.scalar_features[:1])),
        )
    elif case == "force_rows":
        broken = replace(
            batch,
            force_reference=torch.cat(
                (batch.force_reference, batch.force_reference[:1])
            ),
        )
    elif case == "offsets_start":
        broken = replace(batch, atom_offsets=torch.tensor([1, 2, 4]))
    elif case == "offsets_monotonic":
        broken = replace(batch, atom_offsets=torch.tensor([0, 3, 2]))
    elif case == "offsets_terminal":
        broken = replace(batch, atom_offsets=torch.tensor([0, 2, 3]))
    elif case == "feature_width":
        broken = replace(batch, scalar_features=batch.scalar_features[:, :-1])
    elif case == "force_width":
        broken = replace(
            batch, force_reference=batch.force_reference[:, :-1]
        )
    elif case == "finite_energy":
        values = batch.energy_reference.clone()
        values[-1] = float("nan")
        broken = replace(batch, energy_reference=values)
    elif case == "finite_forces":
        values = batch.force_reference.clone()
        values[-1, -1] = float("inf")
        broken = replace(batch, force_reference=values)
    elif case == "indices_dtype":
        broken = replace(batch, structure_index=batch.structure_index.to(torch.int32))
    elif case == "offsets_dtype":
        broken = replace(
            batch, atom_offsets=batch.atom_offsets.to(torch.int32)
        )
    elif case == "floating_device":
        broken = replace(
            batch,
            energy_reference=torch.empty(
                batch.energy_reference.shape,
                dtype=batch.energy_reference.dtype,
                device="meta",
            ),
        )
    else:
        raise AssertionError(f"unknown test case {case}")

    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=10)
    with pytest.raises(CacheCorruptionError):
        writer.append(broken)


def test_invalid_batch_does_not_partially_mutate_writer(tmp_path: Path) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=10)
    batch = batch_with_atom_counts([2, 2])
    energies = batch.energy_reference.clone()
    energies[-1] = float("nan")

    with pytest.raises(CacheCorruptionError):
        writer.append(replace(batch, energy_reference=energies))

    writer.append(batch)
    writer.finalize_split("train")
    assert load_torch_artifact(
        tmp_path / "train" / "shard-000000.pt"
    )["structure_id"] == batch.structure_id


def test_writer_rejects_dtype_changes_between_batches(tmp_path: Path) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=10)
    writer.append(batch_with_atom_counts([2], dtype=torch.float32))
    second = batch_with_atom_counts(
        [2],
        dtype=torch.float64,
        index_start=1,
        structure_prefix="next",
    )
    with pytest.raises(CacheCorruptionError, match="dtype"):
        writer.append(second)


@pytest.mark.parametrize(
    "case",
    [
        "extra",
        "missing",
        "schema_bool",
        "cache_id_type",
        "shard_max_atoms_bool",
        "shard_max_atoms_zero",
        "splits_type",
    ],
)
def test_resume_requires_exact_progress_schema(
    tmp_path: Path, case: str
) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    write_partial_cache(tmp_path)
    progress = load_torch_artifact(tmp_path / "progress.pt")
    if case == "extra":
        progress["extra"] = None
    elif case == "missing":
        del progress["cache_id"]
    elif case == "schema_bool":
        progress["schema_version"] = True
    elif case == "cache_id_type":
        progress["cache_id"] = 1
    elif case == "shard_max_atoms_bool":
        progress["shard_max_atoms"] = True
    elif case == "shard_max_atoms_zero":
        progress["shard_max_atoms"] = 0
    elif case == "splits_type":
        progress["splits"] = []
    atomic_torch_save(tmp_path / "progress.pt", progress)

    with pytest.raises(CacheCorruptionError):
        CacheWriter.resume(tmp_path, expected_cache_id="cache")


@pytest.mark.parametrize(
    "case",
    [
        "extra",
        "missing",
        "complete_integer",
        "counter_bool",
        "shards_tuple",
    ],
)
def test_resume_requires_exact_split_state_schema(
    tmp_path: Path, case: str
) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    write_partial_cache(tmp_path)
    progress = load_torch_artifact(tmp_path / "progress.pt")
    state = progress["splits"]["train"]
    if case == "extra":
        state["extra"] = None
    elif case == "missing":
        del state["next_index"]
    elif case == "complete_integer":
        state["complete"] = 1
    elif case == "counter_bool":
        state["num_structures"] = True
    elif case == "shards_tuple":
        state["shards"] = tuple(state["shards"])
    atomic_torch_save(tmp_path / "progress.pt", progress)

    with pytest.raises(CacheCorruptionError):
        CacheWriter.resume(tmp_path, expected_cache_id="cache")


@pytest.mark.parametrize(
    "case",
    [
        "extra",
        "missing",
        "index_bool",
        "filename_type",
        "hash_format",
        "count_bool",
    ],
)
def test_resume_requires_exact_shard_record_schema(
    tmp_path: Path, case: str
) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    write_partial_cache(tmp_path)
    progress = load_torch_artifact(tmp_path / "progress.pt")
    record = progress["splits"]["train"]["shards"][0]
    if case == "extra":
        record["extra"] = None
    elif case == "missing":
        del record["sha256"]
    elif case == "index_bool":
        record["shard_index"] = False
    elif case == "filename_type":
        record["filename"] = 7
    elif case == "hash_format":
        record["sha256"] = "not-a-sha256"
    elif case == "count_bool":
        record["num_structures"] = True
    atomic_torch_save(tmp_path / "progress.pt", progress)

    with pytest.raises(CacheCorruptionError):
        CacheWriter.resume(tmp_path, expected_cache_id="cache")


@pytest.mark.parametrize(
    "case",
    [
        "extra",
        "missing",
        "schema_bool",
        "complete_integer",
        "splits_type",
    ],
)
def test_loader_requires_exact_manifest_schema(
    tmp_path: Path, case: str
) -> None:
    from confidence_head.cache import CacheCorruptionError, load_complete_cache

    write_two_split_cache(tmp_path, complete=True)
    path = tmp_path / "cache_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if case == "extra":
        manifest["extra"] = None
    elif case == "missing":
        del manifest["splits"]
    elif case == "schema_bool":
        manifest["schema_version"] = True
    elif case == "complete_integer":
        manifest["complete"] = 1
    elif case == "splits_type":
        manifest["splits"] = []
    atomic_json_dump(path, manifest)

    with pytest.raises(CacheCorruptionError):
        load_complete_cache(tmp_path, expected_cache_id="cache")


def test_loader_reports_false_boolean_manifest_as_incomplete(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import CacheIncompleteError, load_complete_cache

    write_two_split_cache(tmp_path, complete=True)
    path = tmp_path / "cache_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    atomic_json_dump(path, manifest)

    with pytest.raises(CacheIncompleteError):
        load_complete_cache(tmp_path, expected_cache_id="cache")


def test_loader_rejects_undeclared_shard_in_orphan_directory(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import CacheCorruptionError, load_complete_cache

    write_two_split_cache(tmp_path, complete=True)
    orphan = tmp_path / "orphan" / "nested" / "shard-000000.pt"
    orphan.parent.mkdir(parents=True)
    shutil.copyfile(tmp_path / "train" / "shard-000000.pt", orphan)

    with pytest.raises(CacheCorruptionError, match="undeclared shard"):
        load_complete_cache(tmp_path, expected_cache_id="cache")


def test_iterator_rejects_incomplete_manifest_object(tmp_path: Path) -> None:
    from confidence_head.cache import (
        CacheIncompleteError,
        iter_cache_batches,
        load_complete_cache,
    )

    write_two_split_cache(tmp_path, complete=True)
    manifest = load_complete_cache(tmp_path, expected_cache_id="cache")
    forged = replace(manifest, complete=False)

    with pytest.raises(CacheIncompleteError):
        list(iter_cache_batches(forged, "train", batch_size=1))


def test_iterator_rejects_manifest_object_not_bound_to_disk(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import (
        CacheCorruptionError,
        iter_cache_batches,
        load_complete_cache,
    )

    write_two_split_cache(tmp_path, complete=True)
    manifest = load_complete_cache(tmp_path, expected_cache_id="cache")
    forged = replace(manifest, splits={})

    with pytest.raises(CacheCorruptionError, match="manifest"):
        list(iter_cache_batches(forged, "train", batch_size=1))


def test_iterator_revalidates_progress_before_first_yield(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import (
        CacheCorruptionError,
        iter_cache_batches,
        load_complete_cache,
    )

    write_two_split_cache(tmp_path, complete=True)
    manifest = load_complete_cache(tmp_path, expected_cache_id="cache")
    progress = load_torch_artifact(tmp_path / "progress.pt")
    progress["extra"] = None
    atomic_torch_save(tmp_path / "progress.pt", progress)

    with pytest.raises(CacheCorruptionError, match="progress"):
        list(iter_cache_batches(manifest, "train", batch_size=1))


def test_writer_rejects_duplicate_structure_ids_across_splits(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=5)
    writer.append(batch_with_atom_counts([2]), split="train")
    writer.finalize_split("train")

    with pytest.raises(CacheCorruptionError, match="duplicate"):
        writer.append(batch_with_atom_counts([2]), split="validation")


def test_resume_rejects_duplicate_structure_ids_across_splits(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    write_two_split_cache(tmp_path, complete=False)
    duplicate_validation_id_and_recommit(tmp_path)

    with pytest.raises(CacheCorruptionError, match="duplicate"):
        CacheWriter.resume(tmp_path, expected_cache_id="cache")


def test_loader_rejects_duplicate_structure_ids_across_splits(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import CacheCorruptionError, load_complete_cache

    write_two_split_cache(tmp_path, complete=True)
    duplicate_validation_id_and_recommit(tmp_path)

    with pytest.raises(CacheCorruptionError, match="duplicate"):
        load_complete_cache(tmp_path, expected_cache_id="cache")


def test_iterator_rejects_duplicate_ids_recommitted_after_open(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import (
        CacheCorruptionError,
        iter_cache_batches,
        load_complete_cache,
    )

    write_two_split_cache(tmp_path, complete=True)
    manifest = load_complete_cache(tmp_path, expected_cache_id="cache")
    duplicate_validation_id_and_recommit(tmp_path)

    with pytest.raises(CacheCorruptionError, match="duplicate"):
        list(iter_cache_batches(manifest, "train", batch_size=1))


def test_resume_wraps_weights_only_rejection_as_cache_corruption(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    write_partial_cache(tmp_path)
    atomic_torch_save(tmp_path / "progress.pt", {"unsafe": UnsafePickle()})

    with pytest.raises(CacheCorruptionError, match="cannot load safely"):
        CacheWriter.resume(tmp_path, expected_cache_id="cache")


def _write_shuffle_cache(root: Path):
    from confidence_head.cache import CacheWriter

    writer = CacheWriter(root, cache_id="cache", shard_max_atoms=4)
    batch = batch_with_atom_counts([1, 2, 1, 3, 1])
    writer.append(
        replace(
            batch, atomic_numbers=torch.arange(1, 9, dtype=torch.long)
        )
    )
    writer.finalize_split("train")
    writer.append(
        batch_with_atom_counts(
            [2], structure_prefix="validation"
        ),
        split="validation",
    )
    writer.finalize_split("validation")
    return writer.finalize()


def _flatten_structure_ids(
    batches: list[ContinuousBatch],
) -> list[str]:
    return [
        structure_id
        for batch in batches
        for structure_id in batch.structure_id
    ]


def test_shuffled_iterator_is_reproducible_and_epoch_changes_order(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import iter_shuffled_cache_batches

    manifest = _write_shuffle_cache(tmp_path)
    first = list(
        iter_shuffled_cache_batches(manifest, "train", 2, seed=17)
    )
    repeated = list(
        iter_shuffled_cache_batches(manifest, "train", 2, seed=17)
    )
    next_epoch = list(
        iter_shuffled_cache_batches(manifest, "train", 2, seed=18)
    )

    assert [batch.structure_id for batch in first] == [
        batch.structure_id for batch in repeated
    ]
    assert all(
        torch.equal(getattr(left, field), getattr(right, field))
        for left, right in zip(first, repeated, strict=True)
        for field in (
            "structure_index",
            "atomic_numbers",
            "atom_offsets",
            "scalar_features",
            "force_prediction",
            "force_reference",
            "energy_prediction",
            "energy_reference",
        )
    )
    assert _flatten_structure_ids(first) != _flatten_structure_ids(
        next_epoch
    )


def test_shuffled_iterator_yields_each_complete_structure_once(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import (
        iter_cache_batches,
        iter_shuffled_cache_batches,
    )

    manifest = _write_shuffle_cache(tmp_path)
    sequential = list(iter_cache_batches(manifest, "train", 10))
    shuffled = list(
        iter_shuffled_cache_batches(manifest, "train", 2, seed=29)
    )

    assert sorted(_flatten_structure_ids(shuffled)) == sorted(
        _flatten_structure_ids(sequential)
    )
    assert len(_flatten_structure_ids(shuffled)) == 5
    assert all(len(batch.structure_id) <= 2 for batch in shuffled)
    atom_counts = {
        "structure-0": 1,
        "structure-1": 2,
        "structure-2": 1,
        "structure-3": 3,
        "structure-4": 1,
    }
    expected_atomic_numbers = {
        "structure-0": [1],
        "structure-1": [2, 3],
        "structure-2": [4],
        "structure-3": [5, 6, 7],
        "structure-4": [8],
    }
    for batch in shuffled:
        expected_offsets = [0]
        for structure_id in batch.structure_id:
            expected_offsets.append(
                expected_offsets[-1] + atom_counts[structure_id]
            )
        assert batch.atom_offsets.tolist() == expected_offsets
        for index, structure_id in enumerate(batch.structure_id):
            atom_start = batch.atom_offsets[index]
            atom_end = batch.atom_offsets[index + 1]
            assert batch.atomic_numbers[atom_start:atom_end].tolist() == (
                expected_atomic_numbers[structure_id]
            )
            assert int(batch.structure_index[index]) == int(structure_id[-1])


def test_shuffled_iterator_flushes_at_non_feature_dtype_boundary(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import (
        CacheWriter,
        iter_shuffled_cache_batches,
    )

    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=1)
    writer.append(approved_payload_batch([1], structure_prefix="first"))
    writer.append(
        batch_with_changed_field_dtype(
            field="energy_reference",
            index_start=1,
            structure_prefix="second",
        )
    )
    writer.finalize_split("train")
    manifest = writer.finalize()

    batches = list(
        iter_shuffled_cache_batches(manifest, "train", 10, seed=3)
    )

    assert len(batches) == 2
    assert {batch.energy_reference.dtype for batch in batches} == {
        torch.float32,
        torch.float64,
    }


def test_shuffled_iterator_does_not_mutate_global_rng_states(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import iter_shuffled_cache_batches

    manifest = _write_shuffle_cache(tmp_path)
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    list(iter_shuffled_cache_batches(manifest, "train", 2, seed=104))

    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_shuffled_iterator_audits_all_shards_before_streaming_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import confidence_head.cache as cache_module
    from confidence_head.cache import iter_shuffled_cache_batches

    manifest = _write_shuffle_cache(tmp_path)
    original_safe_load = cache_module._safe_load_torch
    original_iterator_load = cache_module._load_iterator_shard
    events: list[tuple[str, str, int]] = []

    def recording_safe_load(path: Path, where: str):
        if path.name.startswith("shard-"):
            events.append(
                ("load", path.parent.name, int(path.stem.split("-")[1]))
            )
        return original_safe_load(path, where)

    def recording_iterator_load(*args, **kwargs):
        shard = kwargs["shard"]
        events.append(("consume", kwargs["split"], shard.shard_index))
        return original_iterator_load(*args, **kwargs)

    monkeypatch.setattr(
        cache_module, "_safe_load_torch", recording_safe_load
    )
    monkeypatch.setattr(
        cache_module, "_load_iterator_shard", recording_iterator_load
    )

    iterator = iter_shuffled_cache_batches(manifest, "train", 2, seed=7)
    next(iterator)

    first_consume = next(
        index
        for index, event in enumerate(events)
        if event[0] == "consume"
    )
    audited = {
        (split, shard_index)
        for event, split, shard_index in events[:first_consume]
        if event == "load"
    }
    declared = {
        (split, shard.shard_index)
        for split, shards in manifest.splits.items()
        for shard in shards
    }
    assert audited == declared
    assert first_consume == len(declared)
    assert [
        event for event in events if event[0] == "consume"
    ] == [("consume", "train", 1)]

    list(iterator)

    consumed = [
        event for event in events if event[0] == "consume"
    ]
    assert sorted(consumed) == [
        ("consume", "train", 0),
        ("consume", "train", 1),
    ]
    assert len(consumed) == len(manifest.splits["train"])


@pytest.mark.parametrize("seed", [True, -1, 2**63])
def test_shuffled_iterator_rejects_invalid_seed(
    tmp_path: Path, seed: object
) -> None:
    from confidence_head.cache import iter_shuffled_cache_batches

    manifest = _write_shuffle_cache(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        list(
            iter_shuffled_cache_batches(
                manifest, "train", batch_size=2, seed=seed
            )
        )


def test_shuffled_iterator_rejects_unbound_manifest(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import (
        CacheCorruptionError,
        iter_shuffled_cache_batches,
    )

    manifest = _write_shuffle_cache(tmp_path)
    forged = replace(manifest, splits={})
    with pytest.raises(CacheCorruptionError, match="manifest"):
        list(
            iter_shuffled_cache_batches(
                forged, "train", batch_size=2, seed=0
            )
        )


def test_shuffled_iterator_empty_split_and_tampering_contracts(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import (
        CacheCorruptionError,
        iter_shuffled_cache_batches,
    )

    manifest = _write_shuffle_cache(tmp_path)
    assert list(
        iter_shuffled_cache_batches(
            manifest, "test", 2, seed=0
        )
    ) == []

    (tmp_path / "train" / "shard-000000.pt").write_bytes(b"corrupt")
    with pytest.raises(CacheCorruptionError):
        list(
            iter_shuffled_cache_batches(
                manifest, "train", 2, seed=0
            )
        )


def test_shuffled_iterator_rejects_non_target_corruption_before_first_yield(
    tmp_path: Path,
) -> None:
    from confidence_head.cache import (
        CacheCorruptionError,
        iter_shuffled_cache_batches,
    )

    manifest = _write_shuffle_cache(tmp_path)
    (tmp_path / "validation" / "shard-000000.pt").write_bytes(
        b"corrupt"
    )

    iterator = iter_shuffled_cache_batches(
        manifest, "train", batch_size=2, seed=7
    )
    with pytest.raises(CacheCorruptionError, match="validation"):
        next(iterator)


@pytest.mark.parametrize("batch_size", [True, 0])
def test_shuffled_iterator_rejects_invalid_batch_size(
    tmp_path: Path, batch_size: object
) -> None:
    from confidence_head.cache import iter_shuffled_cache_batches

    manifest = _write_shuffle_cache(tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        list(
            iter_shuffled_cache_batches(
                manifest, "train", batch_size=batch_size, seed=0
            )
        )
