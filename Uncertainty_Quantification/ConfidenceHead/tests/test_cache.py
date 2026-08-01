"""Contracts for durable, versioned ConfidenceHead feature caches."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from confidence_head.artifacts import (
    atomic_json_dump,
    atomic_torch_save,
    load_torch_artifact,
)
from confidence_head.features import ContinuousBatch
from confidence_head.identity import sha256_file


def batch_with_atom_counts(
    counts: list[int],
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    index_start: int = 0,
    structure_prefix: str = "structure",
) -> ContinuousBatch:
    num_atoms = torch.tensor(counts, dtype=torch.long, device=device)
    atom_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            num_atoms.cumsum(0),
        )
    )
    total_atoms = int(atom_offsets[-1])
    return ContinuousBatch(
        indices=torch.arange(
            index_start,
            index_start + len(counts),
            dtype=torch.long,
            device=device,
        ),
        structure_ids=tuple(
            f"{structure_prefix}-{index}" for index in range(len(counts))
        ),
        num_atoms=num_atoms,
        atom_offsets=atom_offsets,
        features=torch.arange(
            total_atoms * 640, dtype=dtype, device=device
        ).reshape(total_atoms, 640),
        reference_energy=torch.arange(
            len(counts), dtype=dtype, device=device
        ),
        reference_forces=torch.full(
            (total_atoms, 3), 0.125, dtype=dtype, device=device
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
    shard["structure_ids"] = ("structure-0",)
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
    broken = replace(batch, features=torch.cat((batch.features, batch.features[:1])))
    with pytest.raises(CacheCorruptionError):
        CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=10).append(broken)


@pytest.mark.parametrize(
    "case",
    [
        "structure_ids_length",
        "num_atoms_length",
        "indices_length",
        "offsets_length",
        "energy_length",
        "feature_rows",
        "force_rows",
        "offsets_start",
        "offsets_monotonic",
        "offsets_terminal",
        "positive_counts",
        "feature_width",
        "force_width",
        "finite_energy",
        "finite_forces",
        "indices_dtype",
        "num_atoms_dtype",
        "offsets_dtype",
        "floating_dtype",
        "floating_device",
    ],
)
def test_append_rejects_every_malformed_batch_boundary(
    tmp_path: Path, case: str
) -> None:
    from confidence_head.cache import CacheCorruptionError, CacheWriter

    batch = batch_with_atom_counts([2, 2])
    if case == "structure_ids_length":
        broken = replace(batch, structure_ids=batch.structure_ids + ("extra",))
    elif case == "num_atoms_length":
        broken = replace(
            batch,
            num_atoms=torch.cat((batch.num_atoms, torch.tensor([1]))),
        )
    elif case == "indices_length":
        broken = replace(
            batch,
            indices=torch.cat((batch.indices, torch.tensor([2]))),
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
            reference_energy=torch.cat(
                (batch.reference_energy, batch.reference_energy[:1])
            ),
        )
    elif case == "feature_rows":
        broken = replace(
            batch,
            features=torch.cat((batch.features, batch.features[:1])),
        )
    elif case == "force_rows":
        broken = replace(
            batch,
            reference_forces=torch.cat(
                (batch.reference_forces, batch.reference_forces[:1])
            ),
        )
    elif case == "offsets_start":
        broken = replace(batch, atom_offsets=torch.tensor([1, 2, 4]))
    elif case == "offsets_monotonic":
        broken = replace(batch, atom_offsets=torch.tensor([0, 3, 2]))
    elif case == "offsets_terminal":
        broken = replace(batch, atom_offsets=torch.tensor([0, 2, 3]))
    elif case == "positive_counts":
        broken = replace(
            batch,
            num_atoms=torch.tensor([2, 0]),
            atom_offsets=torch.tensor([0, 2, 2]),
        )
    elif case == "feature_width":
        broken = replace(batch, features=batch.features[:, :-1])
    elif case == "force_width":
        broken = replace(
            batch, reference_forces=batch.reference_forces[:, :-1]
        )
    elif case == "finite_energy":
        values = batch.reference_energy.clone()
        values[-1] = float("nan")
        broken = replace(batch, reference_energy=values)
    elif case == "finite_forces":
        values = batch.reference_forces.clone()
        values[-1, -1] = float("inf")
        broken = replace(batch, reference_forces=values)
    elif case == "indices_dtype":
        broken = replace(batch, indices=batch.indices.to(torch.int32))
    elif case == "num_atoms_dtype":
        broken = replace(batch, num_atoms=batch.num_atoms.to(torch.int32))
    elif case == "offsets_dtype":
        broken = replace(
            batch, atom_offsets=batch.atom_offsets.to(torch.int32)
        )
    elif case == "floating_dtype":
        broken = replace(
            batch, reference_forces=batch.reference_forces.float()
        )
    elif case == "floating_device":
        broken = replace(
            batch,
            reference_energy=torch.empty(
                batch.reference_energy.shape,
                dtype=batch.reference_energy.dtype,
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
    energies = batch.reference_energy.clone()
    energies[-1] = float("nan")

    with pytest.raises(CacheCorruptionError):
        writer.append(replace(batch, reference_energy=energies))

    writer.append(batch)
    writer.finalize_split("train")
    assert load_torch_artifact(
        tmp_path / "train" / "shard-000000.pt"
    )["structure_ids"] == batch.structure_ids


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
