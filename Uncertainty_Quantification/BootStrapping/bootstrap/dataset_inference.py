"""Stable, domain-aware dataset plans for completed-model inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from ase import Atoms
from ase.io import iread

from .artifacts import sha256_file
from .errors import HardFailure


@dataclass(frozen=True)
class DatasetChunk:
    index: int
    structure_range: tuple[int, int]
    atom_count: int
    identity_sha256: str


@dataclass(frozen=True)
class DatasetPlan:
    source: Path
    source_sha256: str
    domains: tuple[str, ...]
    structure_count: int
    atom_count: int
    chunks: tuple[DatasetChunk, ...]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HardFailure(f"{name} must be a positive integer")
    return value


def _validate_domains(domains: tuple[str, ...]) -> tuple[str, ...]:
    if domains not in {("energy", "forces"), ("energy", "forces", "stress")}:
        raise HardFailure("domains must be energy/forces with optional stress")
    return domains


def _results(atoms: Atoms, structure_index: int, domains: tuple[str, ...]) -> None:
    calculator = atoms.calc
    results = getattr(calculator, "results", None)
    if not isinstance(results, dict):
        raise HardFailure(f"structure {structure_index} has no calculator labels")
    energy = np.asarray(results.get("energy"))
    if energy.shape != () or not np.isfinite(energy).all():
        raise HardFailure(f"structure {structure_index} has no finite energy label")
    forces = np.asarray(results.get("forces"))
    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise HardFailure(f"structure {structure_index} has no finite forces label")
    if "stress" in domains:
        if "stress" not in results:
            raise HardFailure(f"structure {structure_index} has no stress label")
        stress = np.asarray(results["stress"])
        if stress.shape not in {(6,), (3, 3)} or not np.isfinite(stress).all():
            raise HardFailure(f"structure {structure_index} has an invalid stress label")


def _structure_digest(index: int, atoms: Atoms) -> bytes:
    digest = hashlib.sha256()
    digest.update(index.to_bytes(8, "little", signed=False))
    digest.update("\0".join(atoms.get_chemical_symbols()).encode("ascii"))
    for array in (
        np.asarray(atoms.positions, dtype=np.float64),
        np.asarray(atoms.cell.array, dtype=np.float64),
        np.asarray(atoms.pbc, dtype=np.uint8),
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    structure_id = atoms.info.get("structure_id")
    subset = atoms.info.get("subset")
    digest.update(repr((structure_id, subset)).encode("utf-8"))
    return digest.digest()


def _structures(source: Path) -> Iterator[Atoms]:
    try:
        yield from iread(source, index=":")
    except Exception as error:
        raise HardFailure(f"could not stream dataset {source}: {error}") from error


def inspect_dataset(
    source: str | Path,
    *,
    domains: tuple[str, ...],
    max_structures_per_chunk: int,
    max_atoms_per_chunk: int,
) -> DatasetPlan:
    """Audit requested labels and produce deterministic contiguous chunk ranges."""
    source_path = Path(source).expanduser().resolve()
    if source_path.is_symlink() or not source_path.is_file():
        raise HardFailure(f"dataset source must be a regular file: {source_path}")
    selected_domains = _validate_domains(domains)
    structure_limit = _positive_int(max_structures_per_chunk, "max_structures_per_chunk")
    atom_limit = _positive_int(max_atoms_per_chunk, "max_atoms_per_chunk")

    chunks: list[DatasetChunk] = []
    chunk_start = 0
    chunk_atoms = 0
    chunk_structures = 0
    chunk_digest = hashlib.sha256()
    total_atoms = 0
    structure_count = 0

    def close_chunk(stop: int) -> None:
        nonlocal chunk_start, chunk_atoms, chunk_structures, chunk_digest
        if chunk_structures == 0:
            return
        chunks.append(
            DatasetChunk(
                index=len(chunks),
                structure_range=(chunk_start, stop),
                atom_count=chunk_atoms,
                identity_sha256=chunk_digest.hexdigest(),
            )
        )
        chunk_start = stop
        chunk_atoms = 0
        chunk_structures = 0
        chunk_digest = hashlib.sha256()

    for structure_index, atoms in enumerate(_structures(source_path)):
        _results(atoms, structure_index, selected_domains)
        atom_count = len(atoms)
        if atom_count < 1:
            raise HardFailure(f"structure {structure_index} has no atoms")
        if chunk_structures and (
            chunk_structures + 1 > structure_limit or chunk_atoms + atom_count > atom_limit
        ):
            close_chunk(structure_index)
        chunk_digest.update(_structure_digest(structure_index, atoms))
        chunk_structures += 1
        chunk_atoms += atom_count
        total_atoms += atom_count
        structure_count = structure_index + 1
    close_chunk(structure_count)
    if structure_count == 0:
        raise HardFailure("dataset contains no structures")
    if chunks[-1].structure_range[1] != structure_count:
        raise HardFailure("dataset chunk plan is incomplete")
    return DatasetPlan(
        source=source_path,
        source_sha256=sha256_file(source_path),
        domains=selected_domains,
        structure_count=structure_count,
        atom_count=total_atoms,
        chunks=tuple(chunks),
    )


__all__ = ["DatasetChunk", "DatasetPlan", "inspect_dataset"]
