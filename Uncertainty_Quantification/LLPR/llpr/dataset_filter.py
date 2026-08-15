"""Filter extxyz structures that contain atoms without a cutoff neighbour."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from ase import Atoms
from ase.io import iread
from ase.io import read as ase_read
from ase.io import write as ase_write
from matscipy.neighbours import neighbour_list

from .artifacts import sha256_file


def missing_neighbor_indices(atoms: Atoms, cutoff: float) -> tuple[int, ...]:
    """Return indices of atoms that have no neighbour within ``cutoff``."""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    neighbour_atoms = atoms.copy()
    neighbour_atoms.set_cell(neighbour_atoms.cell.complete())
    neighbour_indices = neighbour_list("i", neighbour_atoms, cutoff)
    atoms_with_neighbours = {int(index) for index in neighbour_indices}
    return tuple(index for index in range(len(atoms)) if index not in atoms_with_neighbours)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _audit_record(
    atoms: Atoms, source_index: int, missing_indices: tuple[int, ...]
) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "structure_id": str(atoms.info.get("structure_id", source_index)),
        "num_atoms": len(atoms),
        "missing_neighbor_indices": list(missing_indices),
        "pbc": [bool(value) for value in atoms.pbc],
        "elements": sorted(set(atoms.get_chemical_symbols())),
        "reason": "missing_neighbor_within_cutoff",
    }


def filter_neighborless_extxyz(
    source: str | Path,
    output: str | Path,
    audit: str | Path,
    *,
    cutoff: float,
) -> dict[str, Any]:
    """Stream-filter an extxyz file and atomically emit its audit report."""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    source_path = Path(source)
    output_path = Path(output)
    audit_path = Path(audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = _temporary_path(output_path)
    temporary_audit = _temporary_path(audit_path)
    excluded: list[dict[str, Any]] = []
    total_structures = 0
    retained_structures = 0

    try:
        for source_index, atoms in enumerate(iread(source_path, index=":", format="extxyz")):
            total_structures += 1
            missing_indices = missing_neighbor_indices(atoms, cutoff)
            if missing_indices:
                excluded.append(_audit_record(atoms, source_index, missing_indices))
                continue
            ase_write(
                temporary_output,
                atoms,
                format="extxyz",
                append=retained_structures > 0,
            )
            retained_structures += 1

        if retained_structures == 0:
            temporary_output.touch()
        verified_structures = ase_read(temporary_output, index=":", format="extxyz")
        if len(verified_structures) != retained_structures:
            raise ValueError("temporary extxyz verification count mismatch")

        report: dict[str, Any] = {
            "source_sha256": sha256_file(source_path),
            "output_sha256": sha256_file(temporary_output),
            "cutoff": cutoff,
            "total_structures": total_structures,
            "retained_structures": retained_structures,
            "excluded_structures": len(excluded),
            "excluded": excluded,
        }
        temporary_audit.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if json.loads(temporary_audit.read_text(encoding="utf-8")) != report:
            raise ValueError("temporary audit verification mismatch")

        temporary_output.replace(output_path)
        temporary_audit.replace(audit_path)
        return report
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_audit.unlink(missing_ok=True)
