"""Deterministic whole-structure compatibility filtering for MACE datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from .errors import HardFailure


@dataclass(frozen=True)
class CompatibilityResult:
    source: Path
    structures: tuple[object, ...]
    source_indices: np.ndarray
    excluded_indices: np.ndarray
    unsupported_atomic_numbers: tuple[int, ...]
    source_count: int

    @property
    def retained_count(self) -> int:
        return len(self.structures)


def filter_supported_structures(source: str | Path, supported_atomic_numbers: Sequence[int]) -> CompatibilityResult:
    try:
        from ase.io import read
        structures = read(Path(source).expanduser().resolve(), index=":")
    except (OSError, ValueError, TypeError) as error:
        raise HardFailure(f"could not read dataset {source}: {error}") from error
    if not isinstance(structures, list) or not structures:
        raise HardFailure("dataset must contain at least one structure")
    supported = {int(value) for value in supported_atomic_numbers}
    kept: list[object] = []
    kept_indices: list[int] = []
    excluded: list[int] = []
    unsupported: set[int] = set()
    for index, atoms in enumerate(structures):
        numbers = {int(value) for value in np.asarray(atoms.numbers, dtype=int)}
        missing = numbers - supported
        if missing:
            excluded.append(index)
            unsupported.update(missing)
        else:
            kept.append(atoms)
            kept_indices.append(index)
    if not kept:
        raise HardFailure("all structures contain unsupported elements")
    return CompatibilityResult(
        source=Path(source).expanduser().resolve(),
        structures=tuple(kept),
        source_indices=np.asarray(kept_indices, dtype=np.int64),
        excluded_indices=np.asarray(excluded, dtype=np.int64),
        unsupported_atomic_numbers=tuple(sorted(unsupported)),
        source_count=len(structures),
    )


def extract_targets(structures: Sequence[object], *, require_atomization: bool = True) -> dict[str, np.ndarray]:
    if not structures:
        raise HardFailure("target dataset is empty")
    ids: list[str] = []
    counts: list[int] = []
    energies: list[float] = []
    atomization: list[float] = []
    forces: list[np.ndarray] = []
    for index, atoms in enumerate(structures):
        results = getattr(getattr(atoms, "calc", None), "results", {})
        info = getattr(atoms, "info", {})
        try:
            energy = float(results["energy"])
            force = np.asarray(results["forces"], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise HardFailure(f"structure {index} lacks energy/forces labels") from error
        if require_atomization:
            try:
                atomization_value = float(info["atomization_energy"])
            except (KeyError, TypeError, ValueError) as error:
                raise HardFailure(f"structure {index} lacks atomization_energy") from error
            atomization.append(atomization_value)
        if force.shape != (len(atoms), 3) or not np.isfinite(force).all():
            raise HardFailure(f"structure {index} has invalid forces")
        if not np.isfinite(energy) or (require_atomization and not np.isfinite(atomization[-1])):
            raise HardFailure(f"structure {index} has non-finite energy labels")
        ids.append(str(info.get("structure_id", info.get("configuration_id", index))))
        counts.append(len(atoms))
        energies.append(energy)
        forces.append(force)
    counts_array = np.asarray(counts, dtype=np.int64)
    output: dict[str, np.ndarray] = {
        "structure_ids": np.asarray(ids, dtype=str),
        "num_atoms": counts_array,
        "atom_offsets": np.concatenate(([0], np.cumsum(counts_array))),
        "energy": np.asarray(energies, dtype=float),
        "forces": np.concatenate(forces, axis=0),
    }
    if require_atomization:
        output["atomization_energy"] = np.asarray(atomization, dtype=float)
    return output


__all__ = ["CompatibilityResult", "extract_targets", "filter_supported_structures"]
