"""Dataset structure identities and flattened atom layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .errors import HardFailure


@dataclass(frozen=True)
class DatasetLayout:
    structure_ids: NDArray[np.str_]
    num_atoms: NDArray[np.int64]
    atom_offsets: NDArray[np.int64]

    @property
    def structure_count(self) -> int:
        return int(self.num_atoms.size)

    @property
    def atom_count(self) -> int:
        return int(self.atom_offsets[-1])


def layout_from_atoms(atoms: Sequence[object], *, structure_ids: Sequence[str] | None = None) -> DatasetLayout:
    if not atoms:
        raise HardFailure("dataset must contain at least one structure")
    ids = list(structure_ids) if structure_ids is not None else [f"structure_{index:06d}" for index in range(len(atoms))]
    if len(ids) != len(atoms):
        raise HardFailure("structure_ids length must match dataset")
    if len(set(ids)) != len(ids):
        raise HardFailure("structure_ids must be unique")
    counts = np.asarray([len(value) for value in atoms], dtype=np.int64)
    if np.any(counts < 1):
        raise HardFailure("every structure must contain at least one atom")
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    return DatasetLayout(np.asarray(ids, dtype=np.str_), counts, offsets)


def load_dataset_layout(path: str | Path) -> DatasetLayout:
    source = Path(path).expanduser().resolve()
    try:
        from ase.io import read
        structures = read(source, index=":")
    except (OSError, ValueError, TypeError) as error:
        raise HardFailure(f"could not read extxyz dataset {source}: {error}") from error
    ids = [str(atoms.info.get("structure_id", f"structure_{index:06d}")) for index, atoms in enumerate(structures)]
    return layout_from_atoms(structures, structure_ids=ids)
