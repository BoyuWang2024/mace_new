"""Source-independent extxyz validation and MACE data-loader construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import Any

import ase.io
import numpy as np

from mace.data import AtomicData, KeySpecification, load_from_xyz
from mace.tools import AtomicNumberTable, torch_geometric

from .errors import HardFailure
from .extxyz_standard import read_energy, read_forces, read_stress


_OBSERVABLES = frozenset({"energy", "forces", "stress"})


def load_extxyz(
    path: Path,
    *,
    keys: Mapping[str, str],
    required: Set[str],
    head_name: str = "Default",
) -> list[Any]:
    """Validate required fields for every structure, then use MACE's xyz parser."""
    path = Path(path)
    if path.suffix.lower() not in {".xyz", ".extxyz"} or not path.is_file():
        raise HardFailure("input must be an existing extxyz file")
    if not required or not set(required) <= _OBSERVABLES:
        raise HardFailure("required observables are invalid")
    if not all(isinstance(keys.get(name), str) and keys[name] for name in (*_OBSERVABLES, "head")):
        raise HardFailure("energy, forces, stress, and head keys are required")
    try:
        atoms_list = ase.io.read(path, index=":", format="extxyz")
    except Exception as exc:
        raise HardFailure("extxyz cannot be parsed") from exc
    if not atoms_list:
        raise HardFailure("extxyz contains no structures")
    for index, atoms in enumerate(atoms_list):
        if "energy" in required:
            value = read_energy(atoms, keys)
            if value is None or np.asarray(value).size != 1 or not np.isfinite(value).all():
                raise HardFailure(f"structure {index} has invalid energy")
        if "forces" in required:
            value = read_forces(atoms, keys)
            if value is None or np.asarray(value).shape != (len(atoms), 3) or not np.isfinite(value).all():
                raise HardFailure(f"structure {index} has invalid forces")
        if "stress" in required:
            value = read_stress(atoms, keys)
            if value is None or np.asarray(value).shape not in {(6,), (3, 3)} or not np.isfinite(value).all():
                raise HardFailure(f"structure {index} has invalid stress")
    specification = KeySpecification(
        info_keys={"energy": keys["energy"], "stress": keys["stress"], "head": keys["head"]},
        arrays_keys={"forces": keys["forces"]},
    )
    try:
        _, configurations = load_from_xyz(
            str(path), specification, head_name=head_name, no_data_ok=False
        )
    except Exception as exc:
        raise HardFailure("MACE could not load validated extxyz") from exc
    return list(configurations)


def build_mace_loaders(
    configurations: Sequence[Any],
    *,
    atomic_numbers: Sequence[int],
    cutoff: float,
    batch_size: int,
    shuffle: bool = False,
    heads: list[str] | None = None,
) -> Any:
    """Convert MACE configurations into the native graph DataLoader."""
    if not configurations:
        raise HardFailure("cannot build a loader from no configurations")
    if cutoff <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise HardFailure("cutoff and batch_size must be positive")
    z_table = AtomicNumberTable(list(atomic_numbers))
    atomic_data = [
        AtomicData.from_config(config, z_table=z_table, cutoff=cutoff, heads=heads)
        for config in configurations
    ]
    return torch_geometric.dataloader.DataLoader(
        dataset=atomic_data, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )
