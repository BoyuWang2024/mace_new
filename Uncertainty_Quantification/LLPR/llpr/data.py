"""Streaming extxyz adaptation for LLPR."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from ase.io import iread

from mace.data import AtomicData, KeySpecification, config_from_atoms
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric import Batch

from .artifacts import sha256_file, stable_id


@dataclass(frozen=True)
class DatasetHandle:
    path: Path
    sha256: str
    identity: str
    size: int
    atomic_numbers: tuple[int, ...]
    r_max: float
    head: str


@dataclass
class StructureSample:
    index: int
    structure_id: str
    num_atoms: int
    batch: Batch
    reference_energy_per_atom: torch.Tensor
    reference_forces: torch.Tensor


_REFERENCE_KEYS = KeySpecification(
    info_keys={"energy": "REF_energy"},
    arrays_keys={"forces": "REF_forces"},
)
_STRUCTURE_ID_KEYS = ("structure_id", "config_id", "id")


def _expose_ase_reference_fields(atoms: object) -> None:
    """Expose ASE-reserved extxyz results under the LLPR reference keys."""
    results = getattr(getattr(atoms, "calc", None), "results", {})
    if "REF_energy" not in atoms.info and "energy" in results:
        atoms.info["REF_energy"] = results["energy"]
    if "REF_forces" not in atoms.arrays and "forces" in results:
        atoms.arrays["REF_forces"] = results["forces"]


def build_dataset(
    path: Path,
    expected_sha256: str | None,
    atomic_numbers: Sequence[int],
    r_max: float,
    head: str = "default",
) -> DatasetHandle:
    """Validate an extxyz source and return a small, read-only stream handle."""
    source_path = Path(path)
    actual_sha256 = sha256_file(source_path)
    if (
        expected_sha256 is not None
        and actual_sha256.lower() != expected_sha256.lower()
    ):
        raise ValueError(
            f"dataset SHA256 mismatch: expected {expected_sha256}, actual {actual_sha256}"
        )
    numbers = tuple(int(number) for number in atomic_numbers)
    size = sum(1 for _ in iread(source_path, index=":"))
    identity = stable_id(
        {
            "sha256": actual_sha256,
            "atomic_numbers": numbers,
            "r_max": float(r_max),
            "head": head,
        }
    )
    return DatasetHandle(
        path=source_path,
        sha256=actual_sha256,
        identity=identity,
        size=size,
        atomic_numbers=numbers,
        r_max=float(r_max),
        head=head,
    )


def _structure_id(atoms: object, index: int) -> str:
    info = atoms.info
    for key in _STRUCTURE_ID_KEYS:
        value = info.get(key)
        if value is not None and str(value):
            return str(value)
    return str(index)


def iter_samples(
    dataset: DatasetHandle,
    device: torch.device,
    dtype: torch.dtype,
    start_index: int = 0,
    max_structures: int | None = None,
) -> Iterator[StructureSample]:
    """Yield one device-local MACE batch per extxyz frame."""
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if max_structures is not None and max_structures <= 0:
        raise ValueError("max_structures must be positive or None")

    z_table = AtomicNumberTable(list(dataset.atomic_numbers))
    yielded = 0
    for index, atoms in enumerate(iread(dataset.path, index=":")):
        if index < start_index:
            continue
        if max_structures is not None and yielded >= max_structures:
            break
        _expose_ase_reference_fields(atoms)
        config = config_from_atoms(
            atoms,
            key_specification=_REFERENCE_KEYS,
            head_name=dataset.head,
        )
        if config.properties["energy"] is None:
            raise ValueError(f"structure {index} is missing REF_energy")
        if config.properties["forces"] is None:
            raise ValueError(f"structure {index} is missing REF_forces")
        atomic_data = AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=dataset.r_max,
            heads=[dataset.head],
        )
        batch = Batch.from_data_list([atomic_data])
        for key in batch.keys:
            value = batch[key]
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(
                    device=device,
                    dtype=dtype if value.is_floating_point() else value.dtype,
                )
        num_atoms = len(atoms)
        yield StructureSample(
            index=index,
            structure_id=_structure_id(atoms, index),
            num_atoms=num_atoms,
            batch=batch,
            reference_energy_per_atom=batch.energy[0] / num_atoms,
            reference_forces=batch.forces,
        )
        yielded += 1
