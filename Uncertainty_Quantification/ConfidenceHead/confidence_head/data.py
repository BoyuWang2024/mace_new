"""Strict ASE-to-MACE data adaptation for ConfidenceHead training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import ase.io
import numpy as np
import torch
from ase import Atoms

from .errors import DataContractError
from .identity import sha256_file

if TYPE_CHECKING:
    from .backbone import BackboneIdentity
    from mace.tools.torch_geometric import Batch


ENERGY_KEY = "REF_energy"
FORCES_KEY = "REF_forces"


@dataclass(frozen=True)
class DatasetHandle:
    path: Path
    sha256: str
    structure_ids: tuple[str, ...]
    size: int


@dataclass
class StructureBatch:
    indices: torch.Tensor
    structure_ids: tuple[str, ...]
    num_atoms: torch.Tensor
    atomic_numbers: torch.Tensor
    atom_offsets: torch.Tensor
    mace_batch: Batch
    reference_energy: torch.Tensor
    reference_forces: torch.Tensor


def _array_payload(value: object) -> dict[str, object]:
    array = np.asarray(value)
    normalized_dtype = array.dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(array.astype(normalized_dtype, copy=False))
    return {
        "dtype": normalized.dtype.str,
        "shape": list(normalized.shape),
        "hex": normalized.tobytes(order="C").hex(),
    }


def _reference_values(atoms: Atoms, *, context: str) -> tuple[object, np.ndarray]:
    if ENERGY_KEY not in atoms.info:
        raise DataContractError(f"{context} is missing {ENERGY_KEY}")
    if FORCES_KEY not in atoms.arrays:
        raise DataContractError(f"{context} is missing {FORCES_KEY}")
    energy = atoms.info[ENERGY_KEY]
    energy_array = np.asarray(energy)
    forces = np.asarray(atoms.arrays[FORCES_KEY])
    if energy_array.shape != ():
        raise DataContractError(f"{context} {ENERGY_KEY} must be scalar")
    if forces.shape != (len(atoms), 3):
        raise DataContractError(
            f"{context} {FORCES_KEY} must have shape ({len(atoms)}, 3)"
        )
    if not np.isfinite(energy_array).all() or not np.isfinite(forces).all():
        raise DataContractError(f"{context} reference labels must be finite")
    return energy, forces


def structure_id(atoms: Atoms) -> str:
    """Hash all normalized structure and reference fields deterministically."""
    energy, forces = _reference_values(atoms, context="structure")
    payload = {
        "atomic_numbers": _array_payload(atoms.get_atomic_numbers()),
        "positions": _array_payload(atoms.get_positions()),
        "cell": _array_payload(atoms.cell.array),
        "pbc": _array_payload(atoms.get_pbc()),
        "reference_energy": _array_payload(energy),
        "reference_forces": _array_payload(forces),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_structures(path: Path) -> list[Atoms]:
    try:
        structures = ase.io.read(path, index=":")
    except Exception as error:
        raise DataContractError(f"could not read dataset {path}: {error}") from error
    if isinstance(structures, Atoms):
        structures = [structures]
    if not structures:
        raise DataContractError(f"dataset {path} contains no structures")
    return structures


def _validate_structure(
    atoms: Atoms, *, supported_atomic_numbers: set[int] | None, context: str
) -> None:
    if len(atoms) == 0:
        raise DataContractError(f"{context} contains no atoms")
    _reference_values(atoms, context=context)
    if supported_atomic_numbers is not None:
        for number in atoms.get_atomic_numbers():
            if int(number) not in supported_atomic_numbers:
                raise DataContractError(
                    f"{context} contains unsupported atomic number {int(number)}"
                )


def load_dataset(
    path: Path,
    expected_sha256: str,
    supported_atomic_numbers: Sequence[int],
) -> DatasetHandle:
    """Verify an extxyz dataset and return its stable structure identities."""
    resolved = Path(path).resolve()
    actual_sha256 = sha256_file(resolved)
    if actual_sha256.lower() != expected_sha256.lower():
        raise DataContractError(
            f"dataset SHA-256 mismatch for {resolved}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    structures = _read_structures(resolved)
    supported = {int(number) for number in supported_atomic_numbers}
    structure_ids = []
    for index, atoms in enumerate(structures):
        _validate_structure(
            atoms, supported_atomic_numbers=supported, context=f"structure {index}"
        )
        structure_ids.append(structure_id(atoms))
    return DatasetHandle(
        path=resolved,
        sha256=actual_sha256,
        structure_ids=tuple(structure_ids),
        size=len(structures),
    )


def _structure_ids(source: Path | DatasetHandle) -> tuple[str, ...]:
    if isinstance(source, DatasetHandle):
        return source.structure_ids
    structures = _read_structures(Path(source).resolve())
    for index, atoms in enumerate(structures):
        _validate_structure(
            atoms, supported_atomic_numbers=None, context=f"structure {index}"
        )
    return tuple(structure_id(atoms) for atoms in structures)


def validate_split_isolation(
    train: Path | DatasetHandle,
    validation: Path | DatasetHandle,
    test: Path | DatasetHandle,
    *,
    profile: str,
) -> None:
    """Reject structure-level overlap between production dataset splits."""
    if profile not in {"production", "smoke_test"}:
        raise DataContractError(f"unsupported data profile: {profile}")
    if profile != "production":
        return
    split_ids = {
        "train": set(_structure_ids(train)),
        "validation": set(_structure_ids(validation)),
        "test": set(_structure_ids(test)),
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            raise DataContractError(
                f"production split overlap between {left} and {right}: "
                f"{len(overlap)} structure(s)"
            )


def _convert_floating_tensors(data: Any, dtype: torch.dtype) -> None:
    for key in data.keys:
        value = data[key]
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            data[key] = value.to(dtype=dtype)


def build_structure_batch(
    structures: Sequence[Atoms],
    *,
    indices: Sequence[int],
    backbone: BackboneIdentity,
    device: str | torch.device = "cpu",
) -> StructureBatch:
    """Convert ordered ASE structures to a typed MACE graph batch."""
    from mace.data import AtomicData, KeySpecification, config_from_atoms
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric import Batch

    if len(structures) != len(indices):
        raise DataContractError("structure indices must align with the batch")
    if not structures:
        raise DataContractError("cannot build an empty structure batch")

    supported = set(backbone.atomic_numbers)
    z_table = AtomicNumberTable(list(backbone.atomic_numbers))
    keys = KeySpecification.from_defaults()
    graphs: list[AtomicData] = []
    energies: list[float] = []
    forces: list[np.ndarray] = []
    ids: list[str] = []
    counts: list[int] = []
    atomic_numbers: list[np.ndarray] = []
    for index, atoms in enumerate(structures):
        _validate_structure(
            atoms, supported_atomic_numbers=supported, context=f"structure {index}"
        )
        energy, reference_forces = _reference_values(
            atoms, context=f"structure {index}"
        )
        configuration = config_from_atoms(
            atoms, key_specification=keys, head_name=backbone.selected_head
        )
        graph = AtomicData.from_config(
            configuration,
            z_table=z_table,
            cutoff=backbone.r_max,
            heads=list(backbone.heads),
        )
        _convert_floating_tensors(graph, backbone.dtype)
        graphs.append(graph)
        energies.append(float(energy))
        forces.append(np.asarray(reference_forces))
        ids.append(structure_id(atoms))
        counts.append(len(atoms))
        atomic_numbers.append(np.asarray(atoms.get_atomic_numbers()))

    target_device = torch.device(device)
    num_atoms = torch.tensor(counts, dtype=torch.long, device=target_device)
    atom_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=target_device),
            torch.cumsum(num_atoms, dim=0),
        )
    )
    mace_batch = Batch.from_data_list(graphs).to(target_device)
    return StructureBatch(
        indices=torch.tensor(indices, dtype=torch.long, device=target_device),
        structure_ids=tuple(ids),
        num_atoms=num_atoms,
        atomic_numbers=torch.as_tensor(
            np.concatenate(atomic_numbers, axis=0),
            dtype=torch.long,
            device=target_device,
        ),
        atom_offsets=atom_offsets,
        mace_batch=mace_batch,
        reference_energy=torch.tensor(
            energies, dtype=backbone.dtype, device=target_device
        ),
        reference_forces=torch.as_tensor(
            np.concatenate(forces, axis=0),
            dtype=backbone.dtype,
            device=target_device,
        ),
    )
