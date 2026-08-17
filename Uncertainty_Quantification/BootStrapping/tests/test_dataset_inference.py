from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from Uncertainty_Quantification.BootStrapping.bootstrap.dataset_inference import (
    inspect_dataset,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure


def _dataset(path: Path, *, stress_flags: tuple[bool, ...]) -> Path:
    structures = []
    for index, (count, with_stress) in enumerate(zip((2, 2, 1), stress_flags)):
        atoms = Atoms(
            "H" * count,
            positions=np.arange(count * 3, dtype=float).reshape(count, 3) * 0.1,
            cell=np.eye(3) * (4.0 + index),
            pbc=True,
        )
        results: dict[str, object] = {
            "energy": -float(index + 1),
            "forces": np.full((count, 3), index + 0.25),
        }
        if with_stress:
            results["stress"] = np.arange(6, dtype=float) + index
        atoms.calc = SinglePointCalculator(atoms, **results)
        atoms.info["structure_id"] = 100 + index
        structures.append(atoms)
    write(path, structures, format="extxyz")
    return path


def test_energy_force_plan_preserves_order_and_dual_limits(tmp_path: Path) -> None:
    source = _dataset(tmp_path / "mad.xyz", stress_flags=(False, True, False))
    plan = inspect_dataset(
        source,
        domains=("energy", "forces"),
        max_structures_per_chunk=2,
        max_atoms_per_chunk=64,
    )

    assert plan.source == source.resolve()
    assert plan.domains == ("energy", "forces")
    assert plan.structure_count == 3
    assert plan.atom_count == 5
    assert [chunk.structure_range for chunk in plan.chunks] == [(0, 2), (2, 3)]
    assert [chunk.atom_count for chunk in plan.chunks] == [4, 1]
    assert [chunk.index for chunk in plan.chunks] == [0, 1]
    assert all(len(chunk.identity_sha256) == 64 for chunk in plan.chunks)
    assert len(plan.source_sha256) == 64


def test_atom_limit_starts_a_new_chunk_without_reordering(tmp_path: Path) -> None:
    source = _dataset(tmp_path / "data.extxyz", stress_flags=(True, True, True))
    plan = inspect_dataset(
        source,
        domains=("energy", "forces", "stress"),
        max_structures_per_chunk=10,
        max_atoms_per_chunk=3,
    )

    assert [chunk.structure_range for chunk in plan.chunks] == [(0, 1), (1, 3)]
    assert [chunk.atom_count for chunk in plan.chunks] == [2, 3]


def test_stress_request_rejects_any_missing_stress_label(tmp_path: Path) -> None:
    source = _dataset(tmp_path / "mad.xyz", stress_flags=(True, False, True))
    with pytest.raises(HardFailure, match="structure 1.*stress label"):
        inspect_dataset(
            source,
            domains=("energy", "forces", "stress"),
            max_structures_per_chunk=10,
            max_atoms_per_chunk=100,
        )


def test_dataset_plan_rejects_invalid_domain_order(tmp_path: Path) -> None:
    source = _dataset(tmp_path / "data.extxyz", stress_flags=(True, True, True))
    with pytest.raises(HardFailure, match="domains"):
        inspect_dataset(
            source,
            domains=("forces", "energy"),
            max_structures_per_chunk=10,
            max_atoms_per_chunk=100,
        )
