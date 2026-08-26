from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.scripts.run_reference_experiments import _run


def _write_dataset(path: Path, *, include_unsupported: bool = False) -> None:
    structures = []
    for index, symbols in enumerate(("H", "He")):
        atoms = Atoms(symbols, positions=np.zeros((1, 3)))
        atoms.info["structure_id"] = f"structure-{index}"
        atoms.info["atomization_energy"] = 0.0
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-float(index + 1),
            forces=np.zeros((len(atoms), 3)),
        )
        structures.append(atoms)
    if include_unsupported:
        atoms = Atoms("Li", positions=[[0.0, 0.0, 0.0]])
        atoms.info["structure_id"] = "unsupported"
        atoms.info["atomization_energy"] = 0.0
        atoms.calc = SinglePointCalculator(atoms, energy=-3.0, forces=np.zeros((1, 3)))
        structures.append(atoms)
    write(path, structures, format="extxyz")


def _write_inference(root: Path, *, count: int = 2) -> None:
    root.mkdir(parents=True)
    for index in range(8):
        np.savez(
            root / f"member_{index:03d}.npz",
            energy=np.arange(count, dtype=float) - float(index),
            forces=np.zeros((count, 3)),
        )


def _write_model_and_config(
    tmp_path: Path,
    *,
    include_unsupported: bool = False,
    canonical: bool = False,
) -> Path:
    dataset = tmp_path / "dataset.extxyz"
    _write_dataset(dataset, include_unsupported=include_unsupported)
    inference = tmp_path / "inference.final"
    _write_inference(inference, count=3 if include_unsupported else 2)
    section = {
        "targets": str(dataset),
        "matrix": str(tmp_path / "matrix.npz"),
        "members": [str(inference / f"member_{index:03d}.npz") for index in range(8)],
    }
    if not canonical:
        section = {"dataset": str(dataset), "inference": str(inference)}
    config = {
        "schema_version": 1,
        "model": {"atomic_numbers": [1, 2], "e0": [-1.0, -2.0]},
        "validation": section,
        "test": section,
        "output_root": str(tmp_path / "output"),
    }
    if canonical:
        np.savez(tmp_path / "matrix.npz", matrix=np.asarray([[1.0, 0.0], [0.0, 1.0]]))
        targets = {
            "structure_ids": np.asarray(["structure-0", "structure-1"]),
            "num_atoms": np.asarray([1, 1], dtype=np.int64),
            "energy": np.asarray([-1.0, -2.0]),
            "atomization_energy": np.zeros(2),
            "forces": np.zeros((2, 3)),
        }
        np.savez(tmp_path / "targets.npz", **targets)
        config["validation"]["targets"] = str(tmp_path / "targets.npz")
        config["test"]["targets"] = str(tmp_path / "targets.npz")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_cli_accepts_dataset_and_completed_inference_sections(tmp_path: Path) -> None:
    config_path = _write_model_and_config(tmp_path)

    output = _run(config_path)

    assert (output / "direct_test_mad_e0" / "manifest.json").is_file()
    corrected = np.load(output / "direct_test_mad_e0" / "corrected_member_energies.npz")
    assert corrected["member_energies"].shape == (8, 2)


def test_cli_hard_fails_when_dataset_filter_would_change_prediction_alignment(tmp_path: Path) -> None:
    config_path = _write_model_and_config(tmp_path, include_unsupported=True)

    with pytest.raises(HardFailure, match="retained_count.*source_count"):
        _run(config_path)


def test_cli_keeps_canonical_targets_matrix_members_mode(tmp_path: Path) -> None:
    config_path = _write_model_and_config(tmp_path, canonical=True)

    output = _run(config_path)

    assert (output / "model_aware_val_fit" / "manifest.json").is_file()
