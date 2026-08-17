from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from confidence_head.identity import sha256_file
from confidence_head.workflows.prepare_external_dataset import prepare_compatible_dataset


def test_prepare_accepts_ase_calculator_reference_labels(tmp_path: Path) -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(
        atoms, energy=1.0, forces=np.zeros((1, 3), dtype=float)
    )
    source = tmp_path / "calculator.xyz"
    ase.io.write(source, [atoms], format="extxyz")

    result = prepare_compatible_dataset(
        source_path=source,
        output_path=tmp_path / "compatible.xyz",
        unsupported_atomic_numbers=(84, 86),
        expected_source_sha256=sha256_file(source),
    )

    assert result.manifest["compatible_structures"] == 1
