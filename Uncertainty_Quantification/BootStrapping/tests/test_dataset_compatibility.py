from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from Uncertainty_Quantification.BootStrapping.bootstrap.dataset_compatibility import extract_targets, filter_supported_structures


def _write_dataset(path: Path) -> None:
    from ase.io import write
    keep = Atoms("HHe", positions=[[0, 0, 0], [0, 0, 1]])
    keep.info["structure_id"] = "keep"
    keep.info["atomization_energy"] = -1.0
    keep.calc = SinglePointCalculator(keep, energy=-2.0, forces=np.zeros((2, 3)))
    drop = Atoms("Li", positions=[[0, 0, 0]])
    drop.info["structure_id"] = "drop"
    drop.info["atomization_energy"] = -0.5
    drop.calc = SinglePointCalculator(drop, energy=-1.0, forces=np.zeros((1, 3)))
    write(path, [keep, drop], format="extxyz")


def test_filter_removes_whole_unsupported_structure(tmp_path: Path) -> None:
    source = tmp_path / "data.extxyz"
    _write_dataset(source)
    result = filter_supported_structures(source, [1, 2])
    assert result.source_count == 2
    assert result.retained_count == 1
    np.testing.assert_array_equal(result.source_indices, [0])
    np.testing.assert_array_equal(result.excluded_indices, [1])
    targets = extract_targets(result.structures)
    assert targets["structure_ids"].tolist() == ["keep"]
    np.testing.assert_allclose(targets["energy"], [-2.0])
