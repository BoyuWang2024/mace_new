from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from Uncertainty_Quantification.BootStrapping.bootstrap.data import layout_from_atoms
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure


def test_dataset_layout_flattens_atoms_without_losing_structure_boundaries() -> None:
    layout = layout_from_atoms(
        [Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]), Atoms("He", positions=[[1, 0, 0]])],
        structure_ids=["a", "b"],
    )
    np.testing.assert_array_equal(layout.num_atoms, [2, 1])
    np.testing.assert_array_equal(layout.atom_offsets, [0, 2, 3])
    assert layout.structure_ids.tolist() == ["a", "b"]


def test_dataset_layout_rejects_duplicate_ids() -> None:
    with pytest.raises(HardFailure, match="unique"):
        layout_from_atoms([Atoms("H"), Atoms("H")], structure_ids=["same", "same"])
