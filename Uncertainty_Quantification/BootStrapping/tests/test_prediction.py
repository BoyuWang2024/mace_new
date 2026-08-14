from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
    PredictionArrays,
    TargetArrays,
    load_prediction_arrays,
    load_target_arrays,
    validate_predictions,
    write_prediction_arrays,
    write_target_arrays,
)


def _targets() -> TargetArrays:
    return TargetArrays(
        structure_ids=np.asarray(["a", "b"]),
        num_atoms=np.asarray([2, 1], dtype=np.int64),
        atom_offsets=np.asarray([0, 2, 3], dtype=np.int64),
        energy=np.asarray([1.0, 2.0]),
        forces=np.arange(9, dtype=float).reshape(3, 3),
        stress=np.arange(12, dtype=float).reshape(2, 6),
    )


def _prediction() -> PredictionArrays:
    targets = _targets()
    return PredictionArrays(targets.energy + 0.1, targets.forces + 0.2, targets.stress + 0.3)


def test_prediction_store_roundtrips_without_dtype_or_value_change(tmp_path: Path) -> None:
    targets = _targets()
    prediction = _prediction()
    write_target_arrays(tmp_path / "targets.npz", targets)
    write_prediction_arrays(tmp_path / "member.npz", prediction)
    loaded_targets = load_target_arrays(tmp_path / "targets.npz")
    loaded_prediction = load_prediction_arrays(tmp_path / "member.npz")
    for field in ("structure_ids", "num_atoms", "atom_offsets", "energy", "forces", "stress"):
        np.testing.assert_array_equal(getattr(loaded_targets, field), getattr(targets, field))
    for field in ("energy", "forces", "stress"):
        np.testing.assert_array_equal(getattr(loaded_prediction, field), getattr(prediction, field))
    validate_predictions(loaded_prediction, loaded_targets)


def test_prediction_validation_rejects_force_shape_drift() -> None:
    bad = PredictionArrays(np.zeros(2), np.zeros((2, 3)), np.zeros((2, 6)))
    with pytest.raises(HardFailure, match="forces"):
        validate_predictions(bad, _targets())
