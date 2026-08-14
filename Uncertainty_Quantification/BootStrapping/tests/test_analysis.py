from __future__ import annotations

import numpy as np

from Uncertainty_Quantification.BootStrapping.bootstrap.analysis import analyze_predictions
from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import PredictionArrays, TargetArrays
from Uncertainty_Quantification.BootStrapping.bootstrap.uncertainty import UncertaintyArrays


def test_analysis_flattens_force_components_as_three_n_points() -> None:
    targets = TargetArrays(
        np.asarray(["a"]), np.asarray([2]), np.asarray([0, 2]),
        np.asarray([1.0]), np.zeros((2, 3)), np.zeros((1, 3, 3)),
    )
    prediction = PredictionArrays(np.asarray([2.0]), np.ones((2, 3)), np.ones((1, 3, 3)))
    uq = UncertaintyArrays(
        energy_std=np.asarray([0.5]), force_std=np.arange(1, 7, dtype=float).reshape(2, 3), stress_std=np.ones((1, 3, 3)),
        energy_gmd=np.asarray([0.4]), force_gmd=np.ones((2, 3)), stress_gmd=np.ones((1, 3, 3)),
    )
    result = analyze_predictions(prediction, targets, uq)
    assert result["metrics"]["force"]["point_count"] == 6
    assert result["metrics"]["stress"]["point_count"] == 9
    assert result["metrics"]["force"]["rmse"] == 1.0
    assert result["uncertainty_semantics"]["force"] == "cartesian_component_3N"
    assert result["uncertainty_semantics"]["stress"] == "matrix_component_9N"
