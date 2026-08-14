"""Non-plotting metrics for native prediction and uncertainty arrays."""

from __future__ import annotations

import numpy as np

from .prediction import PredictionArrays, TargetArrays, validate_predictions
from .uncertainty import UncertaintyArrays


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    residual = np.asarray(prediction, dtype=float).reshape(-1) - np.asarray(target, dtype=float).reshape(-1)
    return {
        "point_count": int(residual.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
    }


def analyze_predictions(prediction: PredictionArrays, targets: TargetArrays, uncertainty: UncertaintyArrays) -> dict[str, object]:
    validate_predictions(prediction, targets)
    if uncertainty.force_std.shape != targets.forces.shape or uncertainty.stress_std.shape != targets.stress.shape:
        raise ValueError("uncertainty shape mismatch")
    return {
        "metrics": {
            "energy": _metrics(prediction.energy, targets.energy),
            "force": _metrics(prediction.forces, targets.forces),
            "stress": _metrics(prediction.stress, targets.stress),
        },
        "uncertainty_semantics": {
            "energy": "structure_scalar_N",
            "force": "cartesian_component_3N",
            "stress": "matrix_component_9N",
            "std_ddof": 1,
            "gmd_pairs": "distinct_unordered",
        },
    }
