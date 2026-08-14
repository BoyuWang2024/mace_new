from __future__ import annotations

import numpy as np

from Uncertainty_Quantification.BootStrapping.bootstrap.aggregation import ensemble_mean
from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import PredictionArrays
from Uncertainty_Quantification.BootStrapping.bootstrap.uncertainty import compute_uncertainty


def _member(offset: float) -> PredictionArrays:
    return PredictionArrays(
        energy=np.asarray([1.0, 2.0]) + offset,
        forces=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) + offset,
        stress=np.arange(12, dtype=float).reshape(2, 6) + offset,
    )


def test_ensemble_mean_and_sample_std_are_elementwise() -> None:
    members = [_member(0.0), _member(2.0), _member(4.0)]
    mean = ensemble_mean(members)
    uq = compute_uncertainty(members, ddof=1)
    np.testing.assert_array_equal(mean.forces, _member(2.0).forces)
    np.testing.assert_allclose(uq.force_std, np.full((2, 3), 2.0))
    np.testing.assert_allclose(uq.force_gmd, np.full((2, 3), 8.0 / 3.0))
    assert uq.force_std.shape == (2, 3)
    assert not hasattr(uq, "force_rms_std")


def test_gmd_uses_only_distinct_unordered_member_pairs() -> None:
    members = [_member(0.0), _member(1.0)]
    uq = compute_uncertainty(members, ddof=1)
    np.testing.assert_allclose(uq.energy_gmd, np.ones(2))
    np.testing.assert_allclose(uq.force_gmd, np.ones((2, 3)))
