from __future__ import annotations

import numpy as np
import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.energy_reference import (
    apply_energy_correction,
    count_matrix,
    fit_model_aware_delta,
    recover_mad_e0,
    validate_common_shift,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure


def test_recover_and_apply_direct_mad_e0() -> None:
    numbers = [np.asarray([1, 6]), np.asarray([6, 8]), np.asarray([1, 8])]
    matrix = count_matrix(numbers, [1, 6, 8])
    e0 = np.asarray([-1.0, -2.0, -3.0])
    total = matrix @ e0 + np.asarray([0.2, -0.4, 0.7])
    atomization = np.asarray([0.2, -0.4, 0.7])
    fit = recover_mad_e0(matrix, total, atomization)
    np.testing.assert_allclose(fit.values, e0)
    raw = np.stack([matrix @ np.asarray([-0.5, -1.5, -2.5]), matrix @ np.asarray([-0.4, -1.4, -2.4])])
    corrected = apply_energy_correction(raw, matrix, fit.values - np.asarray([-0.5, -1.5, -2.5]))
    np.testing.assert_allclose(corrected[0], matrix @ e0)


def test_model_aware_uses_ensemble_mean_and_preserves_uq() -> None:
    matrix = np.asarray([[1, 1], [2, 0], [0, 2]], dtype=float)
    reference = np.asarray([-3.0, -2.0, -4.0])
    mean_prediction = np.asarray([-2.0, -1.0, -3.0])
    fit = fit_model_aware_delta(matrix, reference, mean_prediction)
    np.testing.assert_allclose(fit.values, [-0.5, -0.5])
    raw = np.stack([mean_prediction - 0.2, mean_prediction + 0.2])
    corrected = apply_energy_correction(raw, matrix, fit.values)
    validate_common_shift(raw, corrected)
    np.testing.assert_allclose(np.std(raw, axis=0, ddof=1), np.std(corrected, axis=0, ddof=1))


def test_rank_deficiency_is_hard_failure() -> None:
    with pytest.raises(HardFailure, match="rank deficient"):
        recover_mad_e0(np.asarray([[1.0, 0.0], [2.0, 0.0]]), np.asarray([1.0, 2.0]), np.zeros(2))


def test_unknown_element_is_hard_failure() -> None:
    with pytest.raises(HardFailure, match="unsupported"):
        count_matrix([np.asarray([1, 8])], [1, 6])
