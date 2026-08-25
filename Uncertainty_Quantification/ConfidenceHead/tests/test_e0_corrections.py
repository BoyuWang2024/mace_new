from __future__ import annotations

import numpy as np
import pytest

from confidence_head.e0_corrections import (
    E0CorrectionError,
    apply_e0_reestimate,
    apply_e0_replace,
    composition_matrix,
    energy_errors_per_atom,
    fit_e0_reestimate,
)


def test_composition_matrix_uses_checkpoint_element_order() -> None:
    matrix = composition_matrix(
        [np.asarray([8, 1, 8]), np.asarray([6, 1])],
        (1, 6, 8),
    )

    assert matrix.dtype == np.float64
    np.testing.assert_array_equal(matrix, [[1.0, 0.0, 2.0], [1.0, 1.0, 0.0]])


def test_composition_matrix_rejects_unknown_atomic_number() -> None:
    with pytest.raises(E0CorrectionError, match="unsupported atomic numbers: \\[8\\]"):
        composition_matrix([np.asarray([1, 8])], (1, 6))


def test_direct_replace_uses_each_structure_mad_baseline() -> None:
    counts = composition_matrix([[1, 8], [1, 1, 8]], (1, 8))

    result = apply_e0_replace(
        raw_total=[10.0, 20.0],
        reference_total=[8.0, 17.0],
        atomization_total=[3.0, 4.0],
        composition=counts,
        model_e0=[1.0, 2.0],
    )

    np.testing.assert_allclose(result.model_baseline, [3.0, 4.0])
    np.testing.assert_allclose(result.mad_baseline, [5.0, 13.0])
    np.testing.assert_allclose(result.correction, [2.0, 9.0])
    np.testing.assert_allclose(result.corrected_total, [12.0, 29.0])


def test_direct_replace_rejects_nonfinite_atomization_energy() -> None:
    with pytest.raises(E0CorrectionError, match="atomization_total must be finite"):
        apply_e0_replace(
            raw_total=[1.0],
            reference_total=[2.0],
            atomization_total=[np.nan],
            composition=[[1.0]],
            model_e0=[-1.0],
        )


def test_reestimate_recovers_positive_float64_delta() -> None:
    counts = np.asarray([[2.0, 0.0], [0.0, 1.0], [3.0, 1.0]])
    expected_delta = np.asarray([1.5, -2.0])
    raw = np.asarray([10.0, 20.0, 50.0])
    reference = raw + counts @ expected_delta

    fit = fit_e0_reestimate(
        composition=counts,
        reference_total=reference,
        raw_total=raw,
    )

    assert fit.delta_e0.dtype == np.float64
    np.testing.assert_allclose(fit.delta_e0, expected_delta)
    np.testing.assert_allclose(fit.corrected_validation_total, reference)
    assert fit.rank == counts.shape[1]
    assert fit.condition_number > 0.0
    assert fit.residual_rmse == pytest.approx(0.0, abs=1.0e-12)


def test_reestimate_rejects_rank_deficient_validation_composition() -> None:
    with pytest.raises(E0CorrectionError, match="rank deficient: 1/2"):
        fit_e0_reestimate(
            composition=[[1.0, 0.0], [2.0, 0.0]],
            reference_total=[1.0, 2.0],
            raw_total=[0.0, 0.0],
        )


def test_apply_reestimate_adds_composition_delta() -> None:
    corrected = apply_e0_reestimate(
        raw_total=[4.0, 7.0],
        composition=[[1.0, 2.0], [3.0, 1.0]],
        delta_e0=[0.5, -1.0],
    )

    np.testing.assert_allclose(corrected.correction, [-1.5, 0.5])
    np.testing.assert_allclose(corrected.corrected_total, [2.5, 7.5])


def test_energy_errors_are_absolute_per_atom() -> None:
    errors = energy_errors_per_atom(
        reference_total=[12.0, 16.0],
        prediction_total=[10.0, 20.0],
        num_atoms=[2, 4],
    )

    np.testing.assert_allclose(errors, [1.0, 1.0])


def test_energy_errors_reject_nonpositive_atom_count() -> None:
    with pytest.raises(E0CorrectionError, match="num_atoms must be positive"):
        energy_errors_per_atom(
            reference_total=[1.0], prediction_total=[1.0], num_atoms=[0]
        )
