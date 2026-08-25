from __future__ import annotations

import numpy as np
import pytest


def test_direct_test_atomic_baseline_uses_each_structure_mad_baseline() -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference import (
        apply_direct_test_atomic_baseline,
    )

    corrected = apply_direct_test_atomic_baseline(
        raw_total=np.array([10.0, 20.0]),
        reference_total=np.array([8.0, 17.0]),
        atomization_total=np.array([3.0, 4.0]),
        composition=np.array([[1.0, 1.0], [2.0, 1.0]]),
        model_e0=np.array([1.0, 2.0]),
    )

    np.testing.assert_allclose(corrected, [12.0, 29.0])


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"raw_total": np.ones((2, 1))}, "raw_total must be one-dimensional"),
        (
            {"reference_total": np.array([8.0])},
            "structure vectors and composition must have the same number of rows",
        ),
        ({"composition": np.ones(2)}, "composition must be two-dimensional"),
        (
            {"model_e0": np.array([1.0])},
            "composition columns must match model_e0",
        ),
        (
            {"atomization_total": np.array([3.0, np.nan])},
            "atomization_total must contain only finite values",
        ),
    ],
)
def test_direct_test_atomic_baseline_rejects_invalid_inputs(
    overrides: dict[str, np.ndarray], match: str
) -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference import (
        apply_direct_test_atomic_baseline,
    )

    arguments = {
        "raw_total": np.array([10.0, 20.0]),
        "reference_total": np.array([8.0, 17.0]),
        "atomization_total": np.array([3.0, 4.0]),
        "composition": np.array([[1.0, 1.0], [2.0, 1.0]]),
        "model_e0": np.array([1.0, 2.0]),
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=match):
        apply_direct_test_atomic_baseline(**arguments)


def test_model_aware_fit_uses_positive_total_energy_correction() -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference import (
        fit_model_aware_reestimation,
    )

    composition = np.array([[2.0, 0.0], [0.0, 1.0], [3.0, 1.0]])
    expected_delta = np.array([1.5, -2.0])
    raw_total = np.array([10.0, 20.0, 50.0])
    reference_total = raw_total + composition @ expected_delta

    fit = fit_model_aware_reestimation(
        composition=composition,
        reference_total=reference_total,
        raw_total=raw_total,
        model_e0=np.array([-5.0, -8.0]),
    )

    np.testing.assert_allclose(fit.delta_e0, expected_delta)
    np.testing.assert_allclose(fit.new_e0, [-3.5, -10.0])
    np.testing.assert_allclose(fit.corrected_validation_total, reference_total)
    assert fit.rank == 2
    assert fit.residual_norm == pytest.approx(0.0, abs=1.0e-12)


def test_model_aware_application_adds_composition_delta_to_raw_total() -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference import (
        apply_model_aware_correction,
    )

    corrected = apply_model_aware_correction(
        raw_total=np.array([4.0, 7.0]),
        composition=np.array([[1.0, 2.0], [3.0, 1.0]]),
        delta_e0=np.array([0.5, -1.0]),
    )

    np.testing.assert_allclose(corrected, [2.5, 7.5])


def test_model_aware_rank_deficient_fit_is_svd_minimum_norm() -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference import (
        fit_model_aware_reestimation,
    )

    composition = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    raw_total = np.array([10.0, 20.0, 30.0])
    error = np.array([2.0, 4.0, 6.0])

    fit = fit_model_aware_reestimation(
        composition=composition,
        reference_total=raw_total + error,
        raw_total=raw_total,
        model_e0=np.array([-1.0, -1.0]),
    )

    np.testing.assert_allclose(fit.delta_e0, np.linalg.pinv(composition) @ error)
    np.testing.assert_allclose(fit.delta_e0, [1.0, 1.0])
    assert fit.rank == 1
    assert fit.singular_values.shape == (2,)


def test_model_aware_fit_rejects_ensemble_predictions() -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference import (
        fit_model_aware_reestimation,
    )

    with pytest.raises(ValueError, match="raw_total must be one-dimensional"):
        fit_model_aware_reestimation(
            composition=np.ones((2, 1)),
            reference_total=np.ones(2),
            raw_total=np.ones((2, 2)),
            model_e0=np.ones(1),
        )


def test_energy_alpha_uses_corrected_per_atom_residuals() -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference import (
        calibrate_energy_alpha,
    )

    alpha = calibrate_energy_alpha(
        reference_total=np.array([12.0, 16.0]),
        prediction_total=np.array([10.0, 20.0]),
        num_atoms=np.array([2.0, 4.0]),
        q=np.array([1.0, 4.0]),
    )

    assert alpha == pytest.approx(np.sqrt(0.625))
