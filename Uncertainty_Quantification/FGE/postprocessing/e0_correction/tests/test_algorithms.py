from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.uncertainty import scalar_gmd, unbiased_std
from Uncertainty_Quantification.FGE.postprocessing.e0_correction.algorithms import (
    apply_member_delta_e0,
    correction_diagnostics,
    direct_test_correction,
    fit_member_delta_e0,
    test_space_identifiability as check_test_space_identifiability,
)
from Uncertainty_Quantification.FGE.postprocessing.e0_correction.models import (
    CalibrationFit,
    WarningRecord,
)


ATOMIC_NUMBERS = (1, 8)


def _full_rank_case() -> dict[str, Any]:
    composition_val = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    composition_test = np.asarray(
        [[3.0, 1.0], [1.0, 2.0], [2.0, 2.0]], dtype=np.float64
    )
    reference_val = np.asarray([-10.0, -20.0, -31.0, -41.0], dtype=np.float64)
    deltas = np.asarray(
        [[0.5, -1.0], [-0.25, 0.75], [1.25, 0.25]], dtype=np.float64
    )
    raw_val_members = np.stack(
        [reference_val - composition_val @ delta for delta in deltas]
    )
    raw_test_members = np.asarray(
        [
            [-50.0, -60.0, -70.0],
            [-49.0, -61.0, -68.0],
            [-52.0, -59.0, -71.0],
        ],
        dtype=np.float64,
    )
    return {
        "composition_val": composition_val,
        "composition_test": composition_test,
        "reference_val": reference_val,
        "raw_val_members": raw_val_members,
        "raw_test_members": raw_test_members,
    }


def _fit_kwargs(**updates: Any) -> dict[str, Any]:
    case = _full_rank_case()
    values: dict[str, Any] = {
        "composition_val": case["composition_val"],
        "reference_val": case["reference_val"],
        "raw_val_member": case["raw_val_members"][0],
        "composition_test": case["composition_test"],
        "member_id": "member_01",
        "val_atomic_numbers": ATOMIC_NUMBERS,
        "test_atomic_numbers": ATOMIC_NUMBERS,
    }
    values.update(updates)
    return values


def _assert_warning_code(
    warnings: tuple[WarningRecord, ...], expected_code: str
) -> None:
    assert isinstance(warnings, tuple)
    assert warnings
    assert all(isinstance(warning, WarningRecord) for warning in warnings)
    assert expected_code in {warning.code for warning in warnings}


def _manual_weighted_uq(
    members: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normalized = weights / weights.sum()
    mean = np.sum(normalized[:, None] * members, axis=0)
    variance = np.sum(
        normalized[:, None] * np.square(members - mean), axis=0
    ) / (1.0 - np.square(normalized).sum())
    left, right = np.triu_indices(members.shape[0], k=1)
    pair_weights = normalized[left] * normalized[right]
    differences = np.abs(members[left] - members[right])
    gmd = np.sum(pair_weights[:, None] * differences, axis=0) / pair_weights.sum()
    return np.sqrt(variance), gmd


def test_direct_correction_uses_exact_total_energy_formula_for_each_member() -> None:
    raw_total = np.asarray(
        [[-72.0, -139.0, -44.0], [-70.0, -141.0, -43.0]], dtype=np.float32
    )
    reference_total = np.asarray([-76.0, -143.0, -48.0], dtype=np.float32)
    atomization = np.asarray([-11.0, -23.0, -4.0], dtype=np.float32)
    composition = np.asarray(
        [[2.0, 1.0], [1.0, 2.0], [3.0, 0.0]], dtype=np.float32
    )
    model_e0 = np.asarray(
        [[-13.5, -74.0], [-14.0, -73.5]], dtype=np.float32
    )

    corrected = direct_test_correction(
        raw_total=raw_total,
        reference_total=reference_total,
        atomization_energy=atomization,
        composition=composition,
        model_e0=model_e0,
        composition_atomic_numbers=ATOMIC_NUMBERS,
        e0_atomic_numbers=ATOMIC_NUMBERS,
    )

    mad_baseline = reference_total.astype(np.float64) - atomization
    model_baseline = composition.astype(np.float64) @ model_e0.astype(np.float64).T
    expected = raw_total.astype(np.float64) - model_baseline.T + mad_baseline
    np.testing.assert_allclose(corrected, expected, rtol=0.0, atol=1.0e-12)
    assert corrected.shape == raw_total.shape
    assert corrected.dtype == np.float64


def test_direct_correction_never_treats_atomization_as_total_or_per_atom() -> None:
    raw = np.asarray([[-60.0, -110.0]])
    reference = np.asarray([-70.0, -130.0])
    atomization = np.asarray([-5.0, -10.0])
    composition = np.asarray([[2.0, 1.0], [1.0, 3.0]])
    model_e0 = np.asarray([[-10.0, -20.0]])

    corrected = direct_test_correction(
        raw_total=raw,
        reference_total=reference,
        atomization_energy=atomization,
        composition=composition,
        model_e0=model_e0,
        composition_atomic_numbers=ATOMIC_NUMBERS,
        e0_atomic_numbers=ATOMIC_NUMBERS,
    )

    model_baseline = composition @ model_e0[0]
    expected_total = raw[0] - model_baseline + reference - atomization
    atomization_as_total = raw[0] - model_baseline + atomization
    per_atom_mistake = expected_total / composition.sum(axis=1)
    np.testing.assert_allclose(corrected[0], expected_total, rtol=0.0, atol=0.0)
    assert not np.allclose(corrected[0], atomization_as_total)
    assert not np.allclose(corrected[0], per_atom_mistake)


@pytest.mark.parametrize(
    "updates",
    [
        {"raw_total": [[np.nan, 0.0], [1.0, 2.0]]},
        {"reference_total": [0.0, np.inf]},
        {"atomization_energy": [0.0, -np.inf]},
        {"composition": [[1.0, -1.0], [0.0, 1.0]]},
        {"model_e0": [[0.0, np.nan], [1.0, 2.0]]},
        {
            "raw_total": np.empty((2, 0)),
            "reference_total": [],
            "atomization_energy": [],
            "composition": np.empty((0, 2)),
        },
        {"raw_total": [[0.0], [1.0]], "reference_total": [0.0, 1.0]},
        {"composition": [[1.0], [2.0]]},
    ],
)
def test_direct_rejects_nonfinite_negative_empty_or_shape_errors(
    updates: dict[str, Any],
) -> None:
    values: dict[str, Any] = {
        "raw_total": [[0.0, 1.0], [2.0, 3.0]],
        "reference_total": [0.5, 1.5],
        "atomization_energy": [-1.0, -2.0],
        "composition": [[1.0, 0.0], [0.0, 1.0]],
        "model_e0": [[-1.0, -2.0], [-1.5, -2.5]],
        "composition_atomic_numbers": ATOMIC_NUMBERS,
        "e0_atomic_numbers": ATOMIC_NUMBERS,
    }
    values.update(updates)
    with pytest.raises(HardFailure):
        direct_test_correction(**values)


@pytest.mark.parametrize(
    ("model_e0", "e0_atomic_numbers"),
    [
        ([[-1.0, -2.0], [-1.5, -2.5]], (8, 1)),
        ([[-1.0], [-1.5]], (1,)),
    ],
)
def test_direct_rejects_reordered_or_missing_member_e0_columns(
    model_e0: list[list[float]], e0_atomic_numbers: tuple[int, ...]
) -> None:
    with pytest.raises(HardFailure):
        direct_test_correction(
            raw_total=[[0.0, 1.0], [2.0, 3.0]],
            reference_total=[0.5, 1.5],
            atomization_energy=[-1.0, -2.0],
            composition=[[1.0, 0.0], [0.0, 1.0]],
            model_e0=model_e0,
            composition_atomic_numbers=ATOMIC_NUMBERS,
            e0_atomic_numbers=e0_atomic_numbers,
        )


def test_fit_and_apply_match_numpy_lstsq_independently_for_each_member() -> None:
    case = _full_rank_case()
    corrected_members: list[np.ndarray] = []

    for member_index in range(case["raw_val_members"].shape[0]):
        raw_val = case["raw_val_members"][member_index]
        expected_delta = np.linalg.lstsq(
            case["composition_val"], case["reference_val"] - raw_val, rcond=None
        )[0]
        fit, diagnostics, warnings = fit_member_delta_e0(
            **_fit_kwargs(
                raw_val_member=raw_val,
                member_id=f"member_{member_index + 1:02d}",
            )
        )

        assert isinstance(fit, CalibrationFit)
        assert fit.member_id == f"member_{member_index + 1:02d}"
        assert fit.atomic_numbers == ATOMIC_NUMBERS
        np.testing.assert_allclose(
            fit.delta_e0, expected_delta, rtol=0.0, atol=1.0e-12
        )
        assert diagnostics["rcond"] is None
        assert warnings == ()

        corrected = apply_member_delta_e0(
            raw_test_member=case["raw_test_members"][member_index],
            composition_test=case["composition_test"],
            delta_e0=fit.delta_e0,
            composition_atomic_numbers=ATOMIC_NUMBERS,
            delta_atomic_numbers=fit.atomic_numbers,
        )
        expected_test = (
            case["raw_test_members"][member_index]
            + case["composition_test"] @ expected_delta
        )
        np.testing.assert_allclose(
            corrected, expected_test, rtol=0.0, atol=1.0e-12
        )
        assert corrected.dtype == np.float64
        corrected_members.append(corrected)

    corrected_ensemble = np.stack(corrected_members)
    members = torch.from_numpy(corrected_ensemble)
    weights_array = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    weights = torch.from_numpy(weights_array)
    equal_std, equal_gmd = _manual_weighted_uq(
        corrected_ensemble, np.ones(corrected_ensemble.shape[0])
    )
    weighted_std, weighted_gmd = _manual_weighted_uq(
        corrected_ensemble, weights_array
    )
    np.testing.assert_allclose(unbiased_std(members).numpy(), equal_std)
    np.testing.assert_allclose(scalar_gmd(members).numpy(), equal_gmd)
    np.testing.assert_allclose(unbiased_std(members, weights).numpy(), weighted_std)
    np.testing.assert_allclose(scalar_gmd(members, weights).numpy(), weighted_gmd)


def test_fit_api_cannot_consume_test_reference_or_validation_atomization() -> None:
    signature = inspect.signature(fit_member_delta_e0)
    forbidden = {
        "test_reference",
        "reference_test",
        "energy_reference_test",
        "atomization_energy",
        "val_atomization_energy",
    }
    assert forbidden.isdisjoint(signature.parameters)
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    with pytest.raises(TypeError):
        fit_member_delta_e0(**_fit_kwargs(), test_reference=np.zeros(3))
    with pytest.raises(TypeError):
        fit_member_delta_e0(**_fit_kwargs(), val_atomization_energy=np.zeros(4))


def test_fit_target_is_validation_total_reference_minus_raw_member() -> None:
    case = _full_rank_case()
    expected = np.linalg.lstsq(
        case["composition_val"],
        case["reference_val"] - case["raw_val_members"][0],
        rcond=None,
    )[0]

    fit, _, _ = fit_member_delta_e0(**_fit_kwargs())

    np.testing.assert_allclose(fit.delta_e0, expected, rtol=0.0, atol=1.0e-12)


def test_identifiability_uses_fixed_full_svd_rank_and_impact_tolerances() -> None:
    case = _full_rank_case()
    diagnostics, warnings = check_test_space_identifiability(
        composition_val=case["composition_val"],
        composition_test=case["composition_test"],
        val_atomic_numbers=ATOMIC_NUMBERS,
        test_atomic_numbers=ATOMIC_NUMBERS,
    )

    _, singular_values, vh = np.linalg.svd(
        case["composition_val"], full_matrices=True
    )
    rank_tol = (
        np.finfo(np.float64).eps
        * max(case["composition_val"].shape)
        * max(float(singular_values[0]), 1.0)
    )
    rank = int(np.count_nonzero(singular_values > rank_tol))
    v_null = vh[rank:].T
    expected_impact = (
        float(np.max(np.abs(case["composition_test"] @ v_null)))
        if v_null.shape[1]
        else 0.0
    )
    impact_tol = rank_tol * max(
        1.0, float(np.linalg.norm(case["composition_test"], ord=2))
    )

    assert diagnostics["rank"] == rank == 2
    assert diagnostics["full_column_count"] == 2
    np.testing.assert_allclose(diagnostics["singular_values"], singular_values)
    assert diagnostics["rank_tol"] == pytest.approx(rank_tol)
    assert diagnostics["nullspace_test_impact"] == pytest.approx(expected_impact)
    assert diagnostics["impact_tol"] == pytest.approx(impact_tol)
    assert diagnostics["rcond"] is None
    assert warnings == ()


def test_harmless_rank_deficiency_keeps_minimum_norm_solution_and_warns() -> None:
    composition_val = np.asarray(
        [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]]
    )
    composition_test = np.asarray([[4.0, 4.0, 0.0], [1.0, 1.0, 0.0]])
    reference = np.asarray([2.0, 4.0, 6.0])

    fit, diagnostics, warnings = fit_member_delta_e0(
        composition_val=composition_val,
        reference_val=reference,
        raw_val_member=np.zeros(3),
        composition_test=composition_test,
        member_id="member_01",
        val_atomic_numbers=(1, 8, 14),
        test_atomic_numbers=(1, 8, 14),
    )

    expected = np.linalg.lstsq(composition_val, reference, rcond=None)[0]
    np.testing.assert_allclose(fit.delta_e0, expected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(fit.delta_e0, (1.0, 1.0, 0.0), atol=1.0e-12)
    assert fit.rank == diagnostics["rank"] == 1
    assert np.isfinite(fit.condition_number)
    assert diagnostics["nullspace_test_impact"] <= diagnostics["impact_tol"]
    _assert_warning_code(warnings, "harmless_rank_deficiency")


def test_singular_value_equal_to_rank_tolerance_is_excluded_from_rank() -> None:
    composition_val = np.diag(
        np.asarray([float(2**51), 1.0], dtype=np.float64)
    )
    diagnostics, warnings = check_test_space_identifiability(
        composition_val=composition_val,
        composition_test=[[1.0, 0.0]],
        val_atomic_numbers=ATOMIC_NUMBERS,
        test_atomic_numbers=ATOMIC_NUMBERS,
    )

    assert diagnostics["singular_values"][1] == pytest.approx(1.0)
    assert diagnostics["rank_tol"] == pytest.approx(1.0)
    assert diagnostics["rank"] == 1
    _assert_warning_code(warnings, "harmless_rank_deficiency")


def test_full_svd_detects_harmful_underdetermined_nullspace() -> None:
    with pytest.raises(HardFailure):
        check_test_space_identifiability(
            composition_val=[[1.0, 1.0, 1.0]],
            composition_test=[[1.0, 0.0, 0.0]],
            val_atomic_numbers=(1, 8, 14),
            test_atomic_numbers=(1, 8, 14),
        )


def test_test_element_uncovered_by_validation_is_a_hard_failure() -> None:
    with pytest.raises(HardFailure):
        check_test_space_identifiability(
            composition_val=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            composition_test=[[1.0, 1.0]],
            val_atomic_numbers=ATOMIC_NUMBERS,
            test_atomic_numbers=ATOMIC_NUMBERS,
        )


def test_diagnostics_are_json_friendly_and_match_manual_residuals() -> None:
    composition_val = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    raw_val = np.asarray([1.0, 2.0, 3.0])
    reference = np.asarray([2.0, 4.0, 8.0])
    delta = np.asarray([1.0, 2.0])

    diagnostics, warnings = correction_diagnostics(
        composition_val=composition_val,
        reference_val=reference,
        raw_val_member=raw_val,
        delta_e0=delta,
        composition_test=[[2.0, 1.0], [1.0, 2.0]],
        val_atomic_numbers=ATOMIC_NUMBERS,
        test_atomic_numbers=ATOMIC_NUMBERS,
    )

    residual = reference - (raw_val + composition_val @ delta)
    singular_values = np.linalg.svd(composition_val, full_matrices=True)[1]
    required = {
        "rank",
        "full_column_count",
        "singular_values",
        "condition_number",
        "rank_tol",
        "impact_tol",
        "nullspace_test_impact",
        "residual_rmse",
        "residual_max_abs",
        "rcond",
    }
    assert required <= set(diagnostics)
    assert isinstance(diagnostics["singular_values"], tuple)
    np.testing.assert_allclose(diagnostics["singular_values"], singular_values)
    assert diagnostics["residual_rmse"] == pytest.approx(
        float(np.sqrt(np.mean(np.square(residual))))
    )
    assert diagnostics["residual_max_abs"] == pytest.approx(
        float(np.max(np.abs(residual)))
    )
    assert diagnostics["rcond"] is None
    json.dumps(diagnostics, allow_nan=False)
    assert warnings == ()


def test_large_calibration_residual_is_only_a_warning() -> None:
    fit, diagnostics, warnings = fit_member_delta_e0(
        composition_val=[[1.0], [2.0], [3.0]],
        reference_val=[1.0, 1.0, 10.0],
        raw_val_member=[0.0, 0.0, 0.0],
        composition_test=[[1.0], [4.0]],
        member_id="member_01",
        val_atomic_numbers=(1,),
        test_atomic_numbers=(1,),
        residual_rmse_warning_threshold=0.5,
    )

    assert isinstance(fit, CalibrationFit)
    assert diagnostics["residual_rmse"] > 0.5
    _assert_warning_code(warnings, "large_calibration_residual")


def test_ill_conditioned_but_identifiable_matrix_is_only_a_warning() -> None:
    large = 1_000_000.0
    composition = np.asarray(
        [[large, large - 1.0], [large + 1.0, large]], dtype=np.float64
    )
    diagnostics, warnings = check_test_space_identifiability(
        composition_val=composition,
        composition_test=composition,
        val_atomic_numbers=ATOMIC_NUMBERS,
        test_atomic_numbers=ATOMIC_NUMBERS,
        condition_number_warning_threshold=1.0e8,
    )

    assert diagnostics["rank"] == 2
    assert diagnostics["condition_number"] > 1.0e8
    _assert_warning_code(warnings, "ill_conditioned_composition")


def test_no_strict_rmse_improvement_is_only_a_warning() -> None:
    diagnostics, warnings = correction_diagnostics(
        composition_val=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        reference_val=[2.0, 4.0, 8.0],
        raw_val_member=[1.0, 2.0, 3.0],
        delta_e0=[0.0, 0.0],
        composition_test=[[1.0, 0.0], [0.0, 1.0]],
        val_atomic_numbers=ATOMIC_NUMBERS,
        test_atomic_numbers=ATOMIC_NUMBERS,
    )

    assert diagnostics["residual_rmse"] > 0.0
    _assert_warning_code(warnings, "weak_rmse_improvement")


@pytest.mark.parametrize(
    "updates",
    [
        {"composition_val": [[1.0, 0.0], [0.0, np.nan]]},
        {"reference_val": [-10.0, -20.0, -31.0, np.inf]},
        {"raw_val_member": [-10.0, -20.0, -31.0, np.nan]},
        {"composition_test": [[1.0, -1.0]]},
        {
            "composition_val": np.empty((0, 2)),
            "reference_val": [],
            "raw_val_member": [],
        },
        {"reference_val": [-10.0, -20.0]},
        {"raw_val_member": [-10.0, -20.0]},
        {"composition_test": [[1.0, 0.0, 0.0]]},
        {"test_atomic_numbers": (8, 1)},
        {"member_id": ""},
    ],
)
def test_fit_rejects_nonfinite_negative_empty_shape_or_column_order_errors(
    updates: dict[str, Any],
) -> None:
    with pytest.raises(HardFailure):
        fit_member_delta_e0(**_fit_kwargs(**updates))


@pytest.mark.parametrize(
    "updates",
    [
        {"raw_test_member": [1.0, np.nan, 3.0]},
        {"composition_test": [[1.0, 0.0], [-1.0, 1.0], [0.0, 1.0]]},
        {"delta_e0": [0.5, np.inf]},
        {"raw_test_member": [], "composition_test": np.empty((0, 2))},
        {"raw_test_member": [1.0, 2.0]},
        {"delta_e0": [0.5]},
        {"delta_atomic_numbers": (8, 1)},
    ],
)
def test_apply_rejects_nonfinite_negative_empty_shape_or_column_order_errors(
    updates: dict[str, Any],
) -> None:
    values: dict[str, Any] = {
        "raw_test_member": [1.0, 2.0, 3.0],
        "composition_test": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        "delta_e0": [0.5, -1.0],
        "composition_atomic_numbers": ATOMIC_NUMBERS,
        "delta_atomic_numbers": ATOMIC_NUMBERS,
    }
    values.update(updates)
    with pytest.raises(HardFailure):
        apply_member_delta_e0(**values)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            check_test_space_identifiability,
            {
                "composition_val": [[1.0, 0.0], [0.0, 1.0]],
                "composition_test": [[1.0, 0.0]],
                "val_atomic_numbers": ATOMIC_NUMBERS,
                "test_atomic_numbers": ATOMIC_NUMBERS,
                "condition_number_warning_threshold": np.nan,
            },
        ),
        (
            correction_diagnostics,
            {
                "composition_val": [[1.0, 0.0], [0.0, 1.0]],
                "reference_val": [1.0, 2.0],
                "raw_val_member": [0.0, 0.0],
                "delta_e0": [1.0, 2.0],
                "composition_test": [[1.0, 0.0]],
                "val_atomic_numbers": ATOMIC_NUMBERS,
                "test_atomic_numbers": ATOMIC_NUMBERS,
                "residual_rmse_warning_threshold": np.inf,
            },
        ),
        (
            check_test_space_identifiability,
            {
                "composition_val": [[1.0, 0.0], [0.0, 1.0]],
                "composition_test": [[1.0, 0.0]],
                "val_atomic_numbers": ATOMIC_NUMBERS,
                "test_atomic_numbers": ATOMIC_NUMBERS,
                "condition_number_warning_threshold": 10**10000,
            },
        ),
    ],
)
def test_warning_thresholds_must_be_finite_when_provided(
    function: Callable[..., Any], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(HardFailure):
        function(**kwargs)


@pytest.mark.parametrize(
    "updates",
    [
        {"composition_val": [[1.0, 0.0], [0.0, np.nan]]},
        {"composition_test": [[-1.0, 0.0]]},
        {"composition_val": np.empty((0, 2))},
        {"composition_val": [1.0, 0.0]},
        {"composition_test": [[1.0, 0.0, 0.0]]},
        {"test_atomic_numbers": (8, 1)},
    ],
)
def test_identifiability_helper_rejects_its_own_invalid_inputs(
    updates: dict[str, Any],
) -> None:
    values: dict[str, Any] = {
        "composition_val": [[1.0, 0.0], [0.0, 1.0]],
        "composition_test": [[1.0, 0.0]],
        "val_atomic_numbers": ATOMIC_NUMBERS,
        "test_atomic_numbers": ATOMIC_NUMBERS,
    }
    values.update(updates)
    with pytest.raises(HardFailure):
        check_test_space_identifiability(**values)


@pytest.mark.parametrize(
    "updates",
    [
        {"composition_val": [[1.0, 0.0], [0.0, np.nan]]},
        {"composition_test": [[-1.0, 0.0]]},
        {
            "composition_val": np.empty((0, 2)),
            "reference_val": [],
            "raw_val_member": [],
        },
        {"raw_val_member": [0.0]},
        {"composition_test": [[1.0, 0.0, 0.0]]},
        {"test_atomic_numbers": (8, 1)},
    ],
)
def test_correction_diagnostics_rejects_its_own_invalid_inputs(
    updates: dict[str, Any],
) -> None:
    values: dict[str, Any] = {
        "composition_val": [[1.0, 0.0], [0.0, 1.0]],
        "reference_val": [1.0, 2.0],
        "raw_val_member": [0.0, 0.0],
        "delta_e0": [1.0, 2.0],
        "composition_test": [[1.0, 0.0]],
        "val_atomic_numbers": ATOMIC_NUMBERS,
        "test_atomic_numbers": ATOMIC_NUMBERS,
    }
    values.update(updates)
    with pytest.raises(HardFailure):
        correction_diagnostics(**values)


def test_inputs_are_not_mutated_and_nonenergy_fields_are_outside_the_api() -> None:
    case = _full_rank_case()
    raw = case["raw_test_members"].copy()
    composition = case["composition_test"].copy()
    model_e0 = np.asarray([[-1.0, -2.0], [-1.5, -2.5], [-0.5, -3.0]])
    snapshots = (raw.copy(), composition.copy(), model_e0.copy())

    direct_test_correction(
        raw_total=raw,
        reference_total=[-1.0, -2.0, -3.0],
        atomization_energy=[-0.1, -0.2, -0.3],
        composition=composition,
        model_e0=model_e0,
        composition_atomic_numbers=ATOMIC_NUMBERS,
        e0_atomic_numbers=ATOMIC_NUMBERS,
    )
    fit, _, _ = fit_member_delta_e0(**_fit_kwargs())
    apply_member_delta_e0(
        raw_test_member=raw[0],
        composition_test=composition,
        delta_e0=fit.delta_e0,
        composition_atomic_numbers=ATOMIC_NUMBERS,
        delta_atomic_numbers=fit.atomic_numbers,
    )

    for actual, expected in zip((raw, composition, model_e0), snapshots):
        np.testing.assert_array_equal(actual, expected)
    nonenergy_fields = {
        "forces_members",
        "forces_reference",
        "atom_to_structure",
        "structure_ptr",
        "n_atoms",
    }
    for function in (
        direct_test_correction,
        fit_member_delta_e0,
        apply_member_delta_e0,
    ):
        signature = inspect.signature(function)
        assert nonenergy_fields.isdisjoint(signature.parameters)
        assert all(
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )


def test_common_structure_shift_preserves_equal_weight_energy_std_and_gmd() -> None:
    raw = np.asarray([[1.0, 5.0], [2.0, 8.0], [4.0, 9.0]])
    composition = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    shared_e0 = np.asarray([[-2.0, -3.0]] * 3)
    corrected = direct_test_correction(
        raw_total=raw,
        reference_total=[-10.0, -20.0],
        atomization_energy=[-1.0, -2.0],
        composition=composition,
        model_e0=shared_e0,
        composition_atomic_numbers=ATOMIC_NUMBERS,
        e0_atomic_numbers=ATOMIC_NUMBERS,
    )

    raw_tensor = torch.from_numpy(raw)
    corrected_tensor = torch.from_numpy(corrected)
    assert torch.allclose(unbiased_std(corrected_tensor), unbiased_std(raw_tensor))
    assert torch.allclose(scalar_gmd(corrected_tensor), scalar_gmd(raw_tensor))


def test_member_e0_differences_recompute_std_and_gmd_from_corrected_members() -> None:
    raw = np.asarray([[1.0, 5.0], [2.0, 8.0], [4.0, 9.0]])
    composition = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    model_e0 = np.asarray([[-2.0, -3.0], [-1.0, -5.0], [-4.0, -2.0]])
    corrected = direct_test_correction(
        raw_total=raw,
        reference_total=[-10.0, -20.0],
        atomization_energy=[-1.0, -2.0],
        composition=composition,
        model_e0=model_e0,
        composition_atomic_numbers=ATOMIC_NUMBERS,
        e0_atomic_numbers=ATOMIC_NUMBERS,
    )

    members = torch.from_numpy(corrected)
    manual_std = np.std(corrected, axis=0, ddof=1)
    manual_gmd = np.mean(
        np.stack(
            [
                np.abs(corrected[0] - corrected[1]),
                np.abs(corrected[0] - corrected[2]),
                np.abs(corrected[1] - corrected[2]),
            ]
        ),
        axis=0,
    )
    np.testing.assert_allclose(unbiased_std(members).numpy(), manual_std)
    np.testing.assert_allclose(scalar_gmd(members).numpy(), manual_gmd)

    weights_array = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    weights = torch.from_numpy(weights_array)
    weighted_std, weighted_gmd = _manual_weighted_uq(corrected, weights_array)
    np.testing.assert_allclose(unbiased_std(members, weights).numpy(), weighted_std)
    np.testing.assert_allclose(scalar_gmd(members, weights).numpy(), weighted_gmd)
    assert not np.allclose(manual_std, np.std(raw, axis=0, ddof=1))


def test_equal_k_minus_one_and_validation_weighted_std_reuse_fge_rules() -> None:
    members = torch.tensor(
        [[1.0, 2.0], [4.0, 8.0], [10.0, 5.0]], dtype=torch.float64
    )
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    normalized = weights.numpy() / weights.numpy().sum()
    mean = np.sum(normalized[:, None] * members.numpy(), axis=0)
    variance = np.sum(
        normalized[:, None] * np.square(members.numpy() - mean), axis=0
    ) / (1.0 - np.square(normalized).sum())

    np.testing.assert_allclose(
        unbiased_std(members).numpy(), np.std(members.numpy(), axis=0, ddof=1)
    )
    np.testing.assert_allclose(
        unbiased_std(members, weights).numpy(), np.sqrt(variance)
    )
