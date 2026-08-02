"""Hand-derived contracts for continuous errors and immutable bin definitions."""

from __future__ import annotations

import inspect
import math

import pytest
import torch
from confidence_head.binning import (
    BINNING_FORMULA_VERSION,
    BINNING_SCHEMA_VERSION,
    BinningArtifact,
    fit_fixed_linear,
    fit_train_quantile_log,
    labels_from_thresholds,
    representatives_from_thresholds,
)
from confidence_head.identity import run_id, stable_id
from confidence_head.labels import energy_errors, force_errors


def test_force_error_modes() -> None:
    prediction = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    reference = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)

    assert torch.equal(
        force_errors(prediction, reference, "component"),
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
    )
    assert torch.equal(
        force_errors(prediction, reference, "atom_mean"),
        torch.tensor([2.0], dtype=torch.float64),
    )


def test_energy_error_is_absolute_per_atom_error() -> None:
    prediction = torch.tensor([5.0, -1.0], dtype=torch.float32)
    reference = torch.tensor([1.0, 3.0], dtype=torch.float64)
    num_atoms = torch.tensor([2, 4], dtype=torch.long)

    assert torch.equal(
        energy_errors(prediction, reference, num_atoms),
        torch.tensor([2.0, 1.0], dtype=torch.float64),
    )


def test_errors_are_detached_cpu_float64() -> None:
    prediction = torch.tensor(
        [[1.0, 2.0, 3.0]], dtype=torch.float32, requires_grad=True
    )
    reference = torch.zeros(1, 3, dtype=torch.float32)

    errors = force_errors(prediction, reference, "component")

    assert errors.device.type == "cpu"
    assert errors.dtype is torch.float64
    assert errors.requires_grad is False


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: force_errors(
                torch.tensor([[float("nan"), 0.0, 0.0]]),
                torch.zeros(1, 3),
                "component",
            ),
            "finite",
        ),
        (
            lambda: force_errors(torch.zeros(1, 2), torch.zeros(1, 2), "component"),
            "shape",
        ),
        (
            lambda: energy_errors(torch.zeros(1), torch.zeros(1), torch.tensor([0])),
            "positive",
        ),
        (
            lambda: energy_errors(torch.zeros(1), torch.zeros(1), torch.tensor([True])),
            "integers",
        ),
    ],
)
def test_error_formulas_reject_invalid_inputs(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_fixed_linear_threshold_equality_goes_right_and_overflow_is_counted() -> None:
    values = torch.tensor([0.0, 0.1, 0.2, 0.31], dtype=torch.float64)

    branch = fit_fixed_linear(values, num_bins=3, max_error=0.3)

    assert branch.algorithm == "fixed_linear_v1"
    assert branch.thresholds.tolist() == pytest.approx([0.1, 0.2])
    assert labels_from_thresholds(values, branch.thresholds).tolist() == [
        0,
        1,
        2,
        2,
    ]
    assert branch.representatives.tolist() == pytest.approx([0.05, 0.15, 0.25])
    assert branch.counts.tolist() == [1, 1, 2]
    assert branch.representative_sources == (
        "analytic_center",
        "analytic_center",
        "analytic_center",
    )
    assert branch.overflow_count == 1


def test_train_quantile_log_has_hand_checked_anchors_and_medians() -> None:
    values = torch.tensor(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
        dtype=torch.float64,
    )

    branch = fit_train_quantile_log(values, num_bins=4)

    lower = 1.25
    upper = 97.15
    middle = math.sqrt(lower * upper)
    assert branch.algorithm == "train_quantile_log_v1"
    assert branch.thresholds.tolist() == pytest.approx([lower, middle, upper])
    assert branch.representatives.tolist() == pytest.approx(
        [0.5, 3.5, math.sqrt(middle * upper), 100.0]
    )
    assert branch.counts.tolist() == [2, 4, 0, 1]
    assert branch.representative_sources == (
        "median",
        "median",
        "analytic_fallback",
        "median",
    )
    assert branch.overflow_count == 1


def test_empty_bin_fallbacks_are_analytic_and_deterministic() -> None:
    values = torch.tensor([2.0], dtype=torch.float64)
    thresholds = torch.tensor([1.0, 4.0, 16.0], dtype=torch.float64)

    representatives = representatives_from_thresholds(values, thresholds)

    assert representatives.tolist() == pytest.approx([0.5, 2.0, 8.0, 32.0])


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([-0.1, 1.0, 2.0], "non-negative"),
        ([0.0, float("inf"), 2.0], "finite"),
        ([0.0, 0.0, 0.0], "positive"),
        ([1.0, 1.0, 1.0], "lower.*upper"),
    ],
)
def test_train_quantile_log_rejects_invalid_or_degenerate_values(
    values: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_train_quantile_log(torch.tensor(values), num_bins=4)


@pytest.mark.parametrize(
    "values",
    [torch.tensor([-0.1, 0.1]), torch.tensor([0.1, float("nan")])],
)
def test_fixed_linear_rejects_negative_or_nonfinite_values(
    values: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        fit_fixed_linear(values, num_bins=3, max_error=1.0)


def test_labels_reject_invalid_threshold_contracts() -> None:
    with pytest.raises(ValueError, match="increasing"):
        labels_from_thresholds(torch.tensor([0.1]), torch.tensor([0.2, 0.2]))
    with pytest.raises(ValueError, match="non-negative"):
        labels_from_thresholds(torch.tensor([-0.1]), torch.tensor([0.2, 0.4]))


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value


def _artifact_payload(branch, *, experiment_id: str = "experiment-a") -> dict:
    return {
        "schema_version": BINNING_SCHEMA_VERSION,
        "formula_version": BINNING_FORMULA_VERSION,
        "cache_id": "cache-a",
        "experiment_id": experiment_id,
        "binning_id": "pending",
        "run_id": "pending",
        "algorithm": branch.algorithm,
        "branches": {"force": branch.to_payload()},
    }


def _synchronize_semantic_ids(payload: dict, target_mode: str) -> None:
    content = {
        "schema_version": payload["schema_version"],
        "formula_version": payload["formula_version"],
        "cache_id": payload["cache_id"],
        "algorithm": payload["algorithm"],
        "force_target_mode": target_mode,
        "branches": _jsonable(payload["branches"]),
    }
    payload["binning_id"] = stable_id(content)
    payload["run_id"] = run_id(payload["experiment_id"], payload["binning_id"])


def test_binning_id_ignores_experiment_training_identity() -> None:
    parameters = inspect.signature(BinningArtifact.create).parameters
    assert "force_target_mode" in parameters
    branch = fit_fixed_linear(torch.tensor([0.1, 0.2]), num_bins=3, max_error=0.3)

    first = BinningArtifact.create(
        cache_id="cache",
        experiment_id="training-a",
        algorithm="fixed_linear_v1",
        branches={"force": branch},
        force_target_mode="atom_mean",
    )
    second = BinningArtifact.create(
        cache_id="cache",
        experiment_id="training-b",
        algorithm="fixed_linear_v1",
        branches={"force": branch},
        force_target_mode="atom_mean",
    )

    assert first.binning_id == second.binning_id
    assert first.run_id != second.run_id


def test_binning_id_includes_enabled_force_target_mode() -> None:
    parameters = inspect.signature(BinningArtifact.create).parameters
    assert "force_target_mode" in parameters
    branch = fit_fixed_linear(torch.tensor([0.1, 0.2]), num_bins=3, max_error=0.3)

    atom_mean = BinningArtifact.create(
        cache_id="cache",
        experiment_id="same-experiment",
        algorithm="fixed_linear_v1",
        branches={"force": branch},
        force_target_mode="atom_mean",
    )
    component = BinningArtifact.create(
        cache_id="cache",
        experiment_id="same-experiment",
        algorithm="fixed_linear_v1",
        branches={"force": branch},
        force_target_mode="component",
    )

    assert atom_mean.binning_id != component.binning_id


def test_safe_load_rejects_synchronized_fixed_formula_attack() -> None:
    branch = fit_fixed_linear(torch.tensor([0.0, 0.1, 0.2]), num_bins=3, max_error=0.3)
    payload = _artifact_payload(branch)
    payload["branches"]["force"]["thresholds"][0] = 0.11
    _synchronize_semantic_ids(payload, "atom_mean")

    with pytest.raises(ValueError, match="fixed_linear_v1 threshold formula"):
        BinningArtifact.from_payload(payload)


def test_safe_load_rejects_synchronized_fixed_source_attack() -> None:
    branch = fit_fixed_linear(torch.tensor([0.0, 0.1, 0.2]), num_bins=3, max_error=0.3)
    payload = _artifact_payload(branch)
    source_key = (
        "sources"
        if "sources" in payload["branches"]["force"]
        else "representative_sources"
    )
    payload["branches"]["force"][source_key] = ("median",) * 3
    _synchronize_semantic_ids(payload, "atom_mean")

    with pytest.raises(ValueError, match="fixed_linear_v1 sources"):
        BinningArtifact.from_payload(payload)


def test_safe_load_rejects_synchronized_nonfinite_fixed_parameter() -> None:
    branch = fit_fixed_linear(torch.tensor([0.0, 0.1, 0.2]), num_bins=3, max_error=0.3)
    payload = _artifact_payload(branch)
    payload["branches"]["force"]["max_error"] = float("nan")
    _synchronize_semantic_ids(payload, "atom_mean")

    with pytest.raises(ValueError, match="max_error.*finite"):
        BinningArtifact.from_payload(payload)


def test_safe_load_rejects_synchronized_log_endpoint_attack() -> None:
    branch = fit_train_quantile_log(
        torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0]), num_bins=4
    )
    payload = _artifact_payload(branch)
    payload["branches"]["force"]["lower"] *= 0.9
    _synchronize_semantic_ids(payload, "atom_mean")

    with pytest.raises(ValueError, match="train_quantile_log_v1 threshold endpoints"):
        BinningArtifact.from_payload(payload)


def test_safe_load_rejects_synchronized_log_fallback_attack() -> None:
    branch = fit_train_quantile_log(
        torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0]), num_bins=6
    )
    payload = _artifact_payload(branch)
    branch_payload = payload["branches"]["force"]
    source_key = "sources" if "sources" in branch_payload else "representative_sources"
    reps_key = "reps" if "reps" in branch_payload else "representatives"
    fallback_index = list(branch_payload[source_key]).index("analytic_fallback")
    branch_payload[reps_key][fallback_index] += 0.25
    _synchronize_semantic_ids(payload, "atom_mean")

    with pytest.raises(ValueError, match="analytic fallback"):
        BinningArtifact.from_payload(payload)


def test_safe_load_rejects_synchronized_non_float64_artifact_tensor() -> None:
    branch = fit_train_quantile_log(
        torch.arange(1.0, 11.0, dtype=torch.float64), num_bins=3
    )
    assert all(source == "median" for source in branch.representative_sources)
    payload = _artifact_payload(branch)
    payload["branches"]["force"]["reps"] = payload["branches"]["force"]["reps"].float()
    _synchronize_semantic_ids(payload, "atom_mean")

    with pytest.raises(ValueError, match="reps.*CPU float64"):
        BinningArtifact.from_payload(payload, force_target_mode="atom_mean")
