from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Callable

import pytest

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.postprocessing.e0_correction.models import (
    CALIBRATION_SCHEMA_VERSION,
    CORRECTION_SCHEMA_VERSION,
    INTEGRITY_SCHEMA_VERSION,
    CalibrationFit,
    CorrectionMethod,
    CorrectionSummary,
    E0DatasetConfig,
    E0RunConfig,
    ExperimentConfig,
    WarningRecord,
    load_correction_manifest,
    validate_calibration_manifest,
    validate_correction_manifest,
    validate_integrity_audit,
)


EXPERIMENT_CONFIGS = (
    "mace_fge_full_gpu_b64",
    "mace_fge_full_gpu_b64_lr1e-7_1e-6",
    "mace_fge_full_gpu_b64_lr1e-6_1e-5",
    "mace_fge_full_gpu_b64_lr1e-5_1e-4",
)


def _dataset_dict() -> dict[str, Any]:
    return {
        "test_label": "mad_r2scan_test",
        "test_path": "datasets/mad_r2scan/test.xyz",
        "val_label": "mad_r2scan_val",
        "val_path": "datasets/mad_r2scan/val.xyz",
        "energy": "energy",
        "forces": "forces",
        "atomization_energy": "atomization_energy",
        "head": "Default",
        "stress": False,
        "test_forward": False,
    }


def _run_dict() -> dict[str, Any]:
    return {
        "schema_version": "fge.e0-run.v1",
        "dataset": _dataset_dict(),
        "experiments": [
            {
                "label": f"experiment_{index:02d}",
                "config": f"configs/{name}.yaml",
            }
            for index, name in enumerate(EXPERIMENT_CONFIGS, start=1)
        ],
        "methods": ["direct_test_e0", "model_aware_val_e0"],
    }


def _calibration_manifest() -> dict[str, Any]:
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "split": "val",
        "member_ids": ["member_01", "member_02"],
        "structure_ids": ["structure_0001", "structure_0002"],
        "atomic_numbers": [1, 8],
        "composition": [[2, 1], [1, 1]],
        "energy_members": [[-10.0, -5.0], [-9.8, -4.9]],
        "energy_reference": [-10.1, -5.2],
        "n_atoms": [3, 2],
    }


def _correction_manifest() -> dict[str, Any]:
    return {
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "method": "model_aware_val_e0",
        "title": "Model-aware val-calibrated E0",
        "calibration_split": "val",
        "application_split": "test",
        "uses_test_reference_labels": False,
        "evaluation_role": "calibrated_test",
        "validation_decontamination": "exclude_test_identity_overlap_from_val",
        "overlap_removed_count": 2,
        "experiment_label": "experiment_01",
        "dataset_label": "mad_r2scan_test",
        "member_ids": ["member_01", "member_02"],
        "observables": ["energy", "forces"],
        "shape": {"K": 2, "S": 2, "A": 5},
        "structure_range": {"start": 0, "stop": 2},
        "calibration": {
            "rank_min": 2,
            "rank_max": 2,
            "residual_rmse_max": 0.1,
        },
        "warnings": [],
    }


def _integrity_audit() -> dict[str, Any]:
    return {
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "excluded_structure_ids": ["duplicate_0001", "duplicate_0002"],
        "input_sha256": {"test_extxyz": "a" * 64, "val_extxyz": "b" * 64},
        "raw_artifact_sha256": {"prediction": "c" * 64},
        "output_sha256": {"corrected_prediction": "d" * 64},
        "remote_paths": {"test_extxyz": "/remote/private/test.xyz"},
        "runtime": {"host": "compute-node", "python": "3.11"},
    }


@pytest.mark.parametrize(
    ("name", "title", "split", "uses_test", "role"),
    [
        (
            "direct_test_e0",
            "Direct test-informed E0",
            "test",
            True,
            "transductive_diagnostic",
        ),
        (
            "model_aware_val_e0",
            "Model-aware val-calibrated E0",
            "val",
            False,
            "calibrated_test",
        ),
    ],
)
def test_correction_methods_have_fixed_metadata(
    name: str, title: str, split: str, uses_test: bool, role: str
) -> None:
    method = CorrectionMethod.from_name(name)

    assert method.machine_name == name
    assert method.title == title
    assert method.calibration_split == split
    assert method.uses_test_reference_labels is uses_test
    assert method.evaluation_role == role


@pytest.mark.parametrize("name", ["", "direct", "test_e0", "unknown"])
def test_unknown_correction_method_is_a_hard_failure(name: str) -> None:
    with pytest.raises(HardFailure):
        CorrectionMethod.from_name(name)


def test_run_config_has_four_contiguous_experiments_and_two_methods() -> None:
    run = E0RunConfig.from_dict(_run_dict())

    assert tuple(item.label for item in run.experiments) == (
        "experiment_01",
        "experiment_02",
        "experiment_03",
        "experiment_04",
    )
    assert tuple(Path(item.config).stem for item in run.experiments) == EXPERIMENT_CONFIGS
    assert tuple(method.machine_name for method in run.methods) == (
        "direct_test_e0",
        "model_aware_val_e0",
    )
    assert run.dataset.energy == "energy"
    assert run.dataset.forces == "forces"
    assert run.dataset.atomization_energy == "atomization_energy"
    assert run.dataset.head == "Default"


@pytest.mark.parametrize("field", ["energy", "forces", "atomization_energy", "head"])
def test_dataset_requires_explicit_reference_fields(field: str) -> None:
    payload = _dataset_dict()
    payload.pop(field)

    with pytest.raises(HardFailure):
        E0DatasetConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("factory", "extra_key"),
    [
        (_dataset_dict, "energy_key"),
        (lambda: _run_dict()["experiments"][0], "checkpoint"),
        (_run_dict, "output_root"),
    ],
)
def test_config_models_reject_unknown_keys(
    factory: Callable[[], dict[str, Any]], extra_key: str
) -> None:
    payload = factory()
    payload[extra_key] = "unexpected"

    parser = (
        E0DatasetConfig.from_dict
        if "test_label" in payload
        else ExperimentConfig.from_dict
        if "label" in payload
        else E0RunConfig.from_dict
    )
    with pytest.raises(HardFailure):
        parser(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test_label", "../mad_r2scan_test"),
        ("test_label", "mad/r2scan/test"),
        ("test_path", "/srv/data/test.xyz"),
        ("test_path", "../data/test.xyz"),
        ("test_path", "datasets\\mad\\test.xyz"),
    ],
)
def test_dataset_rejects_unsafe_logical_labels_and_paths(
    field: str, value: str
) -> None:
    payload = _dataset_dict()
    payload[field] = value

    with pytest.raises(HardFailure):
        E0DatasetConfig.from_dict(payload)


@pytest.mark.parametrize("shared_field", ["label", "path"])
def test_test_and_validation_must_be_distinct_logical_splits(
    shared_field: str,
) -> None:
    payload = _dataset_dict()
    payload[f"val_{shared_field}"] = payload[f"test_{shared_field}"]

    with pytest.raises(HardFailure):
        E0DatasetConfig.from_dict(payload)


@pytest.mark.parametrize("field", ["stress", "test_forward"])
def test_stress_and_test_forward_are_always_disabled(field: str) -> None:
    payload = _dataset_dict()
    payload[field] = True

    with pytest.raises(HardFailure):
        E0DatasetConfig.from_dict(payload)


@pytest.mark.parametrize("violation", ["labels", "configs", "continuity"])
def test_experiment_registry_is_contiguous_unique_and_one_to_one(
    violation: str,
) -> None:
    payload = _run_dict()
    experiments = payload["experiments"]
    if violation == "labels":
        experiments[1]["label"] = experiments[0]["label"]
    elif violation == "configs":
        experiments[1]["config"] = experiments[0]["config"]
    else:
        experiments[1]["label"] = "experiment_03"

    with pytest.raises(HardFailure):
        E0RunConfig.from_dict(payload)


def test_run_config_rejects_duplicate_methods() -> None:
    payload = _run_dict()
    payload["methods"] = ["direct_test_e0", "direct_test_e0"]

    with pytest.raises(HardFailure):
        E0RunConfig.from_dict(payload)


def test_contract_records_are_immutable() -> None:
    dataset = E0DatasetConfig.from_dict(_dataset_dict())
    experiment = ExperimentConfig.from_dict(_run_dict()["experiments"][0])
    run = E0RunConfig.from_dict(_run_dict())
    fit = CalibrationFit(
        member_id="member_01",
        atomic_numbers=(1, 8),
        delta_e0=(0.1, -0.2),
        rank=2,
        residual_rmse=0.05,
        condition_number=3.0,
    )
    summary = CorrectionSummary(
        structure_count=2,
        atom_count=5,
        energy_rmse_before=4.0,
        energy_rmse_after=0.5,
    )
    warning = WarningRecord(code="weak_improvement", message="RMSE changed little")

    for record, field in (
        (dataset, "head"),
        (experiment, "label"),
        (run, "methods"),
        (fit, "rank"),
        (summary, "structure_count"),
        (warning, "code"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, None)


@pytest.mark.parametrize(
    ("validator", "factory"),
    [
        (validate_calibration_manifest, _calibration_manifest),
        (validate_correction_manifest, _correction_manifest),
        (validate_integrity_audit, _integrity_audit),
    ],
)
def test_schema_validators_accept_only_exact_keys(
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    factory: Callable[[], dict[str, Any]],
) -> None:
    payload = factory()
    assert validator(payload) == payload

    payload["unexpected"] = True
    with pytest.raises(HardFailure):
        validator(payload)


def test_nested_correction_schema_is_exact() -> None:
    payload = _correction_manifest()
    payload["calibration"]["input_sha256"] = "e" * 64

    with pytest.raises(HardFailure):
        validate_correction_manifest(payload)


@pytest.mark.parametrize(
    "leaked_source",
    [
        "/HOME/private/datasets/test.xyz",
        "mace/Ensemble/outputs/legacy-run",
        "outputs/raw/checkpoints/member_01.model",
    ],
)
def test_published_correction_manifest_is_source_neutral(
    leaked_source: str,
) -> None:
    payload = _correction_manifest()
    payload["warnings"] = [{"code": "source_leak", "message": leaked_source}]

    with pytest.raises(HardFailure):
        validate_correction_manifest(payload)


def test_correction_method_metadata_cannot_drift_in_manifest() -> None:
    payload = _correction_manifest()
    payload["uses_test_reference_labels"] = True

    with pytest.raises(HardFailure):
        validate_correction_manifest(payload)


def test_internal_integrity_audit_is_not_a_public_correction_manifest(
    tmp_path: Path,
) -> None:
    audit = _integrity_audit()
    assert validate_integrity_audit(audit) == audit

    with pytest.raises(HardFailure):
        validate_correction_manifest(audit)

    path = tmp_path / "correction_manifest.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(HardFailure):
        load_correction_manifest(path)


def test_public_reader_returns_only_validated_correction_manifest(
    tmp_path: Path,
) -> None:
    payload = _correction_manifest()
    path = tmp_path / "correction_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_correction_manifest(path) == payload
