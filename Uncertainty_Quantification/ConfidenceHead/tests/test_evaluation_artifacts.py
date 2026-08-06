"""Tests for immutable ConfidenceHead test-evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from confidence_head.artifacts import atomic_torch_save, load_torch_artifact
from confidence_head.evaluation_artifacts import (
    EVALUATION_FORMULA_VERSION,
    EVALUATION_INPUT_FILES,
    EVALUATION_SCHEMA_VERSION,
    EvaluationConflictError,
    EvaluationPaths,
    commit_evaluation,
    validate_or_reuse_evaluation,
    validate_prediction_payload,
)


IDENTITY = {
    "run_id": "1" * 64,
    "experiment_id": "2" * 64,
    "cache_id": "3" * 64,
    "binning_id": "4" * 64,
}
INPUT_HASHES = {name: str(index) * 64 for index, name in enumerate(EVALUATION_INPUT_FILES, 1)}


def _branch(sample_count: int) -> dict[str, torch.Tensor]:
    logits = torch.tensor(
        [[2.0, 0.0] if index % 2 == 0 else [0.0, 2.0] for index in range(sample_count)]
    )
    return {
        "logits": logits,
        "labels": torch.arange(sample_count, dtype=torch.int64) % 2,
        "errors": torch.linspace(0.1, 0.1 * sample_count, sample_count),
        "expected_errors": torch.softmax(logits.to(torch.float64), dim=-1)
        @ torch.tensor([0.1, 0.3], dtype=torch.float64),
    }


def _predictions(branch: str = "force") -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "formula_version": EVALUATION_FORMULA_VERSION,
        "split": "test",
        "identity": dict(IDENTITY),
        "enabled_branches": (branch,),
        "structure_ids": ("test-0", "test-1"),
        "structure_offsets": torch.tensor([0, 1, 3], dtype=torch.int64),
        "force_target_mode": "atom_mean" if branch == "force" else None,
    }
    payload[branch] = _branch(3 if branch == "force" else 2)
    return payload


def _branch_metrics(sample_count: int) -> dict[str, int | float]:
    return {
        "sample_count": sample_count,
        "accuracy": 0.5,
        "brier": 0.25,
        "mean_expected_error": 0.2,
        "mean_observed_error": 0.3,
        "mae_expected_vs_error": 0.1,
        "spearman_expected_vs_error": 0.75,
    }


def _metrics(branch: str = "force") -> dict[str, object]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "formula_version": EVALUATION_FORMULA_VERSION,
        "split": "test",
        "identity": dict(IDENTITY),
        "branches": {
            branch: _branch_metrics(3 if branch == "force" else 2),
        },
    }


@pytest.fixture
def paths(tmp_path: Path) -> EvaluationPaths:
    run_root = tmp_path / "run-root"
    (run_root / "run").mkdir(parents=True)
    return EvaluationPaths.from_run_root(run_root)


def test_prediction_schema_binds_structure_and_force_sample_axes() -> None:
    payload = _predictions()

    result = validate_prediction_payload(payload, expected_identity=IDENTITY)

    assert result["structure_ids"] == ("test-0", "test-1")
    assert result["structure_offsets"].tolist() == [0, 1, 3]
    assert result["force"]["logits"].shape == (3, 2)


def test_energy_prediction_schema_uses_structure_sample_axis() -> None:
    payload = _predictions("energy")

    result = validate_prediction_payload(payload, expected_identity=IDENTITY)

    assert result["force_target_mode"] is None
    assert result["energy"]["logits"].shape == (2, 2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update({"unexpected": "field"}),
            "keys",
        ),
        (
            lambda value: value["structure_offsets"].__setitem__(2, 1),
            "offset",
        ),
        (
            lambda value: value["force"].update({"errors": torch.ones(2)}),
            "sample",
        ),
        (
            lambda value: value.update({"enabled_branches": ("energy",)}),
            "branches",
        ),
        (
            lambda value: value["force"]["logits"].__setitem__(
                (0, 0), float("nan")
            ),
            "finite",
        ),
    ],
)
def test_prediction_schema_rejects_malformed_payloads(mutation, message: str) -> None:
    payload = _predictions()
    mutation(payload)

    with pytest.raises(EvaluationConflictError, match=message):
        validate_prediction_payload(payload, expected_identity=IDENTITY)


def test_partial_evaluation_without_manifest_is_rejected(paths: EvaluationPaths) -> None:
    atomic_torch_save(paths.predictions, _predictions())

    with pytest.raises(EvaluationConflictError, match="partial"):
        validate_or_reuse_evaluation(paths, IDENTITY, INPUT_HASHES)
    assert paths.predictions.is_file()
    assert not paths.manifest.exists()


def test_complete_identical_evaluation_is_reused_without_rewrite(
    paths: EvaluationPaths,
) -> None:
    commit_evaluation(paths, _predictions(), _metrics(), INPUT_HASHES)
    before = {path: path.stat().st_mtime_ns for path in paths.outputs}

    assert validate_or_reuse_evaluation(paths, IDENTITY, INPUT_HASHES) is True
    assert before == {path: path.stat().st_mtime_ns for path in paths.outputs}


def test_modified_predictions_are_rejected_and_preserved(paths: EvaluationPaths) -> None:
    commit_evaluation(paths, _predictions(), _metrics(), INPUT_HASHES)
    paths.predictions.write_bytes(paths.predictions.read_bytes() + b"tamper")
    evidence = paths.predictions.read_bytes()

    with pytest.raises(EvaluationConflictError, match="sha256"):
        validate_or_reuse_evaluation(paths, IDENTITY, INPUT_HASHES)
    assert paths.predictions.read_bytes() == evidence


def test_changed_input_hashes_are_rejected(paths: EvaluationPaths) -> None:
    commit_evaluation(paths, _predictions(), _metrics(), INPUT_HASHES)
    changed = dict(INPUT_HASHES)
    changed["best.pt"] = "f" * 64

    with pytest.raises(EvaluationConflictError, match="input"):
        validate_or_reuse_evaluation(paths, IDENTITY, changed)


def test_metrics_sample_count_must_match_predictions(paths: EvaluationPaths) -> None:
    metrics = _metrics()
    metrics["branches"]["force"]["sample_count"] = 2

    with pytest.raises(EvaluationConflictError, match="sample_count"):
        commit_evaluation(paths, _predictions(), metrics, INPUT_HASHES)
    assert not any(path.exists() for path in paths.outputs)


def test_manifest_is_committed_after_data_artifacts(
    paths: EvaluationPaths, monkeypatch
) -> None:
    import confidence_head.evaluation_artifacts as module

    real_json_dump = module.atomic_json_dump

    def fail_metrics(destination: Path, value: object) -> None:
        if destination == paths.metrics:
            raise RuntimeError("injected metrics write failure")
        real_json_dump(destination, value)

    monkeypatch.setattr(module, "atomic_json_dump", fail_metrics)
    with pytest.raises(RuntimeError, match="injected"):
        commit_evaluation(paths, _predictions(), _metrics(), INPUT_HASHES)

    assert paths.predictions.is_file()
    assert not paths.metrics.exists()
    assert not paths.manifest.exists()


def test_manifest_records_exact_output_hashes(paths: EvaluationPaths) -> None:
    commit_evaluation(paths, _predictions(), _metrics(), INPUT_HASHES)

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

    assert manifest["identity"] == IDENTITY
    assert manifest["input_sha256"] == INPUT_HASHES
    assert set(manifest["output_sha256"]) == {
        "test_predictions.pt",
        "test_metrics.json",
    }
    assert load_torch_artifact(paths.predictions)["split"] == "test"
