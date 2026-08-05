"""Independent on-disk validation and normalized result schema signatures."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .aggregation import validation_error_weights, weighted_mean
from .artifacts import (
    atomic_write_json,
    formal_artifact_files,
    normalize_artifact_path,
    sha256_file,
)
from .errors import HardFailure
from .ensemble_branch_checks import assert_stored_weights
from .evaluation import _comparison_pairs, _metric_summary, _uncertainty_payload
from .manifests import build_result_manifest
from .metrics import compute_correlations, compute_errors, compute_risk_coverage
from .prediction import PredictionShape, validate_prediction_payload
from .prediction_views import prediction_member_count


_FORBIDDEN_KEYS = {
    "source_path",
    "source_root",
    "migration_source",
    "old_manifest",
    "data_path",
    "data_sha256",
}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HardFailure(f"cannot parse JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise HardFailure(f"JSON artifact must be a mapping: {path.name}")
    return value


def _verify_artifact(root: Path, artifact: Mapping[str, Any], label: str) -> Path:
    relative = artifact.get("path")
    expected_hash = artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise HardFailure(f"{label} artifact descriptor is invalid")
    path = root / relative
    if normalize_artifact_path(root, path) != relative or not path.is_file():
        raise HardFailure(f"{label} artifact is missing or unsafe")
    if sha256_file(path) != expected_hash:
        raise HardFailure(f"{label} hash mismatch")
    return path


def _reject_provenance(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_KEYS:
                raise HardFailure(f"forbidden provenance field at {path}.{key}")
            _reject_provenance(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_provenance(item, f"{path}[{index}]")
    elif isinstance(value, str) and ("/Ensemble/results/FGE" in value or "\\Ensemble\\results\\FGE" in value):
        raise HardFailure(f"old result provenance leaked at {path}")


def _load_prediction(root: Path) -> tuple[dict[str, Any], PredictionShape]:
    manifest = _json(root / "prediction" / "manifest.json")
    prediction_path = _verify_artifact(root, manifest.get("artifact", {}), "prediction")
    try:
        payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise HardFailure("canonical prediction cannot be loaded") from exc
    shape = validate_prediction_payload(payload)
    symbols = manifest.get("shape_symbols")
    if symbols != {"K": shape.members, "S": shape.structures, "A": shape.atoms}:
        raise HardFailure("prediction manifest shape mismatch")
    _reject_provenance(manifest)
    _reject_provenance(payload)
    return payload, shape


def _load_training(root: Path, member_ids: list[str]) -> dict[str, Any]:
    manifest = _json(root / "training" / "manifest.json")
    members = manifest.get("members")
    if not isinstance(members, list) or [item.get("member_id") for item in members] != member_ids:
        raise HardFailure("training members are not aligned with prediction")
    for member in members:
        _verify_artifact(root, member.get("raw", {}), f"{member['member_id']} raw")
        _verify_artifact(root, member.get("ema", {}), f"{member['member_id']} ema")
    base = manifest.get("base_model_metrics")
    if not isinstance(base, dict) or set(base) != {"energy_rmse", "forces_rmse"}:
        raise HardFailure("base model validation metrics are missing")
    _reject_provenance(manifest)
    return manifest


def _assert_tensor_equal(actual: Any, expected: Tensor, label: str) -> None:
    if not isinstance(actual, Tensor):
        raise HardFailure(f"{label} is not a tensor")
    if actual.dtype != torch.float64 or actual.device.type != "cpu":
        raise HardFailure(f"{label} must be CPU float64")
    if not torch.equal(actual, expected.to(dtype=torch.float64, device="cpu")):
        raise HardFailure(f"{label} value mismatch")


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        raise HardFailure(f"cannot parse CSV artifact: {path.name}") from exc


def _optional_float(value: str) -> float | None:
    if value in {"", "None", "null"}:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise HardFailure("CSV contains NaN or Inf")
    return result


def _validate_branch(
    *,
    root: Path,
    branch: str,
    prediction: Mapping[str, Any],
    energy_weights: Tensor | None,
    force_weights: Tensor | None,
    coverages: tuple[float, ...],
    force_quantile: float,
) -> list[str]:
    branch_root = root / "evaluation" / branch
    try:
        ensemble = torch.load(branch_root / "ensemble.pt", map_location="cpu", weights_only=True)
        uncertainty = torch.load(branch_root / "uncertainty.pt", map_location="cpu", weights_only=True)
    except Exception as exc:
        raise HardFailure(f"{branch} tensor artifact cannot be loaded") from exc
    expected_energy = weighted_mean(prediction["energy_members"], energy_weights)
    expected_forces = weighted_mean(prediction["forces_members"], force_weights)
    _assert_tensor_equal(ensemble.get("energy"), expected_energy, f"{branch}.energy")
    assert_stored_weights(ensemble, energy_weights, force_weights, prediction_member_count(prediction), branch)
    _assert_tensor_equal(ensemble.get("forces"), expected_forces, f"{branch}.forces")
    expected_uncertainty = _uncertainty_payload(prediction, energy_weights, force_weights)
    for name, expected in expected_uncertainty.items():
        _assert_tensor_equal(uncertainty.get(name), expected, f"{branch}.{name}")
    errors = compute_errors(
        energy_prediction=expected_energy,
        forces_prediction=expected_forces,
        energy_reference=prediction["energy_reference"],
        forces_reference=prediction["forces_reference"],
        n_atoms=prediction["n_atoms"],
        atom_to_structure=prediction["atom_to_structure"],
        force_structure_quantile=force_quantile,
    )
    if _json(branch_root / "metrics.json") != _metric_summary(errors):
        raise HardFailure(f"{branch} metrics value mismatch")
    expected_correlations: list[dict[str, Any]] = []
    expected_risk: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name, uncertainty_values, error_values in _comparison_pairs(expected_uncertainty, errors):
        correlation = compute_correlations(uncertainty_values.reshape(-1), error_values.reshape(-1))
        expected_correlations.append(
            {"metric": name, "pearson": correlation["pearson"], "spearman": correlation["spearman"]}
        )
        warnings.extend(f"{name}: {warning}" for warning in correlation["warnings"])
        for row in compute_risk_coverage(
            uncertainty_values.reshape(-1), error_values.reshape(-1), coverages
        ):
            expected_risk.append({"metric": name, **row})
    actual_correlations = _read_csv(branch_root / "correlations.csv")
    if len(actual_correlations) != len(expected_correlations):
        raise HardFailure(f"{branch} correlation row count mismatch")
    for actual, expected in zip(actual_correlations, expected_correlations):
        if actual["metric"] != expected["metric"]:
            raise HardFailure(f"{branch} correlation metric mismatch")
        for key in ("pearson", "spearman"):
            actual_value = _optional_float(actual[key])
            expected_value = expected[key]
            if actual_value is None or expected_value is None:
                if actual_value is not expected_value:
                    raise HardFailure(f"{branch} correlation null mismatch")
            elif actual_value != expected_value:
                raise HardFailure(f"{branch} correlation value mismatch")
    actual_risk = _read_csv(branch_root / "risk_coverage.csv")
    if len(actual_risk) != len(expected_risk):
        raise HardFailure(f"{branch} risk-coverage row count mismatch")
    for actual, expected in zip(actual_risk, expected_risk):
        if (
            actual["metric"] != expected["metric"]
            or float(actual["coverage"]) != expected["coverage"]
            or float(actual["risk"]) != expected["risk"]
            or int(actual["count"]) != expected["count"]
        ):
            raise HardFailure(f"{branch} risk-coverage value mismatch")
    return warnings


def validate_result(config: Any, run_dir: Path) -> dict[str, Any]:
    """Recompute all numerical branches, then publish completion markers last."""
    root = Path(run_dir).resolve()
    prediction, _ = _load_prediction(root)
    training = _load_training(root, prediction["member_ids"])
    ensemble_config = config.section("ensemble")
    evaluation_config = config.section("evaluation")
    energy_rmse = torch.tensor(
        [member["raw_metrics"]["energy_rmse"] for member in training["members"]], dtype=torch.float64
    )
    force_rmse = torch.tensor(
        [member["raw_metrics"]["forces_rmse"] for member in training["members"]], dtype=torch.float64
    )
    energy_weights = validation_error_weights(
        energy_rmse, training["base_model_metrics"]["energy_rmse"], ensemble_config["eps_energy_ratio"]
    )
    force_weights = validation_error_weights(
        force_rmse, training["base_model_metrics"]["forces_rmse"], ensemble_config["eps_force_ratio"]
    )
    coverages = tuple(float(value) for value in evaluation_config["risk_coverages"])
    force_quantile = float(evaluation_config["force_structure_quantile"])
    numerical_warnings = _validate_branch(
        root=root,
        branch="equal_weight",
        prediction=prediction,
        energy_weights=None,
        force_weights=None,
        coverages=coverages,
        force_quantile=force_quantile,
    )
    numerical_warnings.extend(
        _validate_branch(
            root=root,
            branch="validation_weighted",
            prediction=prediction,
            energy_weights=energy_weights,
            force_weights=force_weights,
            coverages=coverages,
            force_quantile=force_quantile,
        )
    )
    report = {
        "schema_version": "fge.validation.v1",
        "status": "PASS",
        "warnings": list(training.get("warnings", [])),
        "numerical_warnings": numerical_warnings,
        "checks": {
            "artifact_hashes": "PASS",
            "prediction_schema": "PASS",
            "independent_recomputation": "PASS",
            "provenance_free": "PASS",
        },
    }
    atomic_write_json(root / "validation.json", report)
    atomic_write_json(
        root / "result_manifest.json",
        build_result_manifest(root=root, project_name=config.project_name),
    )
    return report


def _json_key_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_key_tree(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [] if not value else [_json_key_tree(value[0])]
    if value is None:
        return "null"
    return type(value).__name__


def _symbolic_shape(path: str, key: str, tensor: Tensor) -> list[str | int]:
    shape: list[str | int] = list(tensor.shape)
    if key == "energy_members":
        return ["K", "S"]
    if key == "forces_members":
        return ["K", "A", 3]
    if key == "stress_members":
        return ["K", "S", 3, 3]
    if key in {"energy_reference", "n_atoms"} or key.startswith("energy_"):
        shape[0:1] = ["S"]
    elif key == "structure_ptr":
        shape[0:1] = ["S+1"]
    elif key == "atom_to_structure" or key == "forces_reference" or key.startswith("force_atom") or key.startswith("force_component"):
        shape[0:1] = ["A"]
    elif key == "stress_reference" or key.startswith("force_structure"):
        shape[0:1] = ["S"]
    elif path.endswith("ensemble.pt") and key == "energy":
        shape[0:1] = ["S"]
    elif path.endswith("ensemble.pt") and key == "forces":
        shape[0:1] = ["A"]
    elif key.endswith("weights"):
        shape[0:1] = ["K"]
    return shape


def schema_signature(run_dir: Path) -> dict[str, Any]:
    """Describe artifact structure while normalizing experiment and K/S/A sizes."""
    root = Path(run_dir).resolve()
    artifacts: dict[str, Any] = {}
    for path in formal_artifact_files(root):
        relative = normalize_artifact_path(root, path)
        normalized = re.sub(r"member_[0-9]+", "member_{K}", relative)
        if path.suffix == ".json":
            payload = _json(path)
            artifacts[normalized] = {
                "kind": "json",
                "schema_version": payload.get("schema_version"),
                "keys": _json_key_tree(payload),
            }
        elif path.suffix == ".pt":
            payload = torch.load(path, map_location="cpu", weights_only=True)
            tensors = {
                key: {
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "rank": value.ndim,
                    "shape": _symbolic_shape(relative, key, value),
                }
                for key, value in sorted(payload.items())
                if isinstance(value, Tensor)
            }
            artifacts[normalized] = {
                "kind": "tensor",
                "schema_version": payload.get("schema_version"),
                "keys": sorted(payload),
                "tensors": tensors,
            }
        elif path.suffix == ".csv":
            rows = _read_csv(path)
            artifacts[normalized] = {
                "kind": "csv",
                "columns": list(rows[0]) if rows else [],
            }
        else:
            artifacts[normalized] = {"kind": "binary" if path.suffix == ".model" else "text"}
    return {"schema_version": "fge.schema-signature.v1", "artifacts": artifacts}
