"""Global evaluation of resumable dataset-scoped FGE prediction shards."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .aggregation import validation_error_weights, weighted_mean
from .artifacts import atomic_torch_save, atomic_write_json, sha256_file
from .derived_artifacts import DerivedLayout, verify_prediction_shard_if_present
from .errors import HardFailure
from .evaluation import (
    _atomic_text,
    _comparison_pairs,
    _csv_text,
    _load_training,
    _section,
    _stress_outputs,
    _uncertainty_payload,
)
from .metrics import compute_correlations, compute_errors, compute_risk_coverage
from .prediction import validate_prediction_payload


def _global_risk_rows(
    shard_pairs: Sequence[tuple[Tensor, Tensor]], coverages: Sequence[float]
) -> list[dict[str, float | int]]:
    """Compute exact risk-coverage after concatenating every shard vector."""
    if not shard_pairs:
        raise HardFailure("global risk-coverage has no shard vectors")
    uncertainty = torch.cat([pair[0].reshape(-1) for pair in shard_pairs])
    residual = torch.cat([pair[1].reshape(-1) for pair in shard_pairs])
    return compute_risk_coverage(uncertainty, residual, coverages)


def _dataset_pairs(
    uncertainty: Mapping[str, Tensor], errors: Mapping[str, Tensor]
) -> list[tuple[str, Tensor, Tensor]]:
    pairs = _comparison_pairs(uncertainty, errors)
    if "stress_component_std" in uncertainty:
        pairs.append(
            (
                "stress_component_std",
                uncertainty["stress_component_std"],
                errors["stress_component"],
            )
        )
    return pairs


def _json_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HardFailure(f"{description} cannot be loaded") from exc
    if not isinstance(value, dict):
        raise HardFailure(f"{description} must be a mapping")
    return value


def _prediction_shards(layout: DerivedLayout) -> tuple[dict[str, Any], ...]:
    manifest = _json_mapping(layout.prediction_manifest, "derived prediction manifest")
    if manifest.get("schema_version") != "fge.derived-prediction.v1" or manifest.get(
        "status"
    ) != "PASS":
        raise HardFailure("derived prediction manifest is not PASS")
    if manifest.get("experiment") != layout.experiment:
        raise HardFailure("derived prediction experiment does not match layout")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("dataset") != layout.dataset:
        raise HardFailure("derived prediction dataset does not match layout")
    rows = manifest.get("shards")
    if not isinstance(rows, list) or not rows:
        raise HardFailure("derived prediction manifest has no shards")
    expected_count = manifest.get("shard_count")
    if expected_count != len(rows):
        raise HardFailure("derived prediction shard count is inconsistent")

    payloads: list[dict[str, Any]] = []
    for expected_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("index") != expected_index:
            raise HardFailure("derived prediction shard indices are not contiguous")
        shard_manifest_path = layout.prediction_shard_manifest(expected_index)
        shard_manifest = _json_mapping(
            shard_manifest_path, f"prediction shard {expected_index} manifest"
        )
        artifact = row.get("manifest")
        if not isinstance(artifact, dict):
            raise HardFailure("derived prediction shard reference is invalid")
        expected_path = shard_manifest_path.relative_to(layout.prediction_dir).as_posix()
        if artifact.get("path") != expected_path:
            raise HardFailure("derived prediction shard manifest path is invalid")
        expected_hash = artifact.get("sha256")
        if expected_hash is not None and expected_hash != sha256_file(shard_manifest_path):
            raise HardFailure("derived prediction shard manifest SHA-256 mismatch")
        signature = shard_manifest.get("signature")
        if not isinstance(signature, Mapping) or not verify_prediction_shard_if_present(
            layout, expected_index, signature
        ):
            raise HardFailure(f"prediction shard {expected_index} is missing")
        try:
            payload = torch.load(
                layout.prediction_shard(expected_index),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise HardFailure(f"prediction shard {expected_index} cannot be loaded") from exc
        if not isinstance(payload, dict):
            raise HardFailure(f"prediction shard {expected_index} is not a mapping")
        validate_prediction_payload(payload)
        payloads.append(payload)
    return tuple(payloads)


def _weights(config: Any, member_ids: Sequence[str]) -> tuple[Tensor, Tensor, dict[str, Any]]:
    training = _load_training(
        Path(config.output_dir) / "training" / "manifest.json", member_ids
    )
    ensemble = _section(config, "ensemble")
    try:
        energy_rmse = torch.tensor(
            [member["raw_metrics"]["energy_rmse"] for member in training["members"]],
            dtype=torch.float64,
        )
        force_rmse = torch.tensor(
            [member["raw_metrics"]["forces_rmse"] for member in training["members"]],
            dtype=torch.float64,
        )
        base_energy = float(training["base_model_metrics"]["energy_rmse"])
        base_force = float(training["base_model_metrics"]["forces_rmse"])
        energy_eps = float(ensemble["eps_energy_ratio"])
        force_eps = float(ensemble["eps_force_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HardFailure("dataset evaluation weights are incomplete") from exc
    return (
        validation_error_weights(energy_rmse, base_energy, energy_eps),
        validation_error_weights(force_rmse, base_force, force_eps),
        training,
    )


def _branch_shard(
    prediction: Mapping[str, Any],
    energy_weights: Tensor | None,
    force_weights: Tensor | None,
    force_quantile: float,
) -> dict[str, Any]:
    member_count = int(prediction["energy_members"].shape[0])
    energy = weighted_mean(prediction["energy_members"], energy_weights)
    forces = weighted_mean(prediction["forces_members"], force_weights)
    ensemble: dict[str, Tensor] = {
        "energy": energy.to(device="cpu", dtype=torch.float64),
        "forces": forces.to(device="cpu", dtype=torch.float64),
        "energy_weights": (
            torch.full((member_count,), 1.0 / member_count, dtype=torch.float64)
            if energy_weights is None
            else energy_weights.to(device="cpu", dtype=torch.float64)
        ),
        "force_weights": (
            torch.full((member_count,), 1.0 / member_count, dtype=torch.float64)
            if force_weights is None
            else force_weights.to(device="cpu", dtype=torch.float64)
        ),
    }
    uncertainty = _uncertainty_payload(prediction, energy_weights, force_weights)
    errors = compute_errors(
        energy_prediction=energy,
        forces_prediction=forces,
        energy_reference=prediction["energy_reference"],
        forces_reference=prediction["forces_reference"],
        n_atoms=prediction["n_atoms"],
        atom_to_structure=prediction["atom_to_structure"],
        force_structure_quantile=force_quantile,
    )
    references: dict[str, Tensor] = {
        "energy": prediction["energy_reference"],
        "forces": prediction["forces_reference"],
        "n_atoms": prediction["n_atoms"],
        "atom_to_structure": prediction["atom_to_structure"],
    }
    if prediction["observables"] == ["energy", "forces", "stress"]:
        stress_ensemble, stress_uncertainty, stress_errors = _stress_outputs(
            prediction, force_weights
        )
        ensemble.update(stress_ensemble)
        uncertainty.update(stress_uncertainty)
        errors.update(stress_errors)
        references["stress"] = prediction["stress_reference"]
    return {
        "ensemble": ensemble,
        "uncertainty": uncertainty,
        "errors": errors,
        "references": references,
    }


def _metric_summary(shards: Sequence[Mapping[str, Tensor]]) -> dict[str, dict[str, float | int]]:
    if not shards:
        raise HardFailure("dataset evaluation has no error shards")
    names = tuple(shards[0])
    if any(tuple(shard) != names for shard in shards[1:]):
        raise HardFailure("dataset evaluation shard metrics are inconsistent")
    result: dict[str, dict[str, float | int]] = {}
    for name in names:
        vectors = [shard[name].reshape(-1).to(dtype=torch.float64) for shard in shards]
        count = sum(int(vector.numel()) for vector in vectors)
        absolute_sum = sum(float(vector.sum().item()) for vector in vectors)
        squared_sum = sum(float(vector.square().sum().item()) for vector in vectors)
        result[name] = {
            "mae": absolute_sum / count,
            "rmse": (squared_sum / count) ** 0.5,
            "count": count,
        }
    return result


def _quality_warnings(
    branch: str,
    metrics: Mapping[str, Mapping[str, float | int]],
    correlations: Sequence[Mapping[str, Any]],
    training: Mapping[str, Any],
    multiplier: float,
) -> list[str]:
    warnings: list[str] = []
    base = training["base_model_metrics"]
    checks = (
        ("energy_total", "energy_rmse"),
        ("force_component", "forces_rmse"),
    )
    for metric, base_name in checks:
        if float(metrics[metric]["rmse"]) > float(base[base_name]) * multiplier:
            warnings.append(f"{branch}: {metric} RMSE exceeds weak threshold")
    for row in correlations:
        for coefficient in ("pearson", "spearman"):
            value = row[coefficient]
            if value is None:
                warnings.append(f"{branch}: {row['metric']} correlation is undefined")
            elif abs(float(value)) < 0.2:
                warnings.append(f"{branch}: {row['metric']} {coefficient} is below 0.2")
    return warnings


def _evaluate_branch(
    *,
    branch: str,
    layout: DerivedLayout,
    predictions: Sequence[Mapping[str, Any]],
    energy_weights: Tensor | None,
    force_weights: Tensor | None,
    coverages: Sequence[float],
    force_quantile: float,
    training: Mapping[str, Any],
    weak_multiplier: float,
) -> tuple[dict[str, Any], list[str]]:
    branch_root = layout.evaluation_dir / branch
    shard_outputs: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        output = _branch_shard(
            prediction, energy_weights, force_weights, force_quantile
        )
        path = branch_root / "shards" / f"shard_{index:06d}.pt"
        atomic_torch_save(
            path,
            {
                "schema_version": "fge.derived-evaluation-shard.v1",
                "branch": branch,
                "shard_index": index,
                **output,
            },
        )
        shard_outputs.append(output)
        shard_rows.append(
            {"index": index, "path": path.relative_to(branch_root).as_posix(), "sha256": sha256_file(path)}
        )

    metrics = _metric_summary([output["errors"] for output in shard_outputs])
    pair_names = [name for name, _, _ in _dataset_pairs(
        shard_outputs[0]["uncertainty"], shard_outputs[0]["errors"]
    )]
    correlation_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for name in pair_names:
        pairs = []
        for output in shard_outputs:
            matched = {
                pair_name: (uncertainty, error)
                for pair_name, uncertainty, error in _dataset_pairs(
                    output["uncertainty"], output["errors"]
                )
            }
            pairs.append(matched[name])
        uncertainty = torch.cat([pair[0].reshape(-1) for pair in pairs])
        residual = torch.cat([pair[1].reshape(-1) for pair in pairs])
        correlation = compute_correlations(uncertainty, residual)
        correlation_rows.append(
            {"metric": name, "pearson": correlation["pearson"], "spearman": correlation["spearman"]}
        )
        for row in _global_risk_rows(pairs, coverages):
            risk_rows.append({"metric": name, **row})

    shape_symbols = {
        "S": sum(int(prediction["energy_reference"].numel()) for prediction in predictions),
        "A": sum(int(prediction["forces_reference"].shape[0]) for prediction in predictions),
    }
    atomic_torch_save(
        branch_root / "ensemble.pt",
        {
            "schema_version": "fge.derived-ensemble-index.v1",
            "branch": branch,
            "shape_symbols": shape_symbols,
            "shards": shard_rows,
        },
    )
    atomic_write_json(branch_root / "metrics.json", metrics)
    _atomic_text(
        branch_root / "correlations.csv",
        _csv_text(("metric", "pearson", "spearman"), correlation_rows),
    )
    _atomic_text(
        branch_root / "risk_coverage.csv",
        _csv_text(("metric", "coverage", "risk", "count"), risk_rows),
    )
    warnings = _quality_warnings(
        branch, metrics, correlation_rows, training, weak_multiplier
    )
    return {"metrics": metrics, "shape_symbols": shape_symbols}, warnings


def evaluate_dataset(config: Any, layout: DerivedLayout) -> Path:
    """Evaluate all verified prediction shards and publish global metrics."""
    predictions = _prediction_shards(layout)
    member_ids = predictions[0]["member_ids"]
    if any(prediction["member_ids"] != member_ids for prediction in predictions[1:]):
        raise HardFailure("prediction shard members are not aligned")
    if any(
        prediction["observables"] != predictions[0]["observables"]
        for prediction in predictions[1:]
    ):
        raise HardFailure("prediction shard observables are not aligned")
    energy_weights, force_weights, training = _weights(config, member_ids)
    evaluation = _section(config, "evaluation")
    quality = _section(config, "quality")
    try:
        coverages = tuple(float(value) for value in evaluation["risk_coverages"])
        force_quantile = float(evaluation["force_structure_quantile"])
        weak_multiplier = float(quality["weak_rmse_multiplier"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HardFailure("dataset evaluation configuration is incomplete") from exc

    equal, equal_warnings = _evaluate_branch(
        branch="equal_weight",
        layout=layout,
        predictions=predictions,
        energy_weights=None,
        force_weights=None,
        coverages=coverages,
        force_quantile=force_quantile,
        training=training,
        weak_multiplier=weak_multiplier,
    )
    weighted, weighted_warnings = _evaluate_branch(
        branch="validation_weighted",
        layout=layout,
        predictions=predictions,
        energy_weights=energy_weights,
        force_weights=force_weights,
        coverages=coverages,
        force_quantile=force_quantile,
        training=training,
        weak_multiplier=weak_multiplier,
    )
    warnings = equal_warnings + weighted_warnings
    _atomic_text(
        layout.evaluation_dir / "report.md",
        "# FGE Dataset Evaluation\n\n"
        f"- Dataset: {layout.dataset}\n"
        f"- Prediction shards: {len(predictions)}\n"
        f"- Warnings: {len(warnings)}\n",
    )
    audit_path = layout.evaluation_dir / "audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": "fge.derived-evaluation.v1",
            "status": "PASS",
            "experiment": layout.experiment,
            "dataset": layout.dataset,
            "member_ids": member_ids,
            "observables": predictions[0]["observables"],
            "shard_count": len(predictions),
            "branches": {
                "equal_weight": equal,
                "validation_weighted": weighted,
            },
            "warnings": warnings,
        },
    )
    return audit_path
