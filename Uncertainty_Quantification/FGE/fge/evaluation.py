"""Read-only evaluation of canonical FGE prediction tensors."""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .aggregation import validation_error_weights, weighted_mean
from .artifacts import atomic_torch_save, atomic_write_json, sibling_temporary_path
from .errors import HardFailure
from .metrics import (
    compute_correlations,
    compute_errors,
    compute_risk_coverage,
    compute_stress_errors,
)
from .uncertainty import (
    reduce_atoms_by_structure,
    scalar_gmd,
    unbiased_std,
    vector_gmd,
    vector_std,
)


def _section(config: Any, name: str) -> Mapping[str, Any]:
    value = config.section(name) if hasattr(config, "section") else config.get(name)
    if not isinstance(value, Mapping):
        raise HardFailure(f"configuration section {name!r} is missing")
    return value


def _atomic_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sibling_temporary_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def _require_tensor(
    payload: Mapping[str, Any], name: str, shape: tuple[int, ...], dtype: torch.dtype
) -> Tensor:
    value = payload.get(name)
    if not isinstance(value, Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
        raise HardFailure(f"canonical prediction field {name!r} has invalid shape or dtype")
    result = value.detach().to(device="cpu")
    if result.is_floating_point() and not torch.isfinite(result).all().item():
        raise HardFailure(f"canonical prediction field {name!r} contains NaN or Inf")
    return result


def _load_prediction(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HardFailure("canonical test prediction is missing")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise HardFailure("canonical test prediction cannot be loaded") from exc
    if not isinstance(payload, dict):
        raise HardFailure("canonical prediction must be a mapping")
    fixed = {
        "schema_version": "fge.prediction.v1",
        "split": "test",
        "member_source": "raw",
        "observables": ["energy", "forces"],
    }
    for name, expected in fixed.items():
        if payload.get(name) != expected:
            raise HardFailure(f"canonical prediction field {name!r} is invalid")
    member_ids = payload.get("member_ids")
    if not isinstance(member_ids, list) or len(member_ids) < 2 or not all(
        isinstance(value, str) for value in member_ids
    ):
        raise HardFailure("canonical prediction member_ids are invalid")
    member_count = len(member_ids)
    energy_reference = payload.get("energy_reference")
    forces_reference = payload.get("forces_reference")
    if not isinstance(energy_reference, Tensor) or energy_reference.ndim != 1:
        raise HardFailure("canonical energy reference is invalid")
    if not isinstance(forces_reference, Tensor) or forces_reference.ndim != 2:
        raise HardFailure("canonical forces reference is invalid")
    structure_count = energy_reference.shape[0]
    atom_count = forces_reference.shape[0]
    result = dict(payload)
    result["energy_members"] = _require_tensor(
        payload, "energy_members", (member_count, structure_count), torch.float64
    )
    result["forces_members"] = _require_tensor(
        payload, "forces_members", (member_count, atom_count, 3), torch.float64
    )
    result["energy_reference"] = _require_tensor(
        payload, "energy_reference", (structure_count,), torch.float64
    )
    result["forces_reference"] = _require_tensor(
        payload, "forces_reference", (atom_count, 3), torch.float64
    )
    result["n_atoms"] = _require_tensor(
        payload, "n_atoms", (structure_count,), torch.int64
    )
    result["atom_to_structure"] = _require_tensor(
        payload, "atom_to_structure", (atom_count,), torch.int64
    )
    result["structure_ptr"] = _require_tensor(
        payload, "structure_ptr", (structure_count + 1,), torch.int64
    )
    if (result["n_atoms"] <= 0).any().item() or int(result["n_atoms"].sum()) != atom_count:
        raise HardFailure("canonical n_atoms does not match force atoms")
    expected_mapping = torch.repeat_interleave(
        torch.arange(structure_count, dtype=torch.int64), result["n_atoms"]
    )
    expected_ptr = torch.cat(
        (torch.zeros(1, dtype=torch.int64), result["n_atoms"].cumsum(dim=0))
    )
    if not torch.equal(result["atom_to_structure"], expected_mapping) or not torch.equal(
        result["structure_ptr"], expected_ptr
    ):
        raise HardFailure("canonical structure mapping is inconsistent")
    return result


def _load_training(path: Path, member_ids: Sequence[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HardFailure("training manifest cannot be loaded") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("members"), list):
        raise HardFailure("training manifest has no members")
    if [member.get("member_id") for member in payload["members"]] != list(member_ids):
        raise HardFailure("training and prediction members are not aligned")
    base = payload.get("base_model_metrics")
    if not isinstance(base, dict):
        raise HardFailure("training manifest has no base_model_metrics")
    return payload


def _metric_summary(errors: Mapping[str, Tensor]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, values in errors.items():
        flattened = values.reshape(-1)
        result[name] = {
            "mae": float(flattened.mean().item()),
            "rmse": float(torch.sqrt(flattened.square().mean()).item()),
            "count": int(flattened.numel()),
        }
    return result


def _uncertainty_payload(
    prediction: Mapping[str, Any], energy_weights: Tensor | None, force_weights: Tensor | None
) -> dict[str, Tensor]:
    energy_members = prediction["energy_members"]
    forces_members = prediction["forces_members"]
    n_atoms = prediction["n_atoms"].to(dtype=torch.float64)
    mapping = prediction["atom_to_structure"]
    structure_count = int(n_atoms.numel())
    energy_std = unbiased_std(energy_members, energy_weights)
    energy_gmd = scalar_gmd(energy_members, energy_weights)
    force_atom_std = vector_std(forces_members, force_weights)
    force_atom_gmd = vector_gmd(forces_members, force_weights)
    std_structure = reduce_atoms_by_structure(force_atom_std, mapping, structure_count)
    gmd_structure = reduce_atoms_by_structure(force_atom_gmd, mapping, structure_count)
    return {
        "energy_total_std": energy_std,
        "energy_per_atom_std": energy_std / n_atoms,
        "energy_total_gmd": energy_gmd,
        "energy_per_atom_gmd": energy_gmd / n_atoms,
        "force_component_std": unbiased_std(forces_members, force_weights),
        "force_atom_vector_std": force_atom_std,
        "force_structure_mean_std": std_structure["mean"],
        "force_structure_max_std": std_structure["max"],
        "force_structure_q95_std": std_structure["q95"],
        "force_component_gmd": scalar_gmd(forces_members, force_weights),
        "force_atom_vector_gmd": force_atom_gmd,
        "force_structure_mean_gmd": gmd_structure["mean"],
        "force_structure_max_gmd": gmd_structure["max"],
        "force_structure_q95_gmd": gmd_structure["q95"],
    }


def _comparison_pairs(
    uncertainty: Mapping[str, Tensor], errors: Mapping[str, Tensor]
) -> list[tuple[str, Tensor, Tensor]]:
    pairs: list[tuple[str, Tensor, Tensor]] = []
    for suffix in ("std", "gmd"):
        for prefix in ("energy_total", "energy_per_atom"):
            pairs.append((f"{prefix}_{suffix}", uncertainty[f"{prefix}_{suffix}"], errors[prefix]))
        for prefix, error_name in (
            ("force_component", "force_component"),
            ("force_atom_vector", "force_atom_vector"),
            ("force_structure_mean", "force_structure_mean"),
            ("force_structure_max", "force_structure_max"),
            ("force_structure_q95", "force_structure_q95"),
        ):
            pairs.append(
                (
                    f"{prefix}_{suffix}",
                    uncertainty[f"{prefix}_{suffix}"],
                    errors[error_name],
                )
            )
    return pairs


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _evaluate_branch(
    *,
    branch: str,
    root: Path,
    prediction: Mapping[str, Any],
    energy_weights: Tensor | None,
    force_weights: Tensor | None,
    coverages: Sequence[float],
    force_quantile: float,
) -> tuple[dict[str, Any], list[str]]:
    energy = weighted_mean(prediction["energy_members"], energy_weights)
    forces = weighted_mean(prediction["forces_members"], force_weights)
    ensemble = {
        "schema_version": "fge.ensemble.v1",
        "branch": branch,
        "energy": energy.to(dtype=torch.float64, device="cpu"),
        "forces": forces.to(dtype=torch.float64, device="cpu"),
        "energy_weights": (
            torch.full((prediction["energy_members"].shape[0],), 1.0 / prediction["energy_members"].shape[0], dtype=torch.float64)
            if energy_weights is None
            else energy_weights.to(dtype=torch.float64, device="cpu")
        ),
        "force_weights": (
            torch.full((prediction["forces_members"].shape[0],), 1.0 / prediction["forces_members"].shape[0], dtype=torch.float64)
            if force_weights is None
            else force_weights.to(dtype=torch.float64, device="cpu")
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
    metrics = _metric_summary(errors)
    correlation_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name, uncertainty_values, error_values in _comparison_pairs(uncertainty, errors):
        correlation = compute_correlations(
            uncertainty_values.reshape(-1), error_values.reshape(-1)
        )
        correlation_rows.append(
            {"metric": name, "pearson": correlation["pearson"], "spearman": correlation["spearman"]}
        )
        warnings.extend(f"{name}: {warning}" for warning in correlation["warnings"])
        for row in compute_risk_coverage(
            uncertainty_values.reshape(-1), error_values.reshape(-1), coverages
        ):
            risk_rows.append({"metric": name, **row})

    branch_root = root / branch
    atomic_torch_save(branch_root / "ensemble.pt", ensemble)
    atomic_torch_save(
        branch_root / "uncertainty.pt",
        {"schema_version": "fge.uncertainty.v1", "branch": branch, **uncertainty},
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
    return {"metrics": metrics, "warnings": warnings}, warnings


def evaluate_prediction(config: Any, run_dir: Path) -> dict[str, Any]:
    """Evaluate existing tensors without loading a model or reading extxyz data."""
    run_dir = Path(run_dir)
    prediction = _load_prediction(run_dir / "prediction" / "test_raw.pt")
    training = _load_training(
        run_dir / "training" / "manifest.json", prediction["member_ids"]
    )
    ensemble_config = _section(config, "ensemble")
    evaluation_config = _section(config, "evaluation")
    members = training["members"]
    try:
        energy_rmse = torch.tensor(
            [member["raw_metrics"]["energy_rmse"] for member in members],
            dtype=torch.float64,
        )
        force_rmse = torch.tensor(
            [member["raw_metrics"]["forces_rmse"] for member in members],
            dtype=torch.float64,
        )
        base_energy = float(training["base_model_metrics"]["energy_rmse"])
        base_force = float(training["base_model_metrics"]["forces_rmse"])
        energy_eps = float(ensemble_config["eps_energy_ratio"])
        force_eps = float(ensemble_config["eps_force_ratio"])
        coverages = tuple(float(value) for value in evaluation_config["risk_coverages"])
        force_quantile = float(evaluation_config["force_structure_quantile"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HardFailure("evaluation inputs are incomplete") from exc
    energy_weights = validation_error_weights(energy_rmse, base_energy, energy_eps)
    force_weights = validation_error_weights(force_rmse, base_force, force_eps)

    evaluation_root = run_dir / "evaluation"
    equal, equal_warnings = _evaluate_branch(
        branch="equal_weight",
        root=evaluation_root,
        prediction=prediction,
        energy_weights=None,
        force_weights=None,
        coverages=coverages,
        force_quantile=force_quantile,
    )
    weighted, weighted_warnings = _evaluate_branch(
        branch="validation_weighted",
        root=evaluation_root,
        prediction=prediction,
        energy_weights=energy_weights,
        force_weights=force_weights,
        coverages=coverages,
        force_quantile=force_quantile,
    )
    warnings = equal_warnings + weighted_warnings
    report = (
        "# FGE 评估报告\n\n"
        "本报告由 canonical prediction 直接计算，未加载模型或重新预测。\n\n"
        f"- 成员数：{len(prediction['member_ids'])}\n"
        f"- 结构数：{prediction['energy_reference'].numel()}\n"
        f"- 原子数：{prediction['forces_reference'].shape[0]}\n"
        f"- 警告数：{len(warnings)}\n"
    )
    _atomic_text(evaluation_root / "report.md", report)
    return {"equal_weight": equal, "validation_weighted": weighted, "warnings": warnings}


def _stress_outputs(
    prediction: Mapping[str, Any], stress_weights: Tensor | None
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Build stress ensemble, component STD, and component residual tensors."""
    members = prediction.get("stress_members")
    reference = prediction.get("stress_reference")
    if (
        not isinstance(members, Tensor)
        or members.ndim != 4
        or members.shape[2:] != (3, 3)
    ):
        raise HardFailure("stress_members must have shape [K, S, 3, 3]")
    if (
        not isinstance(reference, Tensor)
        or tuple(reference.shape) != tuple(members.shape[1:])
    ):
        raise HardFailure("stress_reference must have shape [S, 3, 3]")
    if members.dtype != torch.float64 or reference.dtype != torch.float64:
        raise HardFailure("stress tensors must use float64")
    if (
        not torch.isfinite(members).all().item()
        or not torch.isfinite(reference).all().item()
    ):
        raise HardFailure("stress tensors contain NaN or Inf")
    member_count = int(members.shape[0])
    stored = (
        torch.full((member_count,), 1.0 / member_count, dtype=torch.float64)
        if stress_weights is None
        else stress_weights.detach().to(device="cpu", dtype=torch.float64)
    )
    stress = weighted_mean(members, stress_weights).to(
        device="cpu", dtype=torch.float64
    )
    return (
        {"stress": stress, "stress_weights": stored},
        {"stress_component_std": unbiased_std(members, stress_weights)},
        compute_stress_errors(stress, reference),
    )