"""Strict, read-only loading of validated FGE plotting data."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .artifacts import sha256_file
from .errors import HardFailure


BRANCHES = ("equal_weight", "validation_weighted")


@dataclass(frozen=True)
class PlotBranch:
    energy_reference: Tensor
    energy_prediction: Tensor
    energy_residual: Tensor
    energy_uncertainty: Tensor
    force_reference: Tensor
    force_prediction: Tensor
    force_residual: Tensor
    force_uncertainty: Tensor
    metrics: dict[str, Any]
    correlations: tuple[dict[str, Any], ...]
    risk_coverage: tuple[dict[str, Any], ...]
    stress_reference: Tensor | None = None
    stress_prediction: Tensor | None = None
    stress_residual: Tensor | None = None
    stress_uncertainty: Tensor | None = None

    @property
    def has_stress(self) -> bool:
        values = (
            self.stress_reference,
            self.stress_prediction,
            self.stress_residual,
            self.stress_uncertainty,
        )
        if any(value is None for value in values):
            if not all(value is None for value in values):
                raise HardFailure("stress plotting fields are partially present")
            return False
        return True


@dataclass(frozen=True)
class PlotRun:
    name: str
    root: Path
    branches: dict[str, PlotBranch]

    @property
    def has_stress(self) -> bool:
        modes = {branch.has_stress for branch in self.branches.values()}
        if len(modes) != 1:
            raise HardFailure("plot branches expose inconsistent observables")
        return modes == {True}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HardFailure(f"cannot load required JSON: {path}") from exc
    if not isinstance(value, dict):
        raise HardFailure(f"required JSON is not an object: {path}")
    return value


def _tensor_file(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise HardFailure(f"cannot load required tensor file: {path}") from exc
    if not isinstance(value, dict):
        raise HardFailure(f"tensor file is not a mapping: {path}")
    return value


def _finite_tensor(payload: dict[str, Any], field: str, ndim: int) -> Tensor:
    value = payload.get(field)
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise HardFailure(f"field {field!r} has invalid shape")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(result).all().item():
        raise HardFailure(f"field {field!r} contains NaN or Inf")
    return result


def _csv_rows(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return tuple(dict(row) for row in csv.DictReader(stream))
    except Exception as exc:
        raise HardFailure(f"cannot load required CSV: {path}") from exc


def _require_same_shape(label: str, *values: Tensor) -> None:
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) != 1:
        raise HardFailure(f"{label} arrays have mismatched shapes")


def load_plot_run(root: Path) -> PlotRun:
    """Load one validated canonical result without models, data, or prediction."""
    root = Path(root)
    for marker_name in ("validation.json", "result_manifest.json"):
        marker = _json(root / marker_name)
        if marker.get("status") != "PASS":
            raise HardFailure(f"{marker_name} must have PASS status")

    prediction = _tensor_file(root / "prediction" / "test_raw.pt")
    if prediction.get("schema_version") != "fge.prediction.v1":
        raise HardFailure("canonical prediction schema is invalid")
    energy_reference_total = _finite_tensor(prediction, "energy_reference", 1)
    force_reference_matrix = _finite_tensor(prediction, "forces_reference", 2)
    n_atoms = _finite_tensor(prediction, "n_atoms", 1)
    if (n_atoms <= 0).any().item():
        raise HardFailure("n_atoms must be positive")
    _require_same_shape("energy structure", energy_reference_total, n_atoms)
    if force_reference_matrix.shape[1:] != (3,):
        raise HardFailure("forces_reference must have shape [A, 3]")

    branches: dict[str, PlotBranch] = {}
    for branch_name in BRANCHES:
        branch_root = root / "evaluation" / branch_name
        ensemble = _tensor_file(branch_root / "ensemble.pt")
        uncertainty = _tensor_file(branch_root / "uncertainty.pt")
        if ensemble.get("schema_version") != "fge.ensemble.v1" or ensemble.get("branch") != branch_name:
            raise HardFailure(f"invalid ensemble schema for {branch_name}")
        if uncertainty.get("schema_version") != "fge.uncertainty.v1" or uncertainty.get("branch") != branch_name:
            raise HardFailure(f"invalid uncertainty schema for {branch_name}")

        energy_prediction_total = _finite_tensor(ensemble, "energy", 1)
        energy_uncertainty = _finite_tensor(uncertainty, "energy_per_atom_std", 1)
        force_prediction_matrix = _finite_tensor(ensemble, "forces", 2)
        force_uncertainty_matrix = _finite_tensor(uncertainty, "force_component_std", 2)
        _require_same_shape(
            "energy", energy_reference_total, energy_prediction_total, energy_uncertainty
        )
        _require_same_shape(
            "force", force_reference_matrix, force_prediction_matrix, force_uncertainty_matrix
        )

        energy_reference = energy_reference_total / n_atoms
        energy_prediction = energy_prediction_total / n_atoms
        force_reference = force_reference_matrix.reshape(-1)
        force_prediction = force_prediction_matrix.reshape(-1)
        force_uncertainty = force_uncertainty_matrix.reshape(-1)
        branches[branch_name] = PlotBranch(
            energy_reference=energy_reference,
            energy_prediction=energy_prediction,
            energy_residual=(energy_prediction - energy_reference).abs(),
            energy_uncertainty=energy_uncertainty,
            force_reference=force_reference,
            force_prediction=force_prediction,
            force_residual=(force_prediction - force_reference).abs(),
            force_uncertainty=force_uncertainty,
            metrics=_json(branch_root / "metrics.json"),
            correlations=_csv_rows(branch_root / "correlations.csv"),
            risk_coverage=_csv_rows(branch_root / "risk_coverage.csv"),
        )
    return PlotRun(name=root.name, root=root, branches=branches)


def _dataset_branch(root: Path, branch_name: str, has_stress: bool) -> PlotBranch:
    branch_root = root / "evaluation" / branch_name
    index = _tensor_file(branch_root / "ensemble.pt")
    if (
        index.get("schema_version") != "fge.derived-ensemble-index.v1"
        or index.get("branch") != branch_name
    ):
        raise HardFailure(f"invalid derived ensemble index for {branch_name}")
    rows = index.get("shards")
    if not isinstance(rows, list) or not rows:
        raise HardFailure(f"derived evaluation has no shards for {branch_name}")

    shards: list[dict[str, Any]] = []
    for expected_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("index") != expected_index:
            raise HardFailure("derived evaluation shard indices are not contiguous")
        relative = row.get("path")
        expected_hash = row.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise HardFailure("derived evaluation shard reference is invalid")
        path = branch_root / relative
        if sha256_file(path) != expected_hash:
            raise HardFailure("derived evaluation shard SHA-256 mismatch")
        payload = _tensor_file(path)
        if (
            payload.get("schema_version") != "fge.derived-evaluation-shard.v1"
            or payload.get("branch") != branch_name
            or payload.get("shard_index") != expected_index
        ):
            raise HardFailure("derived evaluation shard metadata is invalid")
        shards.append(payload)

    def concatenate(group: str, field: str, ndim: int) -> Tensor:
        values = []
        for shard in shards:
            payload = shard.get(group)
            if not isinstance(payload, dict):
                raise HardFailure(f"derived shard group {group!r} is invalid")
            values.append(_finite_tensor(payload, field, ndim))
        return torch.cat(values, dim=0)

    energy_reference_total = concatenate("references", "energy", 1)
    energy_prediction_total = concatenate("ensemble", "energy", 1)
    n_atoms = concatenate("references", "n_atoms", 1)
    if (n_atoms <= 0).any().item():
        raise HardFailure("n_atoms must be positive")
    _require_same_shape(
        "energy", energy_reference_total, energy_prediction_total, n_atoms
    )
    force_reference_matrix = concatenate("references", "forces", 2)
    force_prediction_matrix = concatenate("ensemble", "forces", 2)
    force_uncertainty_matrix = concatenate(
        "uncertainty", "force_component_std", 2
    )
    _require_same_shape(
        "force",
        force_reference_matrix,
        force_prediction_matrix,
        force_uncertainty_matrix,
    )
    energy_uncertainty = concatenate(
        "uncertainty", "energy_per_atom_std", 1
    )
    energy_reference = energy_reference_total / n_atoms
    energy_prediction = energy_prediction_total / n_atoms
    _require_same_shape(
        "energy", energy_reference, energy_prediction, energy_uncertainty
    )

    stress_fields: dict[str, Tensor | None] = {
        "stress_reference": None,
        "stress_prediction": None,
        "stress_residual": None,
        "stress_uncertainty": None,
    }
    if has_stress:
        stress_reference = concatenate("references", "stress", 3).reshape(-1)
        stress_prediction = concatenate("ensemble", "stress", 3).reshape(-1)
        stress_uncertainty = concatenate(
            "uncertainty", "stress_component_std", 3
        ).reshape(-1)
        _require_same_shape(
            "stress",
            stress_reference,
            stress_prediction,
            stress_uncertainty,
        )
        stress_fields = {
            "stress_reference": stress_reference,
            "stress_prediction": stress_prediction,
            "stress_residual": (stress_prediction - stress_reference).abs(),
            "stress_uncertainty": stress_uncertainty,
        }

    force_reference = force_reference_matrix.reshape(-1)
    force_prediction = force_prediction_matrix.reshape(-1)
    force_uncertainty = force_uncertainty_matrix.reshape(-1)
    return PlotBranch(
        energy_reference=energy_reference,
        energy_prediction=energy_prediction,
        energy_residual=(energy_prediction - energy_reference).abs(),
        energy_uncertainty=energy_uncertainty,
        force_reference=force_reference,
        force_prediction=force_prediction,
        force_residual=(force_prediction - force_reference).abs(),
        force_uncertainty=force_uncertainty,
        metrics=_json(branch_root / "metrics.json"),
        correlations=_csv_rows(branch_root / "correlations.csv"),
        risk_coverage=_csv_rows(branch_root / "risk_coverage.csv"),
        **stress_fields,
    )


def load_dataset_plot_run(root: Path) -> PlotRun:
    """Load one dataset-scoped derived evaluation by concatenating shard tensors."""
    root = Path(root)
    audit = _json(root / "evaluation" / "audit.json")
    if (
        audit.get("schema_version") != "fge.derived-evaluation.v1"
        or audit.get("status") != "PASS"
    ):
        raise HardFailure("derived evaluation audit must have PASS status")
    observables = audit.get("observables")
    if observables not in (
        ["energy", "forces"],
        ["energy", "forces", "stress"],
    ):
        raise HardFailure("derived evaluation observables are invalid")
    has_stress = observables == ["energy", "forces", "stress"]
    branches = {
        branch_name: _dataset_branch(root, branch_name, has_stress)
        for branch_name in BRANCHES
    }
    run = PlotRun(name=root.parent.name, root=root, branches=branches)
    if run.has_stress != has_stress:
        raise HardFailure("derived plot branches do not match audit observables")
    return run