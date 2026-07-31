"""Deterministic six-path LLPR calibration with crash-safe resume."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Mapping

import torch
from torch import Tensor

from .artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    atomic_json_dump,
    atomic_torch_save,
    canonical_json,
    load_torch_artifact,
    require_identity,
)
from .checkpoint import CheckpointIdentity, load_checkpoint
from .config import LLPRConfig
from .curvature import run_root
from .data import DatasetHandle, build_dataset, iter_samples
from .observables import compute_structure_jacobians
from .readout import ReadoutLayout, discover_readout_layout
from .ridge import RidgeRecord, choose_ridge


_VARIANTS = ("he", "hf", "hef")
_TARGETS = ("energy", "forces")


class CholeskyQuadraticForm:
    """Evaluate rows of ``g (H + ridge I)^-1 g.T`` via float64 Cholesky."""

    def __init__(self, curvature: Tensor, ridge: float) -> None:
        matrix = curvature.detach().to(device="cpu", dtype=torch.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("curvature must be a square matrix")
        if not torch.isfinite(matrix).all():
            raise ValueError("curvature must be finite")
        ridge_value = float(ridge)
        if not math.isfinite(ridge_value) or ridge_value < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        regularized = matrix + ridge_value * torch.eye(
            matrix.shape[0], dtype=torch.float64, device="cpu"
        )
        self.factor = torch.linalg.cholesky(regularized)
        self.size = matrix.shape[0]
        self.ridge = ridge_value

    def q(self, gradients: Tensor) -> Tensor:
        """Return one quadratic form per gradient row."""
        rows = gradients.detach().to(device="cpu", dtype=torch.float64)
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        if rows.ndim != 2 or rows.shape[1] != self.size:
            raise ValueError(
                f"gradients must have shape (rows, {self.size})"
            )
        if not torch.isfinite(rows).all():
            raise ValueError("gradients must be finite")
        transformed = torch.linalg.solve_triangular(
            self.factor, rows.T, upper=False
        )
        return torch.sum(transformed.square(), dim=0)


@dataclass(frozen=True)
class CalibrationRecord:
    variant: Literal["he", "hf", "hef"]
    target: Literal["energy", "forces"]
    ridge_mode: Literal["fixed", "condition_number"]
    ridge: float
    alpha: float
    rows: int
    mean_residual_squared_over_q: float


def _squared_residual_over_q(
    residuals: Tensor,
    q: Tensor,
    min_q: float,
) -> Tensor:
    residual_values = residuals.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    q_values = q.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if residual_values.numel() == 0:
        raise ValueError("residuals and q must not be empty")
    if residual_values.shape != q_values.shape:
        raise ValueError("residuals and q must have the same number of rows")
    if not torch.isfinite(residual_values).all():
        raise ValueError("residuals must be finite")
    if not torch.isfinite(q_values).all() or torch.any(q_values <= 0.0):
        raise ValueError("q must contain only finite positive values")
    floor = float(min_q)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("min_q must be finite and positive")
    return residual_values.square() / torch.clamp(q_values, min=floor)


def calibrate_alpha(
    residuals: Tensor,
    q: Tensor,
    min_q: float = 1.0e-30,
) -> float:
    """Return ``sqrt(mean(residual**2 / max(q, min_q)))``."""
    ratios = _squared_residual_over_q(residuals, q, min_q)
    return math.sqrt(float(ratios.mean()))


def _checkpoint_metadata(identity: CheckpointIdentity) -> dict[str, Any]:
    return {
        "sha256": identity.sha256,
        "model_class": identity.model_class,
        "heads": list(identity.heads),
        "selected_head": identity.selected_head,
        "r_max": identity.r_max,
        "atomic_numbers": list(identity.atomic_numbers),
        "dtype": str(identity.dtype),
    }


def _dataset_metadata(dataset: DatasetHandle) -> dict[str, Any]:
    return {
        "sha256": dataset.sha256,
        "identity": dataset.identity,
        "size": dataset.size,
        "atomic_numbers": list(dataset.atomic_numbers),
        "r_max": dataset.r_max,
        "head": dataset.head,
    }


def _ridge_identity(config: LLPRConfig, records: Mapping[str, RidgeRecord]) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "mode": config.ridge.mode,
        "selected": {variant: records[variant].value for variant in _VARIANTS},
    }
    if config.ridge.mode == "fixed":
        identity["value"] = config.ridge.value
    else:
        identity["max_condition_number"] = config.ridge.max_condition_number
    return identity


def _calibration_identity(
    config: LLPRConfig,
    checkpoint: CheckpointIdentity,
    dataset: DatasetHandle,
    curvature_identity: Mapping[str, Any],
    ridges: Mapping[str, RidgeRecord],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "checkpoint": _checkpoint_metadata(checkpoint),
        "curvature": dict(curvature_identity),
        "calibration": _dataset_metadata(dataset),
        "ridge": _ridge_identity(config, ridges),
        "min_q": config.curvature.min_q,
        "limits": {
            "max_structures": config.runtime.max_structures,
            "max_force_components_per_structure": (
                config.runtime.max_force_components_per_structure
            ),
        },
    }


def _load_curvature(
    path: Path,
    checkpoint: CheckpointIdentity,
    layout: ReadoutLayout,
) -> tuple[Mapping[str, Any], dict[str, Tensor]]:
    if not path.exists():
        raise ValueError("completed curvature artifact is missing")
    artifact = load_torch_artifact(path)
    if not isinstance(artifact, Mapping):
        raise ValueError("curvature artifact must be a mapping")
    if artifact.get("status") != "complete":
        raise ValueError("curvature artifact is not complete")
    identity = artifact.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("curvature artifact identity must be a mapping")
    checkpoint_identity = identity.get("checkpoint")
    if not isinstance(checkpoint_identity, Mapping) or (
        checkpoint_identity.get("sha256") != checkpoint.sha256
    ):
        raise ValueError("curvature checkpoint identity mismatch")
    readout_identity = identity.get("readout")
    if not isinstance(readout_identity, Mapping):
        raise ValueError("curvature readout identity must be a mapping")
    require_identity(readout_identity, layout.metadata())

    source_variants = artifact.get("variants")
    if not isinstance(source_variants, Mapping) or set(source_variants) != set(_VARIANTS):
        raise ValueError("curvature artifact must contain he, hf, and hef")
    variants: dict[str, Tensor] = {}
    for variant in _VARIANTS:
        matrix = source_variants[variant]
        if (
            not isinstance(matrix, Tensor)
            or matrix.device.type != "cpu"
            or matrix.dtype != torch.float64
            or matrix.shape != (layout.size, layout.size)
            or not torch.isfinite(matrix).all()
        ):
            raise ValueError(
                f"curvature {variant} must be finite CPU float64 with shape "
                f"({layout.size}, {layout.size})"
            )
        variants[variant] = matrix
    return identity, variants


def _empty_accumulators() -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        variant: {
            target: {"sum": 0.0, "rows": 0}
            for target in _TARGETS
        }
        for variant in _VARIANTS
    }


def _new_progress(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": dict(identity),
        "status": "in_progress",
        "next_index": 0,
        "structures": 0,
        "accumulators": _empty_accumulators(),
    }


def _validate_progress(progress: Mapping[str, Any]) -> None:
    if progress.get("status") not in ("in_progress", "complete"):
        raise ValueError("calibration progress has an invalid status")
    for field in ("next_index", "structures"):
        value = progress.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"calibration progress {field} must be non-negative")
    accumulators = progress.get("accumulators")
    if not isinstance(accumulators, Mapping) or set(accumulators) != set(_VARIANTS):
        raise ValueError("calibration progress has invalid variants")
    for variant in _VARIANTS:
        targets = accumulators[variant]
        if not isinstance(targets, Mapping) or set(targets) != set(_TARGETS):
            raise ValueError("calibration progress has invalid targets")
        for target in _TARGETS:
            accumulator = targets[target]
            if not isinstance(accumulator, Mapping):
                raise ValueError("calibration progress accumulator must be a mapping")
            total = accumulator.get("sum")
            rows = accumulator.get("rows")
            if (
                isinstance(total, bool)
                or not isinstance(total, (float, int))
                or not math.isfinite(float(total))
                or float(total) < 0.0
            ):
                raise ValueError("calibration progress sum must be finite and non-negative")
            if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
                raise ValueError("calibration progress rows must be non-negative")


def _matching_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return canonical_json(actual) == canonical_json(expected)


def _validate_complete_artifacts(
    artifact_path: Path,
    csv_path: Path,
    diagnostics_path: Path,
    identity: Mapping[str, Any],
) -> None:
    for path in (artifact_path, csv_path, diagnostics_path):
        if not path.exists():
            raise ValueError(f"complete calibration is missing {path.name}")
    artifact = load_torch_artifact(artifact_path)
    if not isinstance(artifact, Mapping):
        raise ValueError("calibrations artifact must be a mapping")
    require_identity(artifact.get("identity", {}), identity)
    if artifact.get("status") != "complete":
        raise ValueError("calibrations artifact is not complete")
    records = artifact.get("records")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("calibrations artifact must contain exactly six records")


def _atomic_csv_dump(path: Path, records: list[CalibrationRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CalibrationRecord.__dataclass_fields__)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _ridge_diagnostics(
    identity: Mapping[str, Any],
    variants: Mapping[str, Tensor],
    ridges: Mapping[str, RidgeRecord],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for variant in _VARIANTS:
        eigenvalues = torch.linalg.eigvalsh(variants[variant])
        minimum = float(eigenvalues.min())
        maximum = float(eigenvalues.max())
        regularized_minimum = minimum + ridges[variant].value
        regularized_maximum = maximum + ridges[variant].value
        condition_number = (
            regularized_maximum / regularized_minimum
            if regularized_minimum > 0.0
            else None
        )
        diagnostics[variant] = {
            "ridge_mode": ridges[variant].mode,
            "ridge": ridges[variant].value,
            "eigenvalue_min": minimum,
            "eigenvalue_max": maximum,
            "regularized_condition_number": condition_number,
        }
    return {
        "identity": dict(identity),
        "status": "complete",
        "variants": diagnostics,
    }


def _records_from_progress(
    progress: Mapping[str, Any],
    ridges: Mapping[str, RidgeRecord],
) -> list[CalibrationRecord]:
    records = []
    accumulators = progress["accumulators"]
    for variant in _VARIANTS:
        for target in _TARGETS:
            accumulator = accumulators[variant][target]
            rows = int(accumulator["rows"])
            if rows == 0:
                raise ValueError(f"calibration {variant}/{target} has no rows")
            mean = float(accumulator["sum"]) / rows
            records.append(
                CalibrationRecord(
                    variant=variant,
                    target=target,
                    ridge_mode=ridges[variant].mode,
                    ridge=ridges[variant].value,
                    alpha=math.sqrt(mean),
                    rows=rows,
                    mean_residual_squared_over_q=mean,
                )
            )
    return records


def run_calibrate(config: LLPRConfig) -> Path:
    """Calibrate energy and force scales for all three curvature variants."""
    loaded = load_checkpoint(config.checkpoint, torch.device("cpu"))
    layout = discover_readout_layout(loaded.model)
    dataset = build_dataset(
        config.calibration.path,
        config.calibration.expected_sha256,
        loaded.identity.atomic_numbers,
        loaded.identity.r_max,
        loaded.identity.selected_head,
    )
    curvature_path = (
        run_root(config, loaded.identity.sha256) / "curvature" / "base_curvature.pt"
    )
    curvature_identity, variants = _load_curvature(
        curvature_path, loaded.identity, layout
    )
    ridges = {
        variant: choose_ridge(variants[variant], config.ridge)
        for variant in _VARIANTS
    }
    identity = _calibration_identity(
        config, loaded.identity, dataset, curvature_identity, ridges
    )

    calibration_dir = (
        run_root(config, loaded.identity.sha256)
        / "calibration"
        / "deterministic"
    )
    progress_path = calibration_dir / "progress.pt"
    artifact_path = calibration_dir / "calibrations.pt"
    csv_path = calibration_dir / "calibrations.csv"
    diagnostics_path = calibration_dir / "ridge_diagnostics.json"

    progress: dict[str, Any] | None = None
    if progress_path.exists():
        candidate = load_torch_artifact(progress_path)
        if not isinstance(candidate, Mapping):
            raise ValueError("calibration progress must be a mapping")
        candidate_identity = candidate.get("identity", {})
        if (
            candidate.get("status") == "complete"
            and isinstance(candidate_identity, Mapping)
            and _matching_identity(candidate_identity, identity)
        ):
            _validate_progress(candidate)
            _validate_complete_artifacts(
                artifact_path, csv_path, diagnostics_path, identity
            )
            return artifact_path
        if config.runtime.resume:
            require_identity(candidate_identity, identity)
            _validate_progress(candidate)
            progress = dict(candidate)

    solvers = {
        variant: CholeskyQuadraticForm(
            variants[variant], ridge=ridges[variant].value
        )
        for variant in _VARIANTS
    }
    if progress is None:
        progress = _new_progress(identity)
        atomic_torch_save(progress_path, progress)

    target_structures = dataset.size
    if config.runtime.max_structures is not None:
        target_structures = min(target_structures, config.runtime.max_structures)
    if progress["next_index"] > target_structures:
        raise ValueError("calibration progress next_index exceeds the calibration limit")

    requested_device = torch.device(config.runtime.device)
    loaded.model.to(requested_device)
    remaining = target_structures - progress["next_index"]
    if remaining > 0:
        samples = iter_samples(
            dataset,
            device=requested_device,
            dtype=loaded.identity.dtype,
            start_index=progress["next_index"],
            max_structures=remaining,
        )
        for sample in samples:
            jacobians = compute_structure_jacobians(
                model=loaded.model,
                batch=sample.batch,
                layout=layout,
                force_component_chunk_size=config.runtime.force_component_chunk_size,
                max_force_components=(
                    config.runtime.max_force_components_per_structure
                ),
            )
            energy_residual = torch.tensor(
                [
                    float(sample.reference_energy_per_atom.detach().cpu())
                    - jacobians.energy_per_atom
                ],
                dtype=torch.float64,
            )
            indices = jacobians.force_indices.detach().to(device="cpu", dtype=torch.long)
            reference_forces = sample.reference_forces.detach().to(
                device="cpu", dtype=torch.float64
            ).reshape(-1)
            predicted_forces = jacobians.forces.detach().to(
                device="cpu", dtype=torch.float64
            ).reshape(-1)
            if (
                indices.ndim != 1
                or jacobians.g_forces.shape[0] != indices.numel()
                or (indices.numel() and int(indices.max()) >= reference_forces.numel())
                or (indices.numel() and int(indices.max()) >= predicted_forces.numel())
            ):
                raise ValueError("force Jacobian rows do not match force indices")
            force_residuals = reference_forces[indices] - predicted_forces[indices]

            for variant in _VARIANTS:
                energy_ratios = _squared_residual_over_q(
                    energy_residual,
                    solvers[variant].q(jacobians.g_energy),
                    config.curvature.min_q,
                )
                force_ratios = _squared_residual_over_q(
                    force_residuals,
                    solvers[variant].q(jacobians.g_forces),
                    config.curvature.min_q,
                )
                energy_accumulator = progress["accumulators"][variant]["energy"]
                energy_accumulator["sum"] += float(energy_ratios.sum())
                energy_accumulator["rows"] += int(energy_ratios.numel())
                force_accumulator = progress["accumulators"][variant]["forces"]
                force_accumulator["sum"] += float(force_ratios.sum())
                force_accumulator["rows"] += int(force_ratios.numel())

            progress["next_index"] = sample.index + 1
            progress["structures"] += 1
            if progress["structures"] % config.runtime.save_every_structures == 0:
                atomic_torch_save(progress_path, progress)

    records = _records_from_progress(progress, ridges)
    artifact = {
        "identity": identity,
        "status": "complete",
        "records": [asdict(record) for record in records],
    }
    atomic_torch_save(artifact_path, artifact)
    _atomic_csv_dump(csv_path, records)
    atomic_json_dump(
        diagnostics_path,
        _ridge_diagnostics(identity, variants, ridges),
    )
    progress["status"] = "complete"
    atomic_torch_save(progress_path, progress)
    return artifact_path
