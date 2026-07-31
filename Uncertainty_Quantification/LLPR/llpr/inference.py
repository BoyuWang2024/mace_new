"""Six-path LLPR evaluation into resumable canonical CSV artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

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
    sha256_file,
)
from .calibration import CholeskyQuadraticForm
from .checkpoint import CheckpointIdentity, load_checkpoint
from .config import LLPRConfig
from .curvature import run_root
from .data import DatasetHandle, build_dataset, iter_samples
from .observables import compute_structure_jacobians
from .readout import ReadoutLayout, discover_readout_layout


ENERGY_FIELDS = [
    "structure_id",
    "num_atoms",
    "reference",
    "prediction",
    "residual",
    "q",
    "variance",
    "std",
    "variant",
    "target",
]
FORCE_FIELDS = [
    "structure_id",
    "num_atoms",
    "atom_index",
    "direction",
    "reference",
    "prediction",
    "residual",
    "q",
    "variance",
    "std",
    "variant",
    "target",
]
FORCE_STRUCTURE_FIELDS = [
    "structure_id",
    "num_atoms",
    "components",
    "mae",
    "rmse",
    "mean_q",
    "mean_variance",
    "variant",
    "target",
]

_VARIANTS = ("he", "hf", "hef")
_CSV_SPECS = {
    "energy.csv": ENERGY_FIELDS,
    "force_components.csv": FORCE_FIELDS,
    "force_structure.csv": FORCE_STRUCTURE_FIELDS,
}


def _csv_bytes(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    for row in rows:
        missing = set(fieldnames) - set(row)
        if missing:
            raise ValueError(f"CSV row is missing fields: {sorted(missing)}")
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


class TransactionalCSV:
    """Append CSV rows while exposing durable byte-offset commit points."""

    def __init__(
        self,
        path: Path,
        fieldnames: Sequence[str],
        *,
        reset: bool = False,
    ) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        if not self.fieldnames or len(set(self.fieldnames)) != len(self.fieldnames):
            raise ValueError("CSV fieldnames must be non-empty and unique")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w+b" if reset or not self.path.exists() else "r+b"
        self._handle = self.path.open(mode, buffering=0)
        header = _csv_bytes(self.fieldnames, [])
        header_buffer = io.StringIO(newline="")
        csv.writer(header_buffer, lineterminator="\n").writerow(self.fieldnames)
        header = header_buffer.getvalue().encode("utf-8")
        self.header_byte_offset = len(header)
        if mode == "w+b":
            self._handle.write(header)
            self._durable_flush()
        else:
            self._handle.seek(0)
            actual_header = self._handle.read(self.header_byte_offset)
            if actual_header != header:
                self.close()
                raise ValueError(f"CSV header mismatch for {self.path}")
            self._handle.seek(0, os.SEEK_END)

    @property
    def byte_offset(self) -> int:
        """Return the current byte position after all appended data."""
        return int(self._handle.tell())

    def append(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Append rows without declaring them committed."""
        self._handle.write(_csv_bytes(self.fieldnames, rows))

    def _durable_flush(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def flush(self) -> int:
        """Make current bytes durable and return their byte offset."""
        self._durable_flush()
        return self.byte_offset

    def restore(self, byte_offset: int) -> None:
        """Discard every byte after a previously committed offset."""
        if isinstance(byte_offset, bool) or not isinstance(byte_offset, int):
            raise ValueError("CSV byte offset must be an integer")
        self._handle.seek(0, os.SEEK_END)
        size = self.byte_offset
        if byte_offset < self.header_byte_offset or byte_offset > size:
            raise ValueError(
                f"CSV byte offset {byte_offset} is outside "
                f"[{self.header_byte_offset}, {size}] for {self.path}"
            )
        self._handle.seek(byte_offset)
        self._handle.truncate()
        self._durable_flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> TransactionalCSV:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


def validate_q(
    q: Tensor,
    *,
    structure_index: int,
    target: Literal["energy", "forces"],
) -> Tensor:
    """Validate quadratic forms and floor only positive values below 1e-30."""
    values = q.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        raise ValueError(f"q at structure {structure_index} for {target} is empty")
    invalid = ~torch.isfinite(values) | (values <= 0.0)
    if torch.any(invalid):
        positions = torch.nonzero(invalid, as_tuple=False).reshape(-1).tolist()
        raise ValueError(
            f"q at structure {structure_index} for {target} must contain only "
            f"finite positive values; invalid rows={positions}"
        )
    return torch.clamp(values, min=1.0e-30)


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


def _load_curvature(
    path: Path,
    checkpoint: CheckpointIdentity,
    layout: ReadoutLayout,
) -> tuple[Mapping[str, Any], dict[str, Tensor]]:
    if not path.exists():
        raise ValueError("completed curvature artifact is missing")
    artifact = load_torch_artifact(path)
    if not isinstance(artifact, Mapping) or artifact.get("status") != "complete":
        raise ValueError("curvature artifact must be a complete mapping")
    identity = artifact.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("curvature artifact identity must be a mapping")
    checkpoint_identity = identity.get("checkpoint")
    if not isinstance(checkpoint_identity, Mapping) or (
        checkpoint_identity.get("sha256") != checkpoint.sha256
    ):
        raise ValueError("curvature checkpoint identity mismatch")
    require_identity(identity.get("readout", {}), layout.metadata())
    source_variants = artifact.get("variants")
    if not isinstance(source_variants, Mapping) or set(source_variants) != set(_VARIANTS):
        raise ValueError("curvature artifact must contain he, hf, and hef")
    variants: dict[str, Tensor] = {}
    for variant in _VARIANTS:
        matrix = source_variants[variant]
        if (
            not isinstance(matrix, Tensor)
            or matrix.shape != (layout.size, layout.size)
            or matrix.dtype != torch.float64
            or matrix.device.type != "cpu"
            or not torch.isfinite(matrix).all()
        ):
            raise ValueError(
                f"curvature {variant} must be finite CPU float64 with shape "
                f"({layout.size}, {layout.size})"
            )
        variants[variant] = matrix
    return identity, variants


def _load_calibrations(
    path: Path,
    checkpoint: CheckpointIdentity,
    curvature_identity: Mapping[str, Any],
    config: LLPRConfig,
) -> tuple[Mapping[str, Any], dict[str, dict[str, Any]]]:
    if not path.exists():
        raise ValueError("completed calibration artifact is missing")
    artifact = load_torch_artifact(path)
    if not isinstance(artifact, Mapping) or artifact.get("status") != "complete":
        raise ValueError("calibration artifact must be a complete mapping")
    identity = artifact.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("calibration artifact identity must be a mapping")
    checkpoint_identity = identity.get("checkpoint")
    if not isinstance(checkpoint_identity, Mapping) or (
        checkpoint_identity.get("sha256") != checkpoint.sha256
    ):
        raise ValueError("calibration checkpoint identity mismatch")
    require_identity(identity.get("curvature", {}), curvature_identity)
    if identity.get("min_q") != config.curvature.min_q:
        raise ValueError("calibration min_q does not match evaluation configuration")

    records = artifact.get("records")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("calibration artifact must contain exactly six records")
    by_variant: dict[str, dict[str, Any]] = {variant: {} for variant in _VARIANTS}
    for source in records:
        if not isinstance(source, Mapping):
            raise ValueError("calibration record must be a mapping")
        variant = source.get("variant")
        target = source.get("target")
        if variant not in _VARIANTS or target not in ("energy", "forces"):
            raise ValueError("calibration record has an invalid path")
        if target in by_variant[variant]:
            raise ValueError("calibration artifact contains a duplicate path")
        ridge_mode = source.get("ridge_mode")
        ridge = float(source.get("ridge"))
        alpha = float(source.get("alpha"))
        rows = source.get("rows")
        if ridge_mode != config.ridge.mode:
            raise ValueError("calibration ridge mode does not match evaluation config")
        if config.ridge.mode == "fixed" and ridge != config.ridge.value:
            raise ValueError("calibration ridge does not match evaluation config")
        if not math.isfinite(ridge) or ridge < 0.0:
            raise ValueError("calibration ridge must be finite and non-negative")
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("calibration alpha must be finite and non-negative")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError("calibration rows must be a positive integer")
        by_variant[variant][target] = {
            "ridge_mode": ridge_mode,
            "ridge": ridge,
            "alpha": alpha,
            "rows": rows,
        }
    for variant in _VARIANTS:
        if set(by_variant[variant]) != {"energy", "forces"}:
            raise ValueError(f"calibration {variant} must contain energy and forces")
        energy = by_variant[variant]["energy"]
        forces = by_variant[variant]["forces"]
        if (energy["ridge_mode"], energy["ridge"]) != (
            forces["ridge_mode"],
            forces["ridge"],
        ):
            raise ValueError(f"calibration {variant} must share one ridge")
    return identity, by_variant


def _load_cholesky_diagnostics(
    path: Path,
    calibration_identity: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        raise ValueError("calibration ridge diagnostics are missing")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("status") != "complete":
        raise ValueError("ridge diagnostics must be a complete mapping")
    require_identity(document.get("identity", {}), calibration_identity)
    variants = document.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(_VARIANTS):
        raise ValueError("ridge diagnostics must contain he, hf, and hef")
    result: dict[str, Mapping[str, Any]] = {}
    for variant in _VARIANTS:
        value = variants[variant]
        if not isinstance(value, Mapping):
            raise ValueError("ridge diagnostics variant must be a mapping")
        result[variant] = value
    return result


def _evaluation_identity(
    config: LLPRConfig,
    checkpoint: CheckpointIdentity,
    layout: ReadoutLayout,
    curvature_path: Path,
    curvature_identity: Mapping[str, Any],
    calibration_path: Path,
    diagnostics_path: Path,
    calibration_identity: Mapping[str, Any],
    calibration_records: Mapping[str, Mapping[str, Any]],
    dataset: DatasetHandle,
) -> dict[str, Any]:
    ridge_identity: dict[str, Any] = {"mode": config.ridge.mode}
    if config.ridge.mode == "fixed":
        ridge_identity["value"] = config.ridge.value
    else:
        ridge_identity["max_condition_number"] = config.ridge.max_condition_number
    return {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "checkpoint": _checkpoint_metadata(checkpoint),
        "readout": layout.metadata(),
        "curvature": {
            "sha256": sha256_file(curvature_path),
            "identity": dict(curvature_identity),
        },
        "calibration": {
            "sha256": sha256_file(calibration_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "identity": dict(calibration_identity),
            "records": {
                variant: {
                    target: dict(calibration_records[variant][target])
                    for target in ("energy", "forces")
                }
                for variant in _VARIANTS
            },
        },
        "test": _dataset_metadata(dataset),
        "ridge": ridge_identity,
        "min_q": config.curvature.min_q,
        "limits": {
            "max_structures": config.runtime.max_structures,
            "max_force_components_per_structure": (
                config.runtime.max_force_components_per_structure
            ),
        },
        "observables": {
            "energy": "per_atom",
            "forces": "per_cartesian_component",
            "residual": "reference_minus_prediction",
            "variance": "alpha_squared_times_q",
        },
    }


def _writer_key(variant: str, filename: str) -> str:
    return f"{variant}/{filename}"


def _new_progress(identity: Mapping[str, Any], writers: Mapping[str, TransactionalCSV]) -> dict[str, Any]:
    return {
        "identity": dict(identity),
        "status": "in_progress",
        "next_index": 0,
        "structures": 0,
        "csv_offsets": {key: writer.byte_offset for key, writer in writers.items()},
    }


def _validate_progress(progress: Mapping[str, Any]) -> None:
    if progress.get("status") not in ("in_progress", "complete"):
        raise ValueError("evaluation progress has an invalid status")
    for field in ("next_index", "structures"):
        value = progress.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"evaluation progress {field} must be non-negative")
    offsets = progress.get("csv_offsets")
    expected = {
        _writer_key(variant, filename)
        for variant in _VARIANTS
        for filename in _CSV_SPECS
    }
    if not isinstance(offsets, Mapping) or set(offsets) != expected:
        raise ValueError("evaluation progress must contain all nine CSV offsets")
    for value in offsets.values():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("evaluation CSV offsets must be non-negative integers")


def _open_writers(
    evaluation_dir: Path,
    *,
    reset: bool,
) -> dict[str, TransactionalCSV]:
    writers: dict[str, TransactionalCSV] = {}
    try:
        for variant in _VARIANTS:
            for filename, fields in _CSV_SPECS.items():
                key = _writer_key(variant, filename)
                writers[key] = TransactionalCSV(
                    evaluation_dir / variant / filename,
                    fields,
                    reset=reset,
                )
    except BaseException:
        for writer in writers.values():
            writer.close()
        raise
    return writers


def _restore_writers(
    writers: Mapping[str, TransactionalCSV],
    offsets: Mapping[str, Any],
) -> None:
    for key, writer in writers.items():
        writer.restore(int(offsets[key]))


def _close_writers(writers: Mapping[str, TransactionalCSV]) -> None:
    for writer in writers.values():
        writer.close()


def _finite_float(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(fields):
            raise ValueError(f"CSV header mismatch for {path}")
        return list(reader)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"rows": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return {
        "rows": len(values),
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "min": min(values),
        "max": max(values),
    }


def _target_summary(rows: Sequence[Mapping[str, str]], variant: str, target: str) -> dict[str, Any]:
    residuals: list[float] = []
    q_values: list[float] = []
    variances: list[float] = []
    std_values: list[float] = []
    standardized: list[float] = []
    coverage = {"1sigma": 0, "2sigma": 0, "3sigma": 0}
    for row in rows:
        if row.get("variant") != variant or row.get("target") != target:
            raise ValueError(f"CSV row does not belong to {variant}/{target}")
        residual = _finite_float(row["residual"], "residual")
        q_value = _finite_float(row["q"], "q")
        variance = _finite_float(row["variance"], "variance")
        std_value = _finite_float(row["std"], "std")
        residuals.append(residual)
        q_values.append(q_value)
        variances.append(variance)
        std_values.append(std_value)
        absolute = abs(residual)
        for multiplier in (1, 2, 3):
            if absolute <= multiplier * std_value:
                coverage[f"{multiplier}sigma"] += 1
        if std_value > 0.0:
            standardized.append(residual / std_value)
        elif residual == 0.0:
            standardized.append(0.0)
    count = len(rows)
    return {
        "rows": count,
        "mae": math.fsum(abs(value) for value in residuals) / count if count else None,
        "rmse": (
            math.sqrt(math.fsum(value * value for value in residuals) / count)
            if count
            else None
        ),
        "q": _distribution(q_values),
        "variance": _distribution(variances),
        "std": _distribution(std_values),
        "coverage": {
            key: value / count if count else None for key, value in coverage.items()
        },
        "standardized_residual": _distribution(standardized),
    }


def summarize_variant(
    variant_dir: Path,
    *,
    variant: str,
    ridge_mode: str,
    ridge: float,
    energy_alpha: float,
    force_alpha: float,
    cholesky_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute one variant summary exclusively from its formal CSV files."""
    if variant not in _VARIANTS:
        raise ValueError(f"unknown curvature variant: {variant}")
    root = Path(variant_dir)
    energy_rows = _read_csv(root / "energy.csv", ENERGY_FIELDS)
    force_rows = _read_csv(root / "force_components.csv", FORCE_FIELDS)
    structure_rows = _read_csv(
        root / "force_structure.csv", FORCE_STRUCTURE_FIELDS
    )
    structure_values = {
        field: [_finite_float(row[field], field) for row in structure_rows]
        for field in ("mae", "rmse", "mean_q", "mean_variance")
    }
    for row in structure_rows:
        if row.get("variant") != variant or row.get("target") != "forces":
            raise ValueError(f"force-structure row does not belong to {variant}/forces")
    return {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "variant": variant,
        "ridge": {"mode": ridge_mode, "value": ridge},
        "alpha": {"energy": energy_alpha, "forces": force_alpha},
        "counts": {
            "structures": len(energy_rows),
            "force_components": len(force_rows),
            "force_structures": len(structure_rows),
        },
        "energy": _target_summary(energy_rows, variant, "energy"),
        "forces": _target_summary(force_rows, variant, "forces"),
        "force_structure": {
            "rows": len(structure_rows),
            "components": sum(int(row["components"]) for row in structure_rows),
            **{
                field: _distribution(values)
                for field, values in structure_values.items()
            },
        },
        "cholesky_diagnostics": dict(cholesky_diagnostics),
    }


def _append_structure(
    writers: Mapping[str, TransactionalCSV],
    sample: Any,
    jacobians: Any,
    solvers: Mapping[str, CholeskyQuadraticForm],
    calibrations: Mapping[str, Mapping[str, Any]],
) -> None:
    energy_reference = float(sample.reference_energy_per_atom.detach().cpu())
    energy_prediction = float(jacobians.energy_per_atom)
    energy_residual = energy_reference - energy_prediction
    if not all(math.isfinite(value) for value in (energy_reference, energy_prediction, energy_residual)):
        raise ValueError(f"energy values at structure {sample.index} must be finite")

    indices = jacobians.force_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    reference_forces = sample.reference_forces.detach().to(
        device="cpu", dtype=torch.float64
    ).reshape(-1)
    predicted_forces = jacobians.forces.detach().to(
        device="cpu", dtype=torch.float64
    ).reshape(-1)
    if (
        indices.numel() != jacobians.g_forces.shape[0]
        or indices.numel() == 0
        or torch.any(indices < 0)
        or int(indices.max()) >= reference_forces.numel()
        or int(indices.max()) >= predicted_forces.numel()
    ):
        raise ValueError("force Jacobian rows do not match force indices")
    force_references = reference_forces[indices]
    force_predictions = predicted_forces[indices]
    force_residuals = force_references - force_predictions
    if not torch.isfinite(force_references).all() or not torch.isfinite(force_predictions).all():
        raise ValueError(f"force values at structure {sample.index} must be finite")

    for variant in _VARIANTS:
        energy_q = validate_q(
            solvers[variant].q(jacobians.g_energy.reshape(1, -1)),
            structure_index=sample.index,
            target="energy",
        )[0]
        force_q = validate_q(
            solvers[variant].q(jacobians.g_forces),
            structure_index=sample.index,
            target="forces",
        )
        energy_alpha = float(calibrations[variant]["energy"]["alpha"])
        force_alpha = float(calibrations[variant]["forces"]["alpha"])
        energy_variance = energy_alpha**2 * float(energy_q)
        force_variance = force_alpha**2 * force_q
        energy_std = math.sqrt(energy_variance)
        force_std = torch.sqrt(force_variance)

        writers[_writer_key(variant, "energy.csv")].append(
            [
                {
                    "structure_id": sample.structure_id,
                    "num_atoms": sample.num_atoms,
                    "reference": energy_reference,
                    "prediction": energy_prediction,
                    "residual": energy_residual,
                    "q": float(energy_q),
                    "variance": energy_variance,
                    "std": energy_std,
                    "variant": variant,
                    "target": "energy",
                }
            ]
        )
        force_rows = []
        for row_index, component_index in enumerate(indices.tolist()):
            force_rows.append(
                {
                    "structure_id": sample.structure_id,
                    "num_atoms": sample.num_atoms,
                    "atom_index": component_index // 3,
                    "direction": component_index % 3,
                    "reference": float(force_references[row_index]),
                    "prediction": float(force_predictions[row_index]),
                    "residual": float(force_residuals[row_index]),
                    "q": float(force_q[row_index]),
                    "variance": float(force_variance[row_index]),
                    "std": float(force_std[row_index]),
                    "variant": variant,
                    "target": "forces",
                }
            )
        writers[_writer_key(variant, "force_components.csv")].append(force_rows)
        writers[_writer_key(variant, "force_structure.csv")].append(
            [
                {
                    "structure_id": sample.structure_id,
                    "num_atoms": sample.num_atoms,
                    "components": indices.numel(),
                    "mae": float(force_residuals.abs().mean()),
                    "rmse": float(torch.sqrt(force_residuals.square().mean())),
                    "mean_q": float(force_q.mean()),
                    "mean_variance": float(force_variance.mean()),
                    "variant": variant,
                    "target": "forces",
                }
            ]
        )


def _validate_complete_outputs(
    evaluation_dir: Path,
    progress: Mapping[str, Any],
) -> None:
    writers = _open_writers(evaluation_dir, reset=False)
    try:
        _restore_writers(writers, progress["csv_offsets"])
    finally:
        _close_writers(writers)
    for variant in _VARIANTS:
        summary_path = evaluation_dir / variant / "summary.json"
        if not summary_path.exists():
            raise ValueError(f"complete evaluation is missing {variant}/summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, Mapping) or summary.get("variant") != variant:
            raise ValueError(f"invalid completed summary for {variant}")


def run_evaluate(config: LLPRConfig) -> Path:
    """Evaluate all three curvature variants in one structure traversal."""
    loaded = load_checkpoint(config.checkpoint, torch.device("cpu"))
    layout = discover_readout_layout(loaded.model)
    dataset = build_dataset(
        config.test.path,
        config.test.expected_sha256,
        loaded.identity.atomic_numbers,
        loaded.identity.r_max,
        loaded.identity.selected_head,
    )
    root = run_root(config, loaded.identity.sha256)
    curvature_path = root / "curvature" / "base_curvature.pt"
    curvature_identity, variants = _load_curvature(
        curvature_path, loaded.identity, layout
    )
    calibration_dir = root / "calibration" / "deterministic"
    calibration_path = calibration_dir / "calibrations.pt"
    diagnostics_path = calibration_dir / "ridge_diagnostics.json"
    calibration_identity, calibrations = _load_calibrations(
        calibration_path,
        loaded.identity,
        curvature_identity,
        config,
    )
    diagnostics = _load_cholesky_diagnostics(
        diagnostics_path, calibration_identity
    )
    identity = _evaluation_identity(
        config,
        loaded.identity,
        layout,
        curvature_path,
        curvature_identity,
        calibration_path,
        diagnostics_path,
        calibration_identity,
        calibrations,
        dataset,
    )

    evaluation_dir = root / "evaluation" / "deterministic"
    progress_path = evaluation_dir / "progress.pt"
    progress: dict[str, Any] | None = None
    if progress_path.exists():
        candidate = load_torch_artifact(progress_path)
        if not isinstance(candidate, Mapping):
            raise ValueError("evaluation progress must be a mapping")
        candidate_identity = candidate.get("identity", {})
        matching = canonical_json(candidate_identity) == canonical_json(identity)
        if candidate.get("status") == "complete" and matching:
            _validate_progress(candidate)
            _validate_complete_outputs(evaluation_dir, candidate)
            return evaluation_dir
        if config.runtime.resume:
            require_identity(candidate_identity, identity)
            _validate_progress(candidate)
            progress = dict(candidate)

    writers = _open_writers(evaluation_dir, reset=progress is None)
    try:
        if progress is None:
            progress = _new_progress(identity, writers)
            atomic_torch_save(progress_path, progress)
        else:
            _restore_writers(writers, progress["csv_offsets"])

        target_structures = dataset.size
        if config.runtime.max_structures is not None:
            target_structures = min(target_structures, config.runtime.max_structures)
        if progress["next_index"] > target_structures:
            raise ValueError("evaluation progress next_index exceeds the test limit")

        solvers = {
            variant: CholeskyQuadraticForm(
                variants[variant],
                ridge=float(calibrations[variant]["energy"]["ridge"]),
            )
            for variant in _VARIANTS
        }
        remaining = target_structures - progress["next_index"]
        if remaining > 0:
            requested_device = torch.device(config.runtime.device)
            loaded.model.to(requested_device)
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
                    force_component_chunk_size=(
                        config.runtime.force_component_chunk_size
                    ),
                    max_force_components=(
                        config.runtime.max_force_components_per_structure
                    ),
                )
                _append_structure(
                    writers,
                    sample,
                    jacobians,
                    solvers,
                    calibrations,
                )
                offsets = {key: writer.flush() for key, writer in writers.items()}
                progress["csv_offsets"] = offsets
                progress["next_index"] = sample.index + 1
                progress["structures"] += 1
                atomic_torch_save(progress_path, progress)
        if progress["next_index"] != target_structures:
            raise ValueError("test dataset ended before the configured evaluation limit")
    finally:
        _close_writers(writers)

    for variant in _VARIANTS:
        energy = calibrations[variant]["energy"]
        forces = calibrations[variant]["forces"]
        summary = summarize_variant(
            evaluation_dir / variant,
            variant=variant,
            ridge_mode=str(energy["ridge_mode"]),
            ridge=float(energy["ridge"]),
            energy_alpha=float(energy["alpha"]),
            force_alpha=float(forces["alpha"]),
            cholesky_diagnostics=diagnostics[variant],
        )
        atomic_json_dump(evaluation_dir / variant / "summary.json", summary)
    progress["status"] = "complete"
    atomic_torch_save(progress_path, progress)
    return evaluation_dir
