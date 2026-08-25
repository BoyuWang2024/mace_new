"""Post-process canonical MACE predictions with two MAD E0 experiments."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from .artifacts import atomic_write_json, atomic_write_npz, sha256_file
from .energy_reference import (
    apply_energy_correction,
    fit_model_aware_delta,
    recover_mad_e0,
    validate_common_shift,
)
from .errors import HardFailure


_METHODS = ("direct_test_mad_e0", "model_aware_val_fit")


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise HardFailure(f"could not load prediction artifact {source}: {error}") from error


def _load_members(paths: Sequence[str | Path]) -> list[dict[str, np.ndarray]]:
    if len(paths) != 8:
        raise HardFailure("exactly 8 member prediction artifacts are required")
    members = [_load_npz(path) for path in paths]
    fields = set(members[0])
    if any(set(member) != fields for member in members):
        raise HardFailure("member prediction fields differ")
    return members


def _required(targets: Mapping[str, np.ndarray], names: Sequence[str]) -> None:
    missing = [name for name in names if name not in targets]
    if missing:
        raise HardFailure(f"target artifact is missing fields: {missing}")
    numeric_names = [name for name in names if name != "structure_ids"]
    if any(not np.isfinite(np.asarray(targets[name], dtype=float)).all() for name in numeric_names):
        raise HardFailure("target artifact contains NaN or Inf")


def _stack_energy(members: Sequence[Mapping[str, np.ndarray]], expected: int) -> np.ndarray:
    values = np.stack([np.asarray(member["energy"], dtype=float).reshape(-1) for member in members], axis=0)
    if values.shape != (8, expected) or not np.isfinite(values).all():
        raise HardFailure("member energy arrays have inconsistent shapes or non-finite values")
    return values


def _stack_forces(members: Sequence[Mapping[str, np.ndarray]], expected_atoms: int) -> np.ndarray:
    values = np.stack([np.asarray(member["forces"], dtype=float) for member in members], axis=0)
    if values.shape != (8, expected_atoms, 3) or not np.isfinite(values).all():
        raise HardFailure("member force arrays have inconsistent shapes or non-finite values")
    return values




def _gmd(values: np.ndarray) -> np.ndarray:
    """Mean absolute difference over distinct unordered member pairs."""
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[0] < 2:
        raise HardFailure("GMD requires at least two members")
    pairs = [np.abs(array[left] - array[right]) for left in range(array.shape[0]) for right in range(left + 1, array.shape[0])]
    return np.mean(np.stack(pairs, axis=0), axis=0)

def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def _write_method(
    destination: Path,
    method: str,
    corrected: np.ndarray,
    target_energy: np.ndarray,
    num_atoms: np.ndarray,
    structure_ids: np.ndarray,
    fit: Mapping[str, object],
    raw: np.ndarray,
    force_metrics: Mapping[str, float],
    metadata: Mapping[str, object],
) -> None:
    validate_common_shift(raw, corrected)
    counts = np.asarray(num_atoms, dtype=float)
    ensemble = np.mean(corrected, axis=0)
    std = np.std(corrected, axis=0, ddof=1)
    gmd = _gmd(corrected)
    arrays_path = atomic_write_npz(
        destination / "corrected_member_energies.npz",
        structure_ids=np.asarray(structure_ids, dtype=str),
        num_atoms=np.asarray(num_atoms, dtype=np.int64),
        member_energies=corrected,
        ensemble_energy=ensemble,
        energy_per_atom=ensemble / counts,
        energy_std=std,
        energy_gmd=gmd,
        target_energy=np.asarray(target_energy, dtype=float),
        target_energy_per_atom=np.asarray(target_energy, dtype=float) / counts,
    )
    metric_document = {
        "energy_total": _metrics(ensemble, target_energy),
        "energy_per_atom": _metrics(ensemble / counts, target_energy / counts),
        "force": dict(force_metrics),
    }
    metrics_path = atomic_write_json(destination / "metrics.json", metric_document)
    calibration_path = atomic_write_json(destination / "calibration.json", dict(fit))
    manifest = {
        "schema": "mace.bootstrap.energy-reference-experiment/v1",
        "method": method,
        "calibration_split": metadata["calibration_split"],
        "uses_test_reference_labels": bool(metadata["uses_test_reference_labels"]),
        "evaluation_role": metadata["evaluation_role"],
        "request_metadata": dict(metadata),
        "artifacts": {
            "corrected_member_energies.npz": sha256_file(arrays_path),
            "metrics.json": sha256_file(metrics_path),
            "calibration.json": sha256_file(calibration_path),
        },
    }
    atomic_write_json(destination / "manifest.json", manifest)


def run_reference_experiments(
    *,
    test_members: Sequence[str | Path],
    test_targets: Mapping[str, np.ndarray],
    validation_members: Sequence[str | Path],
    validation_targets: Mapping[str, np.ndarray],
    test_matrix: np.ndarray,
    validation_matrix: np.ndarray,
    model_atomic_numbers: Sequence[int],
    model_e0: np.ndarray,
    output_root: str | Path,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Run both approved E0 corrections and publish immutable artifacts."""
    _required(test_targets, ("structure_ids", "num_atoms", "energy", "atomization_energy", "forces"))
    _required(validation_targets, ("structure_ids", "num_atoms", "energy", "atomization_energy", "forces"))
    test_ids = np.asarray(test_targets["structure_ids"], dtype=str)
    val_ids = np.asarray(validation_targets["structure_ids"], dtype=str)
    test_counts = np.asarray(test_targets["num_atoms"], dtype=np.int64)
    val_counts = np.asarray(validation_targets["num_atoms"], dtype=np.int64)
    if len(set(test_ids.tolist())) != test_ids.size or len(set(val_ids.tolist())) != val_ids.size:
        raise HardFailure("target structure_ids must be unique")
    test_raw_members = _load_members(test_members)
    val_raw_members = _load_members(validation_members)
    test_raw = _stack_energy(test_raw_members, test_ids.size)
    val_raw = _stack_energy(val_raw_members, val_ids.size)
    test_forces = _stack_forces(test_raw_members, int(np.sum(test_counts)))
    val_forces = _stack_forces(val_raw_members, int(np.sum(val_counts)))
    if test_forces.shape[1] != np.asarray(test_targets["forces"]).shape[0]:
        raise HardFailure("test force target atom count differs from member predictions")
    if val_forces.shape[1] != np.asarray(validation_targets["forces"]).shape[0]:
        raise HardFailure("validation force target atom count differs from member predictions")
    test_energy = np.asarray(test_targets["energy"], dtype=float)
    val_energy = np.asarray(validation_targets["energy"], dtype=float)
    test_atomization = np.asarray(test_targets["atomization_energy"], dtype=float)
    val_atomization = np.asarray(validation_targets["atomization_energy"], dtype=float)
    direct_fit = recover_mad_e0(test_matrix, test_energy, test_atomization)
    val_mean = np.mean(val_raw, axis=0)
    model_fit = fit_model_aware_delta(validation_matrix, val_energy, val_mean)
    model_e0 = np.asarray(model_e0, dtype=float).reshape(-1)
    if model_e0.size != direct_fit.values.size:
        raise HardFailure("model E0 size does not match matrix element count")
    direct_delta = direct_fit.values - model_e0
    direct_corrected = apply_energy_correction(test_raw, test_matrix, direct_delta)
    model_corrected = apply_energy_correction(test_raw, test_matrix, model_fit.values)
    force_ensemble = np.mean(test_forces, axis=0)
    force_std = np.std(test_forces, axis=0, ddof=1)
    force_gmd = _gmd(test_forces)
    force_target = np.asarray(test_targets["forces"], dtype=float)
    force_metrics = _metrics(force_ensemble, force_target)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    shared_force = root / "shared_force"
    force_path = atomic_write_npz(
        shared_force / "force.npz",
        force_ensemble=force_ensemble,
        force_std=force_std,
        force_gmd=force_gmd,
        target_forces=force_target,
    )
    force_metrics_path = atomic_write_json(shared_force / "metrics.json", force_metrics)
    atomic_write_json(shared_force / "manifest.json", {
        "schema": "mace.bootstrap.shared-force/v1",
        "domain": "forces",
        "artifacts": {"force.npz": sha256_file(force_path), "metrics.json": sha256_file(force_metrics_path)},
    })
    base_metadata = {
        "model_atomic_numbers": [int(value) for value in model_atomic_numbers],
        "model_e0": model_e0.tolist(),
        "validation_structure_count": int(val_ids.size),
        "test_structure_count": int(test_ids.size),
        "uses_test_reference_labels": False,
        "evaluation_role": "calibrated_test",
    }
    if metadata:
        base_metadata.update(metadata)
    direct_metadata = dict(base_metadata, calibration_split="test", uses_test_reference_labels=True, evaluation_role="oracle_baseline")
    model_metadata = dict(base_metadata, calibration_split="validation", uses_test_reference_labels=False, evaluation_role="calibrated_test")
    fit_common = {
        "model_e0": model_e0.tolist(),
        "direct_mad_e0": direct_fit.values.tolist(),
        "direct_delta": direct_delta.tolist(),
        "model_aware_delta": model_fit.values.tolist(),
        "test_matrix_rank": direct_fit.rank,
        "validation_matrix_rank": model_fit.rank,
        "test_singular_values": direct_fit.singular_values.tolist(),
        "validation_singular_values": model_fit.singular_values.tolist(),
        "direct_test_fit_max_abs_residual": direct_fit.max_abs_residual,
        "model_aware_validation_fit_max_abs_residual": model_fit.max_abs_residual,
    }
    _write_method(root / "direct_test_mad_e0", "direct_test_mad_e0", direct_corrected, test_energy, test_counts, test_ids, dict(fit_common, **direct_metadata), test_raw, force_metrics, direct_metadata)
    _write_method(root / "model_aware_val_fit", "model_aware_val_fit", model_corrected, test_energy, test_counts, test_ids, dict(fit_common, **model_metadata), test_raw, force_metrics, model_metadata)
    experiment_manifest = {
        "schema": "mace.bootstrap.energy-reference/v1",
        "methods": list(_METHODS),
        "shared_force_manifest": str((shared_force / "manifest.json").name),
        "metadata": base_metadata,
    }
    atomic_write_json(root / "experiment_manifest.json", experiment_manifest)
    return root


__all__ = ["run_reference_experiments"]
