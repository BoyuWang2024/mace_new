"""No-compute mapping of old tensors into canonical arrays."""

from __future__ import annotations

import numpy as np
import torch

from ...bootstrap.errors import HardFailure
from ...bootstrap.prediction import PredictionArrays, TargetArrays, validate_predictions, validate_targets
from .legacy_reader import LegacyAnalysis, LegacyPrediction, load_legacy_payload


def _numpy(value: object, location: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise HardFailure(f"legacy tensor is missing: {location}")
    array = value.detach().cpu().numpy()
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise HardFailure(f"legacy tensor is non-finite: {location}")
    return array


def normalize_legacy_predictions(item: LegacyPrediction) -> tuple[TargetArrays, tuple[PredictionArrays, ...]]:
    payload = load_legacy_payload(item.members)
    if payload.get("split") != item.split or payload.get("branch") != item.mode:
        raise HardFailure(f"legacy prediction identity mismatch: {item.members}")
    references = payload.get("references")
    if not isinstance(references, dict):
        raise HardFailure(f"legacy references are missing: {item.members}")
    counts = np.asarray(payload.get("n_atoms"), dtype=np.int64)
    offsets = np.asarray(payload.get("ptr"), dtype=np.int64)
    ids = np.asarray([str(value) for value in payload.get("structure_ids", [])])
    targets = TargetArrays(ids, counts, offsets, _numpy(references.get("E"), "references.E"), _numpy(references.get("F"), "references.F"), _numpy(references.get("S"), "references.S"))
    validate_targets(targets)
    energy = _numpy(payload.get("E_members"), "E_members")
    forces = _numpy(payload.get("F_members"), "F_members")
    stress = _numpy(payload.get("S_members"), "S_members")
    if energy.shape[0] != len(payload.get("member_ids", [])) or energy.shape[0] < 2:
        raise HardFailure("legacy member prediction axis is invalid")
    members = tuple(PredictionArrays(energy[index], forces[index], stress[index]) for index in range(energy.shape[0]))
    for member in members:
        validate_predictions(member, targets)
    return targets, members


def normalize_legacy_analysis(item: LegacyAnalysis) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ensemble_payload = load_legacy_payload(item.ensemble)
    uncertainty_payload = load_legacy_payload(item.uncertainty)
    ensemble = {
        "energy": _numpy(ensemble_payload.get("E_total"), "ensemble.E_total"),
        "energy_per_atom": _numpy(ensemble_payload.get("E_per_atom"), "ensemble.E_per_atom"),
        "forces": _numpy(ensemble_payload.get("F"), "ensemble.F"),
        "stress": _numpy(ensemble_payload.get("S"), "ensemble.S"),
    }
    std = uncertainty_payload.get("std")
    gmd = uncertainty_payload.get("gmd")
    if not isinstance(std, dict) or not isinstance(gmd, dict):
        raise HardFailure(f"legacy uncertainty estimators are missing: {item.uncertainty}")
    uncertainty = {
        "energy_std": _numpy(std.get("E_total"), "std.E_total"),
        "energy_per_atom_std": _numpy(std.get("E_per_atom"), "std.E_per_atom"),
        "force_std": _numpy(std.get("F_component"), "std.F_component"),
        "stress_std": _numpy(std.get("S_component"), "std.S_component"),
        "legacy_force_vector_std": _numpy(std.get("F_vector"), "std.F_vector"),
        "legacy_force_structure_q95_std": _numpy(std.get("F_structure_q95"), "std.F_structure_q95"),
        "energy_gmd": _numpy(gmd.get("E_total"), "gmd.E_total"),
        "energy_per_atom_gmd": _numpy(gmd.get("E_per_atom"), "gmd.E_per_atom"),
        "force_gmd": _numpy(gmd.get("F_component"), "gmd.F_component"),
        "stress_gmd": _numpy(gmd.get("S_component"), "gmd.S_component"),
        "legacy_force_vector_gmd": _numpy(gmd.get("F_vector"), "gmd.F_vector"),
        "legacy_force_structure_q95_gmd": _numpy(gmd.get("F_structure_q95"), "gmd.F_structure_q95"),
    }
    return ensemble, uncertainty


def load_legacy_index(path: object) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=False)
    result = _numpy(value, str(path))
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise HardFailure(f"legacy bootstrap index is invalid: {path}")
    return result
