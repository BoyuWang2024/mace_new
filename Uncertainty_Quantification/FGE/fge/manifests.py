"""Canonical manifest builders for FGE training, prediction, and results."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import formal_artifact_files, normalize_artifact_path, sha256_file
from .errors import HardFailure


def _require_finite_json(value: Any, path: str = "payload") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HardFailure(f"{path} contains NaN/Inf")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise HardFailure(f"{path} contains a non-string JSON key")
            _require_finite_json(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_json(item, f"{path}[{index}]")
        return
    raise HardFailure(f"{path} contains unsupported JSON type {type(value).__name__}")


def _artifact(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": normalize_artifact_path(root, path),
        "sha256": sha256_file(path),
    }


def build_training_manifest(
    *,
    root: Path,
    project_name: str,
    k_requested: int,
    base_model_path: Path,
    base_model_metrics: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    warnings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a strict contiguous raw/EMA member registry."""

    if k_requested < 2 or len(members) != k_requested:
        raise HardFailure("training manifest K does not match committed members")
    if set(base_model_metrics) != {"energy_rmse", "forces_rmse"}:
        raise HardFailure("base model metrics must contain energy_rmse and forces_rmse")
    serialized_base_metrics = deepcopy(dict(base_model_metrics))
    _require_finite_json(serialized_base_metrics, "base_model_metrics")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in serialized_base_metrics.values()):
        raise HardFailure("base model RMSE values must be finite and nonnegative")
    serialized_members: list[dict[str, Any]] = []
    for index, member in enumerate(members, start=1):
        expected_id = f"member_{index:02d}"
        if member.get("member_id") != expected_id or member.get("cycle") != index:
            raise HardFailure("training members must be contiguous and cycle-aligned")
        if member.get("frozen_backbone_verified") is not True:
            raise HardFailure(f"frozen backbone was not verified for {expected_id}")
        raw_metrics = deepcopy(member.get("raw_metrics"))
        ema_metrics = deepcopy(member.get("ema_metrics"))
        _require_finite_json(raw_metrics, f"{expected_id}.raw_metrics")
        _require_finite_json(ema_metrics, f"{expected_id}.ema_metrics")
        serialized_members.append(
            {
                "member_id": expected_id,
                "cycle": index,
                "raw": _artifact(root, Path(member["raw_path"])),
                "ema": _artifact(root, Path(member["ema_path"])),
                "raw_metrics": raw_metrics,
                "ema_metrics": ema_metrics,
                "frozen_backbone_verified": True,
            }
        )
    warning_payload = deepcopy(list(warnings))
    _require_finite_json(warning_payload, "warnings")
    return {
        "schema_version": "fge.training.v1",
        "project_name": project_name,
        "training_mode": "readout_only_official_mace",
        "trainable_scope": "readouts",
        "expected_readout_parameter_count": 2192,
        "ema_mode": "global",
        "main_member_source": "raw",
        "k_requested": k_requested,
        "k_committed": len(serialized_members),
        "base_model_sha256": sha256_file(base_model_path),
        "base_model_metrics": serialized_base_metrics,
        "members": serialized_members,
        "warnings": warning_payload,
    }


def build_prediction_manifest(
    *,
    root: Path,
    prediction_path: Path,
    member_count: int,
    structure_count: int,
    atom_count: int,
    observables: Sequence[str],
) -> dict[str, Any]:
    """Describe one canonical raw prediction without source paths."""

    if member_count < 2 or structure_count < 1 or atom_count < 1:
        raise HardFailure("prediction shape symbols must be positive with K >= 2")
    if tuple(observables) not in {("energy", "forces"), ("energy", "forces", "stress")}:
        raise HardFailure("prediction observables must be energy/forces with optional stress")
    return {
        "schema_version": "fge.prediction.v1",
        "split": "test",
        "member_source": "raw",
        "branches": {"raw": "available", "ema": "not_generated"},
        "observables": list(observables),
        "shape_symbols": {
            "K": member_count,
            "S": structure_count,
            "A": atom_count,
        },
        "dtype": "float64",
        "artifact": _artifact(root, prediction_path),
    }


def build_result_manifest(*, root: Path, project_name: str) -> dict[str, Any]:
    """Hash every formal artifact after validation and before final publication."""

    result_root = Path(root).resolve()
    validation_path = result_root / "validation.json"
    if not validation_path.is_file():
        raise HardFailure("result has no validation.json")
    artifacts = [_artifact(result_root, path) for path in formal_artifact_files(result_root)]
    return {
        "schema_version": "fge.result.v1",
        "project_name": project_name,
        "status": "PASS",
        "artifacts": artifacts,
    }
