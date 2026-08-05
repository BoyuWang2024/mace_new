"""Strict reader for the historical FGE result layout."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from fge.artifacts import sha256_file
from fge.errors import HardFailure
from fge.prediction import validate_prediction_payload


@dataclass(frozen=True)
class LegacyMember:
    member_id: str
    cycle: int
    raw_path: Path
    raw_sha256: str
    ema_path: Path
    ema_sha256: str
    raw_metrics: dict[str, float]
    ema_metrics: dict[str, float]
    warnings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LegacyRun:
    root: Path
    base_metrics: dict[str, float]
    members: tuple[LegacyMember, ...]
    prediction: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HardFailure(f"legacy JSON cannot be loaded: {path.name}") from exc
    if not isinstance(value, dict):
        raise HardFailure(f"legacy JSON must be a mapping: {path.name}")
    return value


def _finite_metric(payload: Mapping[str, Any], old_key: str, label: str) -> float:
    value = payload.get(old_key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HardFailure(f"{label} is missing")
    result = float(value)
    if not math.isfinite(result):
        raise HardFailure(f"{label} contains NaN/Inf")
    if result < 0:
        raise HardFailure(f"{label} is negative")
    return result


def _metrics(payload: Any, label: str) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise HardFailure(f"{label} metrics are missing")
    return {
        "energy_rmse": _finite_metric(payload, "rmse_e_per_atom", f"{label}.rmse_e_per_atom"),
        "forces_rmse": _finite_metric(payload, "rmse_f", f"{label}.rmse_f"),
    }


def _inside(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise HardFailure(f"{label} path is invalid")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HardFailure(f"{label} path escapes the legacy result") from exc
    if not candidate.is_file():
        raise HardFailure(f"{label} file is missing")
    return candidate


def _verify_hash(path: Path, expected: Any, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise HardFailure(f"{label} hash is invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise HardFailure(f"{label} hash mismatch")
    return actual


def _canonical_prediction(root: Path, member_count: int) -> dict[str, Any]:
    path = root / "predictions" / "test_raw_member_predictions.pt"
    try:
        old = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise HardFailure("legacy raw prediction cannot be loaded") from exc
    if not isinstance(old, Mapping):
        raise HardFailure("legacy raw prediction must be a mapping")
    if old.get("member_source") != "raw" or old.get("split") != "test":
        raise HardFailure("legacy prediction is not the raw test branch")
    source_ids = old.get("member_ids")
    if source_ids != list(range(1, member_count + 1)):
        raise HardFailure("legacy prediction members are not contiguous")
    required = {
        "energy_members": "E_members",
        "forces_members": "F_members",
        "energy_reference": "E_ref",
        "forces_reference": "F_ref",
        "n_atoms": "n_atoms",
        "atom_to_structure": "atom_to_structure",
        "structure_ptr": "ptr",
    }
    tensors: dict[str, torch.Tensor] = {}
    for new_name, old_name in required.items():
        value = old.get(old_name)
        if not isinstance(value, torch.Tensor):
            raise HardFailure(f"legacy prediction field {old_name} is missing")
        dtype = torch.float64 if value.is_floating_point() else torch.int64
        tensors[new_name] = value.detach().to(device="cpu", dtype=dtype)
    payload: dict[str, Any] = {
        "schema_version": "fge.prediction.v1",
        "split": "test",
        "member_source": "raw",
        "member_ids": [f"member_{index:02d}" for index in range(1, member_count + 1)],
        "observables": ["energy", "forces"],
        **tensors,
    }
    validate_prediction_payload(payload)
    return payload


def read_legacy_run(root: Path) -> LegacyRun:
    """Read and hash-verify one complete legacy run without loading a model."""
    root = Path(root).resolve()
    manifest = _load_json(root / "fge_manifest.json")
    if manifest.get("crash") is not None:
        raise HardFailure("legacy experiment records a crash")
    requested = manifest.get("K_requested")
    rows = manifest.get("members")
    if not isinstance(requested, int) or requested < 2 or not isinstance(rows, list):
        raise HardFailure("legacy member registry is invalid")
    if len(rows) != requested or manifest.get("K_valid") != requested or manifest.get("K_usable") != requested:
        raise HardFailure("legacy experiment does not contain every requested member")
    members: list[LegacyMember] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping) or row.get("status") != "committed":
            raise HardFailure(f"legacy member {index} is not committed")
        if row.get("member_id") != index or row.get("cycle_index") != index:
            raise HardFailure("legacy member order is not contiguous")
        raw = _inside(root, row.get("raw_model_path"), f"member {index} raw")
        ema = _inside(root, row.get("ema_model_path"), f"member {index} EMA")
        raw_hash = _verify_hash(raw, row.get("raw_sha256"), f"member {index} raw")
        ema_hash = _verify_hash(ema, row.get("ema_sha256"), f"member {index} EMA")
        warnings: list[dict[str, Any]] = []
        for branch in ("quality", "quality_ema"):
            quality = row.get(branch)
            if isinstance(quality, Mapping) and quality.get("status") != "valid":
                warnings.append(
                    {"code": "legacy_quality", "member": f"member_{index:02d}", "branch": branch}
                )
        members.append(
            LegacyMember(
                member_id=f"member_{index:02d}",
                cycle=index,
                raw_path=raw,
                raw_sha256=raw_hash,
                ema_path=ema,
                ema_sha256=ema_hash,
                raw_metrics=_metrics(row.get("validation_raw"), f"member {index} raw"),
                ema_metrics=_metrics(row.get("validation_ema"), f"member {index} EMA"),
                warnings=tuple(warnings),
            )
        )
    base_source = manifest.get("base_validation_metrics")
    if not isinstance(base_source, Mapping):
        base_source = _load_json(root / "base_validation_metrics.json")
    return LegacyRun(
        root=root,
        base_metrics=_metrics(base_source, "base validation"),
        members=tuple(members),
        prediction=_canonical_prediction(root, requested),
    )

