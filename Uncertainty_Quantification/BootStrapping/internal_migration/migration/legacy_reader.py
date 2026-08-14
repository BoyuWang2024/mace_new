"""Read-only discovery and authentication of the old MACE BootStrapping tree."""

from __future__ import annotations

import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ...bootstrap.artifacts import sha256_file
from ...bootstrap.errors import HardFailure


@dataclass(frozen=True)
class LegacyMember:
    index: int
    member_id: str
    seed: int
    models: dict[str, Path]
    resume: Path
    indices: Path
    oob_indices: Path
    sample_summary: Path
    epoch_metrics: Path


@dataclass(frozen=True)
class LegacyPrediction:
    mode: str
    split: str
    members: Path
    base: Path


@dataclass(frozen=True)
class LegacyAnalysis:
    mode: str
    split: str
    ensemble: Path
    uncertainty: Path
    metrics: Path
    correlation: Path
    risk_coverage: Path


@dataclass(frozen=True)
class LegacyRunAudit:
    source: Path
    members: tuple[LegacyMember, ...]
    member_predictions: tuple[LegacyPrediction, ...]
    analyses: tuple[LegacyAnalysis, ...]
    source_snapshot: dict[str, dict[str, object]]

    @property
    def model_files(self) -> tuple[Path, ...]:
        return tuple(path for member in self.members for path in member.models.values())

    @property
    def resume_files(self) -> tuple[Path, ...]:
        return tuple(member.resume for member in self.members)


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise HardFailure(f"legacy {label} must not be a symlink: {path}")
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise HardFailure(f"missing legacy {label}: {path}") from error
    if not stat.S_ISREG(mode):
        raise HardFailure(f"legacy {label} must be a regular file: {path}")
    return path


def load_legacy_payload(path: str | Path) -> dict[str, Any]:
    source = _regular(Path(path).expanduser().resolve(), "payload")
    try:
        value = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise HardFailure(f"could not load legacy payload {source}: {error}") from error
    if not isinstance(value, dict):
        raise HardFailure(f"legacy payload root must be a mapping: {source}")
    return value


def _json(path: Path) -> dict[str, Any]:
    _regular(path, "JSON")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HardFailure(f"could not load legacy JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise HardFailure(f"legacy JSON root must be an object: {path}")
    return value


def _member(source: Path, member_root: Path, index: int, fallback_seed: int) -> LegacyMember:
    member_id = f"member_{index + 1:02d}"
    if member_root.name != member_id:
        raise HardFailure("legacy members must be contiguous member_01..member_B")
    models = {
        f"{mode}_{stage}": _regular(member_root / "models" / f"{member_id}.{mode}_{stage}.pt", "model")
        for mode in ("raw", "ema") for stage in ("best", "final")
    }
    pointer = _json(member_root / "resume.current.json")
    generation = pointer.get("generation")
    if not isinstance(generation, str) or re.fullmatch(r"[A-Za-z0-9_-]+", generation) is None:
        raise HardFailure(f"legacy resume generation is invalid: {member_root}")
    resume = _regular(member_root / "resume_generations" / f"{generation}.pt", "latest resume")
    summary = source / "bootstrap_samples" / f"{member_id}.summary.json"
    seed_value = _json(summary).get("seed", fallback_seed)
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise HardFailure(f"legacy member seed is invalid: {summary}")
    return LegacyMember(
        index=index, member_id=member_id, seed=seed_value, models=models, resume=resume,
        indices=_regular(source / "bootstrap_samples" / f"{member_id}.indices.pt", "bootstrap indices"),
        oob_indices=_regular(source / "bootstrap_samples" / f"{member_id}.oob_indices.pt", "OOB indices"),
        sample_summary=_regular(summary, "sample summary"),
        epoch_metrics=_regular(member_root / "epoch_metrics.jsonl", "epoch metrics"),
    )


def inspect_legacy_run(source: str | Path) -> LegacyRunAudit:
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir() or source_path.is_symlink():
        raise HardFailure(f"legacy run root is invalid: {source_path}")
    _regular(source_path / "config_resolved.yaml", "resolved config")
    _regular(source_path / "manifest.json", "manifest")
    _regular(source_path / "member_registry.json", "member registry")
    try:
        config = yaml.safe_load((source_path / "config_resolved.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise HardFailure(f"could not load legacy resolved config: {error}") from error
    bootstrap = config.get("bootstrap", {}) if isinstance(config, Mapping) else {}
    base_seed = bootstrap.get("base_seed", 2026) if isinstance(bootstrap, Mapping) else 2026
    member_roots = sorted(path for path in (source_path / "members").iterdir() if path.is_dir())
    if len(member_roots) < 2:
        raise HardFailure("legacy run must contain at least two members")
    members = tuple(_member(source_path, root, index, int(base_seed) + index + 1) for index, root in enumerate(member_roots))
    predictions: list[LegacyPrediction] = []
    analyses: list[LegacyAnalysis] = []
    for mode in ("raw", "ema"):
        for split in ("val", "test"):
            prediction_root = source_path / "predictions" / mode / split
            predictions.append(LegacyPrediction(mode, split, _regular(prediction_root / "members.pt", "member predictions"), _regular(prediction_root / "base.pt", "base predictions")))
            analyses.append(LegacyAnalysis(
                mode, split,
                _regular(source_path / "ensemble" / mode / split / "ensemble.pt", "ensemble"),
                _regular(source_path / "uncertainty" / mode / split / "uncertainty.pt", "uncertainty"),
                _regular(source_path / "metrics" / mode / split / "metrics.json", "metrics"),
                _regular(source_path / "metrics" / mode / split / "correlation.json", "correlation"),
                _regular(source_path / "metrics" / mode / split / "risk_coverage.json", "risk coverage"),
            ))
    tracked = [source_path / "config_resolved.yaml", source_path / "manifest.json", source_path / "member_registry.json"]
    tracked += list(path for member in members for path in (*member.models.values(), member.resume, member.indices, member.oob_indices, member.sample_summary, member.epoch_metrics))
    tracked += [path for item in predictions for path in (item.members, item.base)]
    tracked += [path for item in analyses for path in (item.ensemble, item.uncertainty, item.metrics, item.correlation, item.risk_coverage)]
    snapshot = {path.relative_to(source_path).as_posix(): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in tracked}
    return LegacyRunAudit(source_path, members, tuple(predictions), tuple(analyses), snapshot)
