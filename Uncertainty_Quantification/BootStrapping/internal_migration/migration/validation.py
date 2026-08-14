"""Strict source-to-canonical equivalence checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...bootstrap.artifacts import sha256_file
from ...bootstrap.errors import HardFailure
from ...bootstrap.prediction import load_prediction_arrays, load_target_arrays
from .legacy_reader import LegacyRunAudit
from .normalization import load_legacy_index, normalize_legacy_analysis, normalize_legacy_predictions


@dataclass(frozen=True)
class MigrationValidation:
    validated_models: int
    validated_resumes: int
    validated_predictions: int
    validated_analyses: int


def _equal(expected: np.ndarray, actual: np.ndarray, location: str) -> None:
    if expected.dtype != actual.dtype or expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise HardFailure(f"migrated array differs: {location}")


def validate_migrated_run(audit: LegacyRunAudit, destination: str | Path) -> MigrationValidation:
    root = Path(destination).expanduser().resolve()
    model_count = resume_count = prediction_count = analysis_count = 0
    for member in audit.members:
        target_root = root / "members" / f"member_{member.index:03d}"
        for key, source in member.models.items():
            if sha256_file(source) != sha256_file(target_root / "models" / f"{key}.model"):
                raise HardFailure(f"migrated model SHA-256 differs: {source}")
            model_count += 1
        if sha256_file(member.resume) != sha256_file(target_root / "resume" / "latest.pt"):
            raise HardFailure(f"migrated resume SHA-256 differs: {member.resume}")
        resume_count += 1
        _equal(load_legacy_index(member.indices), np.load(target_root / "sampling" / "indices.npy", allow_pickle=False), "bootstrap indices")
        _equal(load_legacy_index(member.oob_indices), np.load(target_root / "sampling" / "oob_indices.npy", allow_pickle=False), "OOB indices")
    canonical_targets: dict[str, object] = {}
    for item in audit.member_predictions:
        targets, members = normalize_legacy_predictions(item)
        loaded_targets = load_target_arrays(root / "predictions" / item.split / "targets.npz")
        for name in targets.__dataclass_fields__:
            _equal(getattr(targets, name), getattr(loaded_targets, name), f"{item.mode}/{item.split}/targets.{name}")
        canonical_targets[item.split] = loaded_targets
        for index, expected in enumerate(members):
            actual = load_prediction_arrays(root / "predictions" / item.split / "members" / f"member_{index:03d}" / f"{item.mode}.npz")
            for name in expected.__dataclass_fields__:
                _equal(getattr(expected, name), getattr(actual, name), f"{item.mode}/{item.split}/member_{index}.{name}")
            prediction_count += 1
    for item in audit.analyses:
        ensemble, uncertainty = normalize_legacy_analysis(item)
        with np.load(root / "ensemble" / item.split / f"{item.mode}.npz", allow_pickle=False) as actual:
            for name, expected in ensemble.items():
                _equal(expected, actual[name], f"ensemble/{item.mode}/{item.split}/{name}")
        with np.load(root / "uncertainty" / item.split / f"{item.mode}.npz", allow_pickle=False) as actual:
            for name, expected in uncertainty.items():
                _equal(expected, actual[name], f"uncertainty/{item.mode}/{item.split}/{name}")
        analysis_count += 1
    return MigrationValidation(model_count, resume_count, prediction_count, analysis_count)
