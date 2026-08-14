"""Read-only validation of a complete canonical BootStrapping run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import HardFailure
from .manifests import validate_manifest
from .prediction import load_prediction_arrays, load_target_arrays, validate_predictions
from .schema import RUN_SCHEMA


@dataclass(frozen=True)
class RunValidation:
    member_count: int
    model_count: int
    resume_count: int
    prediction_count: int
    analysis_count: int


def _json(path: Path) -> dict[str, object]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HardFailure(f"could not load JSON {path}: {error}") from error
    if not isinstance(result, dict):
        raise HardFailure(f"JSON root must be an object: {path}")
    return result


def validate_run(root: str | Path, *, require_predictions: bool = True) -> RunValidation:
    run = Path(root).expanduser().resolve()
    if not run.is_dir() or run.is_symlink():
        raise HardFailure(f"run root is invalid: {run}")
    manifest_path = run / "run_manifest.json"
    manifest = validate_manifest(manifest_path, expected_schema=RUN_SCHEMA)
    metadata = manifest["metadata"]
    member_count = metadata.get("member_count") if isinstance(metadata, dict) else None
    if isinstance(member_count, bool) or not isinstance(member_count, int) or member_count < 2:
        raise HardFailure("run manifest member_count must be at least 2")
    if metadata.get("status") != "complete":
        raise HardFailure("run manifest status must be complete")
    origin_path = run / "origin_manifest.json"
    origin_text = origin_path.read_text(encoding="utf-8")
    if any(marker in origin_text for marker in ("/XYFS01", "\\\\wsl", "C:\\")):
        raise HardFailure("formal origin manifest contains an absolute path")
    _json(origin_path)

    expected = {entry["path"] for entry in manifest["artifacts"]}
    actual = {path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file() and path != manifest_path}
    extras = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extras:
        raise HardFailure(f"run contains unmanifested file: {extras[0]}")
    if missing:
        raise HardFailure(f"run manifest references missing file: {missing[0]}")
    forbidden = [path for path in run.rglob("*") if path.name in {"resume_generations", "wandb", "plots", "cache"}]
    if forbidden:
        raise HardFailure(f"run contains forbidden legacy output: {forbidden[0]}")

    model_count = resume_count = 0
    for index in range(member_count):
        member = run / "members" / f"member_{index:03d}"
        for mode in ("raw", "ema"):
            for stage in ("best", "final"):
                path = member / "models" / f"{mode}_{stage}.model"
                if not path.is_file() or path.is_symlink():
                    raise HardFailure(f"member model is missing: {path}")
                model_count += 1
        resume = member / "resume" / "latest.pt"
        if not resume.is_file() or resume.is_symlink():
            raise HardFailure(f"member resume is missing: {resume}")
        resume_count += 1

    prediction_count = analysis_count = 0
    if require_predictions:
        modes = metadata.get("parameter_modes")
        splits = metadata.get("splits")
        if modes != ["raw", "ema"] or splits != ["val", "test"]:
            raise HardFailure("complete run modes/splits are invalid")
        for split in splits:
            targets = load_target_arrays(run / "predictions" / split / "targets.npz")
            for mode in modes:
                for index in range(member_count):
                    values = load_prediction_arrays(run / "predictions" / split / "members" / f"member_{index:03d}" / f"{mode}.npz")
                    validate_predictions(values, targets)
                    prediction_count += 1
                with np.load(run / "uncertainty" / split / f"{mode}.npz", allow_pickle=False) as archive:
                    required = {"force_std", "force_gmd", "stress_std", "stress_gmd"}
                    if not required <= set(archive.files) or "force_rms_std" in archive.files:
                        raise HardFailure(f"uncertainty fields are invalid: {split}/{mode}")
                    if archive["force_std"].shape != targets.forces.shape or archive["stress_std"].shape != targets.stress.shape:
                        raise HardFailure(f"uncertainty shapes are invalid: {split}/{mode}")
                for name in ("metrics.json", "correlation.json", "risk_coverage.json"):
                    _json(run / "analysis" / split / mode / name)
                analysis_count += 1
    return RunValidation(member_count, model_count, resume_count, prediction_count, analysis_count)
