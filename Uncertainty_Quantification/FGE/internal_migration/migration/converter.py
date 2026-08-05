"""Atomic conversion into the public, source-independent FGE schema."""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from typing import Any

from fge.artifacts import (
    StagingExperiment,
    atomic_torch_save,
    atomic_write_json,
    sha256_file,
    sibling_temporary_path,
)
from fge.errors import HardFailure
from fge.evaluation import evaluate_prediction
from fge.manifests import build_prediction_manifest, build_training_manifest
from fge.prediction import validate_prediction_payload
from fge.validation import schema_signature, validate_result

from .legacy_reader import LegacyRun, read_legacy_run


def _copy_verified(source: Path, destination: Path, expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sibling_temporary_path(destination) as temporary:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected_hash:
            raise HardFailure(f"copied artifact hash mismatch: {destination.name}")
        os.replace(temporary, destination)


def _write_formal_inputs(config: Any, legacy: LegacyRun, root: Path) -> None:
    members: list[dict[str, Any]] = []
    training_warnings: list[dict[str, Any]] = []
    for member in legacy.members:
        raw_destination = root / "training" / "members" / "raw" / f"{member.member_id}.model"
        ema_destination = root / "training" / "members" / "ema" / f"{member.member_id}.model"
        _copy_verified(member.raw_path, raw_destination, member.raw_sha256)
        _copy_verified(member.ema_path, ema_destination, member.ema_sha256)
        training_warnings.extend(member.warnings)
        members.append(
            {
                "member_id": member.member_id,
                "cycle": member.cycle,
                "raw_path": raw_destination,
                "ema_path": ema_destination,
                "raw_metrics": member.raw_metrics,
                "ema_metrics": member.ema_metrics,
                "frozen_backbone_verified": True,
            }
        )
    base_model = Path(config.section("paths")["base_checkpoint"])
    training_manifest = build_training_manifest(
        root=root,
        project_name=config.project_name,
        k_requested=len(members),
        base_model_path=base_model,
        base_model_metrics=legacy.base_metrics,
        members=members,
        warnings=training_warnings,
    )
    atomic_write_json(root / "training" / "manifest.json", training_manifest)

    shape = validate_prediction_payload(legacy.prediction)
    prediction_path = root / "prediction" / "test_raw.pt"
    atomic_torch_save(prediction_path, legacy.prediction)
    atomic_write_json(
        root / "prediction" / "manifest.json",
        build_prediction_manifest(
            root=root,
            prediction_path=prediction_path,
            member_count=shape.members,
            structure_count=shape.structures,
            atom_count=shape.atoms,
            observables=legacy.prediction["observables"],
        ),
    )


def _write_audit(audit_path: Path, legacy: LegacyRun, destination: Path) -> None:
    atomic_write_json(
        audit_path,
        {
            "schema_version": "fge.internal-conversion-audit.v1",
            "status": "PASS",
            "source_root": str(legacy.root),
            "destination": str(destination),
            "member_count": len(legacy.members),
            "source_hashes": {
                member.member_id: {"raw": member.raw_sha256, "ema": member.ema_sha256}
                for member in legacy.members
            },
            "formal_schema_signature": schema_signature(destination),
        },
    )


def convert_legacy_run(
    *, config: Any, legacy_root: Path, destination: Path, audit_path: Path
) -> Path:
    """Convert existing model/prediction artifacts; never train or infer."""
    legacy = read_legacy_run(legacy_root)
    configured_k = int(config.section("training")["member_count"])
    if len(legacy.members) != configured_k:
        raise HardFailure("configuration member_count does not match legacy result")
    base_model = Path(config.section("paths")["base_checkpoint"])
    if not base_model.is_file():
        raise HardFailure("configured base checkpoint is missing")
    destination = Path(destination).resolve()
    with StagingExperiment(destination) as staging:
        _write_formal_inputs(config, legacy, staging.staging_root)
        evaluate_prediction(config, staging.staging_root)
        validate_result(config, staging.staging_root)
        published = staging.publish()
    try:
        _write_audit(Path(audit_path), legacy, published)
    except Exception as exc:
        warnings.warn(f"formal result is valid but internal audit could not be written: {exc}", RuntimeWarning)
    return published

