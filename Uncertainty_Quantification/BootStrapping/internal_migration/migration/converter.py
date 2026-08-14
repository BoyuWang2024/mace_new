"""Atomic, idempotent, no-compute publication of old MACE results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...bootstrap.artifacts import RunLock, atomic_write_json, atomic_write_npy, atomic_write_npz, copy_file_exact, sibling_staging
from ...bootstrap.errors import HardFailure
from ...bootstrap.manifests import build_run_manifest
from ...bootstrap.prediction import write_prediction_arrays, write_target_arrays
from ...bootstrap.schema import ORIGIN_SCHEMA, RUN_SCHEMA
from .legacy_reader import LegacyRunAudit
from .normalization import load_legacy_index, normalize_legacy_analysis, normalize_legacy_predictions
from .validation import validate_migrated_run


@dataclass(frozen=True)
class MigrationPublication:
    destination: Path
    written: bool
    member_count: int


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _source_unchanged(audit: LegacyRunAudit) -> None:
    from ...bootstrap.artifacts import sha256_file
    for relative, descriptor in audit.source_snapshot.items():
        path = audit.source / relative
        if path.stat().st_size != descriptor["size_bytes"] or sha256_file(path) != descriptor["sha256"]:
            raise HardFailure(f"legacy source drift detected: {relative}")


def _write_external_audit(audit: LegacyRunAudit, destination: Path, audit_root: Path) -> None:
    audit_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(audit_root / "migration_audit.json", {
        "schema": "mace.bootstrap.migration-audit/v1",
        "source": str(audit.source), "destination": str(destination), "status": "validated",
        "source_snapshot": audit.source_snapshot,
    })


def convert_legacy_run(audit: LegacyRunAudit, destination: str | Path, audit_root: str | Path) -> MigrationPublication:
    destination_path = Path(destination).expanduser().resolve()
    audit_path = Path(audit_root).expanduser().resolve()
    if _contains(audit.source, destination_path) or _contains(destination_path, audit.source):
        raise HardFailure("legacy source and destination must be separate")
    if _contains(destination_path, audit_path) or _contains(audit.source, audit_path):
        raise HardFailure("audit root must be outside source and formal destination")
    _source_unchanged(audit)
    lock_path = destination_path.parent / f".{destination_path.name}.migration.lock"
    with RunLock(lock_path):
        if destination_path.exists():
            validate_migrated_run(audit, destination_path)
            _write_external_audit(audit, destination_path, audit_path)
            return MigrationPublication(destination_path, False, len(audit.members))
        with sibling_staging(destination_path) as staging:
            artifacts: list[Path] = []
            for member in audit.members:
                target = staging / "members" / f"member_{member.index:03d}"
                for key, source in member.models.items():
                    artifacts.append(copy_file_exact(source, target / "models" / f"{key}.model"))
                artifacts.append(copy_file_exact(member.resume, target / "resume" / "latest.pt"))
                artifacts.append(copy_file_exact(member.epoch_metrics, target / "epoch_metrics.jsonl"))
                artifacts.append(copy_file_exact(member.sample_summary, target / "sampling" / "summary.json"))
                artifacts.append(atomic_write_npy(target / "sampling" / "indices.npy", load_legacy_index(member.indices)))
                artifacts.append(atomic_write_npy(target / "sampling" / "oob_indices.npy", load_legacy_index(member.oob_indices)))

            written_targets: dict[str, object] = {}
            for item in audit.member_predictions:
                targets, members = normalize_legacy_predictions(item)
                target_path = staging / "predictions" / item.split / "targets.npz"
                if item.split not in written_targets:
                    artifacts.append(write_target_arrays(target_path, targets))
                    written_targets[item.split] = targets
                for index, values in enumerate(members):
                    artifacts.append(write_prediction_arrays(staging / "predictions" / item.split / "members" / f"member_{index:03d}" / f"{item.mode}.npz", values))

            for item in audit.analyses:
                ensemble, uncertainty = normalize_legacy_analysis(item)
                artifacts.append(atomic_write_npz(staging / "ensemble" / item.split / f"{item.mode}.npz", **ensemble))
                artifacts.append(atomic_write_npz(staging / "uncertainty" / item.split / f"{item.mode}.npz", **uncertainty))
                analysis_root = staging / "analysis" / item.split / item.mode
                artifacts.append(copy_file_exact(item.metrics, analysis_root / "metrics.json"))
                artifacts.append(copy_file_exact(item.correlation, analysis_root / "correlation.json"))
                artifacts.append(copy_file_exact(item.risk_coverage, analysis_root / "risk_coverage.json"))

            origin = atomic_write_json(staging / "origin_manifest.json", {
                "schema": ORIGIN_SCHEMA, "origin": "legacy_mace_bootstrap",
                "source_tree_sha256": sorted(descriptor["sha256"] for descriptor in audit.source_snapshot.values()),
                "transforms": ["byte_copy_models_and_latest_resume", "tensor_to_numpy", "member_axis_slice", "field_rename_only"],
            })
            artifacts.append(origin)
            validate_migrated_run(audit, staging)
            build_run_manifest(staging, schema=RUN_SCHEMA, artifacts=artifacts, metadata={"member_count": len(audit.members), "parameter_modes": ["raw", "ema"], "splits": ["val", "test"], "status": "complete"})
        _write_external_audit(audit, destination_path, audit_path)
    return MigrationPublication(destination_path, True, len(audit.members))
