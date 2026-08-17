"""Read-only binding of an authenticated completed bootstrap ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import require_regular_child, sha256_file
from .errors import HardFailure
from .manifests import validate_manifest
from .schema import RUN_SCHEMA
from .validation import validate_run


@dataclass(frozen=True)
class CompletedMember:
    index: int
    path: Path
    sha256: str


@dataclass(frozen=True)
class CompletedRun:
    root: Path
    member_count: int
    run_manifest_sha256: str
    members: tuple[CompletedMember, ...]


def audit_completed_run(
    root: str | Path,
    *,
    expected_members: int = 8,
) -> CompletedRun:
    """Validate a run and bind exactly its ordered raw-best model artifacts."""
    if isinstance(expected_members, bool) or not isinstance(expected_members, int) or expected_members < 2:
        raise HardFailure("expected member count must be an integer of at least 2")
    run = Path(root).expanduser().resolve()
    validation = validate_run(run, require_predictions=False)
    if validation.member_count != expected_members:
        raise HardFailure(
            f"completed run member count must be {expected_members}, got {validation.member_count}"
        )
    manifest_path = run / "run_manifest.json"
    manifest = validate_manifest(manifest_path, expected_schema=RUN_SCHEMA)
    descriptors = {
        entry["path"]: entry
        for entry in manifest["artifacts"]
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    members: list[CompletedMember] = []
    for index in range(expected_members):
        relative = f"members/member_{index:03d}/models/raw_best.model"
        descriptor = descriptors.get(relative)
        if descriptor is None:
            raise HardFailure(f"completed run is missing raw-best manifest entry: {relative}")
        path = run / relative
        require_regular_child(run, path)
        digest = sha256_file(path)
        if descriptor.get("sha256") != digest:
            raise HardFailure(f"completed run raw-best SHA-256 drift: {relative}")
        members.append(CompletedMember(index=index, path=path, sha256=digest))
    return CompletedRun(
        root=run,
        member_count=expected_members,
        run_manifest_sha256=sha256_file(manifest_path),
        members=tuple(members),
    )


__all__ = ["CompletedMember", "CompletedRun", "audit_completed_run"]
