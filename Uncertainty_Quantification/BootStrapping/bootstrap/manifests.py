"""Creation and recursive validation of immutable public manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import atomic_write_json, require_regular_child, sha256_file
from .errors import HardFailure


def artifact_descriptor(root: str | Path, path: str | Path) -> dict[str, object]:
    root_path = Path(root).expanduser().absolute()
    artifact_path = Path(path).expanduser().absolute()
    relative = require_regular_child(root_path, artifact_path)
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
    }


def build_run_manifest(
    root: str | Path,
    *,
    schema: str,
    artifacts: Iterable[str | Path],
    metadata: Mapping[str, Any] | None = None,
    filename: str = "run_manifest.json",
) -> Path:
    root_path = Path(root).expanduser().absolute()
    entries = [artifact_descriptor(root_path, artifact) for artifact in artifacts]
    document = {
        "schema": schema,
        "metadata": dict(metadata or {}),
        "artifacts": sorted(entries, key=lambda item: str(item["path"])),
    }
    return atomic_write_json(root_path / filename, document)


def validate_manifest(path: str | Path, *, expected_schema: str) -> dict[str, object]:
    manifest_path = Path(path).expanduser().absolute()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise HardFailure(f"manifest is not a regular file: {manifest_path}")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HardFailure(f"could not load manifest {manifest_path}: {error}") from error
    if not isinstance(document, dict) or set(document) != {"schema", "metadata", "artifacts"}:
        raise HardFailure("manifest has unknown or missing root keys")
    if document["schema"] != expected_schema:
        raise HardFailure(f"manifest schema must be {expected_schema}, got {document['schema']}")
    if not isinstance(document["metadata"], dict) or not isinstance(document["artifacts"], list):
        raise HardFailure("manifest metadata/artifacts have invalid types")
    root = manifest_path.parent
    for entry in document["artifacts"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes"}:
            raise HardFailure("manifest artifact entry is invalid")
        relative_text = entry["path"]
        if not isinstance(relative_text, str):
            raise HardFailure("manifest artifact path must be a string")
        relative = Path(relative_text)
        if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
            raise HardFailure(f"manifest artifact escapes run root: {relative_text}")
        artifact = root / relative
        require_regular_child(root, artifact)
        if artifact.stat().st_size != entry["size_bytes"]:
            raise HardFailure(f"manifest artifact size drift: {relative_text}")
        if sha256_file(artifact) != entry["sha256"]:
            raise HardFailure(f"manifest artifact SHA-256 drift: {relative_text}")
    return document
