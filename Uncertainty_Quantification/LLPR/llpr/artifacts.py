"""Stable identities and crash-safe persistence for LLPR artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

import torch


SCHEMA_VERSION = 1
FORMULA_VERSION = "mace-readout-gram-energy-per-atom-v1"


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for artifact identity checks."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_id(value: Any, length: int = 16) -> str:
    """Return a short, deterministic SHA-256 identifier for *value*."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Ensure an artifact identity matches the identity it was built with."""
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(
            f"artifact identity mismatch: expected={canonical_json(expected)}, "
            f"actual={canonical_json(actual)}"
        )


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        return Path(handle.name)


def _replace_temporary_file(temporary_path: Path, destination: Path) -> None:
    try:
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_json_dump(path: Path, value: Any) -> None:
    """Write JSON atomically so readers never observe a partial artifact."""
    destination = Path(path)
    temporary_path = _temporary_path(destination)
    try:
        temporary_path.write_text(canonical_json(value), encoding="utf-8")
        _replace_temporary_file(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_torch_save(path: Path, value: Any) -> None:
    """Save a PyTorch artifact atomically."""
    destination = Path(path)
    temporary_path = _temporary_path(destination)
    try:
        torch.save(value, temporary_path)
        _replace_temporary_file(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_torch_artifact(path: Path) -> Any:
    """Load an LLPR artifact on CPU without relying on the source device."""
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")
