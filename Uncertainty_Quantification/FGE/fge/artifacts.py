"""Atomic artifact IO and the canonical FGE result layout."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import yaml

from .errors import HardFailure


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one regular file without loading it into memory."""

    source = Path(path)
    if not source.is_file():
        raise HardFailure(f"artifact is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


@contextmanager
def sibling_temporary_path(target: Path) -> Iterator[Path]:
    """Yield a unique file beside target so os.replace remains atomic."""

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """Write strict JSON and atomically replace the destination."""

    try:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise HardFailure(f"non-finite JSON or unsupported value: {exc}") from exc
    with sibling_temporary_path(Path(path)) as temporary:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        _fsync_file(temporary)
        os.replace(temporary, path)


def atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic UTF-8 YAML through a sibling temporary file."""

    text = yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)
    with sibling_temporary_path(Path(path)) as temporary:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        _fsync_file(temporary)
        os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    """Save a CPU-safe torch payload atomically."""

    with sibling_temporary_path(Path(path)) as temporary:
        torch.save(payload, temporary)
        _fsync_file(temporary)
        os.replace(temporary, path)


def normalize_artifact_path(root: Path, path: Path) -> str:
    """Return a POSIX relative path and reject every result-root escape."""

    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise HardFailure(
            f"artifact must use a relative path inside result root: {path}"
        ) from exc
    if relative == Path(".") or ".." in relative.parts:
        raise HardFailure(f"artifact must use a relative path to a file: {path}")
    return relative.as_posix()


@dataclass(frozen=True)
class ExperimentLayout:
    """Canonical paths for one formal FGE result."""

    root: Path

    @property
    def preflight_dir(self) -> Path:
        return self.root / "preflight"

    @property
    def training_dir(self) -> Path:
        return self.root / "training"

    @property
    def training_manifest(self) -> Path:
        return self.training_dir / "manifest.json"

    @property
    def prediction_dir(self) -> Path:
        return self.root / "prediction"

    @property
    def prediction_tensor(self) -> Path:
        return self.prediction_dir / "test_raw.pt"

    @property
    def prediction_manifest(self) -> Path:
        return self.prediction_dir / "manifest.json"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"

    @property
    def result_manifest(self) -> Path:
        return self.root / "result_manifest.json"

    @property
    def work_dir(self) -> Path:
        return self.root / "_work"


class StagingExperiment:
    """Build a complete sibling directory and publish it with one rename."""

    def __init__(self, final_root: Path):
        self.final_root = Path(final_root).resolve()
        if (self.final_root / "result_manifest.json").exists():
            raise HardFailure(f"completed result already exists: {self.final_root}")
        if self.final_root.exists():
            raise HardFailure(f"output directory already exists: {self.final_root}")
        self.staging_root = self.final_root.with_name(
            f".{self.final_root.name}.staging-{uuid.uuid4().hex}"
        )
        self.layout = ExperimentLayout(self.staging_root)
        self._published = False

    def __enter__(self) -> "StagingExperiment":
        self.staging_root.parent.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir()
        return self

    def publish(self) -> Path:
        if self._published:
            raise HardFailure("staging experiment was already published")
        if not self.layout.validation.is_file():
            raise HardFailure("staging result has no validation.json")
        if self.final_root.exists():
            raise HardFailure(f"output directory already exists: {self.final_root}")
        os.replace(self.staging_root, self.final_root)
        self._published = True
        return self.final_root

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if not self._published and self.staging_root.exists():
            shutil.rmtree(self.staging_root)


def formal_artifact_files(root: Path) -> tuple[Path, ...]:
    """Enumerate formal files while excluding transient/internal state."""

    result_root = Path(root).resolve()
    excluded_roots = {"_work", "_internal_migration"}
    files: list[Path] = []
    for path in result_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(result_root)
        if relative.parts[0] in excluded_roots:
            continue
        if relative.as_posix() == "result_manifest.json":
            continue
        files.append(path)
    return tuple(sorted(files, key=lambda item: item.relative_to(result_root).as_posix()))
