"""Safe paths and atomic, immutable artifact publication primitives."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import HardFailure


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise HardFailure(f"artifact path contains symlink: {current}")


def _prepare_parent(path: Path) -> None:
    _reject_symlink_components(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent)


def safe_child(root: str | Path, *parts: str) -> Path:
    """Return a lexical child after rejecting absolute, traversal and symlinks."""

    root_path = _absolute(root)
    _reject_symlink_components(root_path)
    candidate = root_path
    for part in parts:
        fragment = Path(part)
        if fragment.is_absolute() or part in {"", ".", ".."} or ".." in fragment.parts:
            raise HardFailure(f"artifact path escapes run root: {part}")
        candidate /= fragment
    _reject_symlink_components(candidate)
    return candidate


def require_regular_child(root: str | Path, path: str | Path) -> Path:
    root_path = _absolute(root)
    candidate = _absolute(path)
    _reject_symlink_components(candidate)
    try:
        relative = candidate.relative_to(root_path)
    except ValueError as error:
        raise HardFailure(f"artifact escapes run root: {candidate}") from error
    if relative == Path(".") or ".." in relative.parts:
        raise HardFailure(f"artifact escapes run root: {candidate}")
    try:
        mode = candidate.stat().st_mode
    except OSError as error:
        raise HardFailure(f"artifact is missing: {candidate}") from error
    if not stat.S_ISREG(mode):
        raise HardFailure(f"artifact is not a regular file: {candidate}")
    return relative


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    source = _absolute(path)
    _reject_symlink_components(source)
    try:
        mode = source.stat().st_mode
    except OSError as error:
        raise HardFailure(f"could not stat artifact {source}: {error}") from error
    if not stat.S_ISREG(mode):
        raise HardFailure(f"artifact is not a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as error:
        raise HardFailure(f"could not read artifact {source}: {error}") from error
    return digest.hexdigest()


def _publish_bytes(path: str | Path, payload: bytes) -> Path:
    destination = _absolute(path)
    _prepare_parent(destination)
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and not destination.is_symlink():
            try:
                if destination.read_bytes() == payload:
                    return destination
            except OSError as error:
                raise HardFailure(f"could not verify existing artifact {destination}: {error}") from error
        raise HardFailure(f"artifact already exists with different content: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_file() and not destination.is_symlink() and destination.read_bytes() == payload:
                return destination
            raise HardFailure(f"artifact already exists with different content: {destination}") from None
        _fsync_directory(destination.parent)
    except OSError as error:
        raise HardFailure(f"could not publish artifact {destination}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | Path, document: Any) -> Path:
    try:
        payload = (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HardFailure(f"document is not JSON serializable: {error}") from error
    return _publish_bytes(path, payload)


def atomic_write_yaml(path: str | Path, document: Any) -> Path:
    try:
        payload = yaml.safe_dump(document, sort_keys=True, allow_unicode=True).encode("utf-8")
    except yaml.YAMLError as error:
        raise HardFailure(f"document is not YAML serializable: {error}") from error
    return _publish_bytes(path, payload)


def atomic_write_npy(path: str | Path, array: Any) -> Path:
    buffer = io.BytesIO()
    try:
        np.save(buffer, array, allow_pickle=False)
    except (TypeError, ValueError) as error:
        raise HardFailure(f"could not serialize NumPy array: {error}") from error
    return _publish_bytes(path, buffer.getvalue())


def atomic_write_npz(path: str | Path, **arrays: Any) -> Path:
    buffer = io.BytesIO()
    try:
        np.savez(buffer, **arrays)
    except (TypeError, ValueError) as error:
        raise HardFailure(f"could not serialize NumPy archive: {error}") from error
    return _publish_bytes(path, buffer.getvalue())


def copy_file_exact(source: str | Path, destination: str | Path) -> Path:
    source_path = _absolute(source)
    destination_path = _absolute(destination)
    source_hash = sha256_file(source_path)
    _prepare_parent(destination_path)
    if destination_path.exists() or destination_path.is_symlink():
        if destination_path.is_file() and not destination_path.is_symlink() and sha256_file(destination_path) == source_hash:
            return destination_path
        raise HardFailure(f"artifact already exists with different content: {destination_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with source_path.open("rb") as source_handle, temporary.open("wb") as target:
            shutil.copyfileobj(source_handle, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if sha256_file(temporary) != source_hash:
            raise HardFailure(f"copied artifact failed SHA-256 verification: {source_path}")
        try:
            os.link(temporary, destination_path)
        except FileExistsError:
            if destination_path.is_file() and not destination_path.is_symlink() and sha256_file(destination_path) == source_hash:
                return destination_path
            raise HardFailure(f"artifact already exists with different content: {destination_path}") from None
        _fsync_directory(destination_path.parent)
    except OSError as error:
        raise HardFailure(f"could not copy artifact {source_path} to {destination_path}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination_path


class RunLock(AbstractContextManager["RunLock"]):
    """An exclusive create-only lock file for one publication operation."""

    def __init__(self, path: str | Path) -> None:
        self.path = _absolute(path)
        self._owned = False

    def __enter__(self) -> "RunLock":
        _prepare_parent(self.path)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise HardFailure(f"run is locked: {self.path}") from None
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._owned = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False


@contextmanager
def sibling_staging(destination: str | Path) -> Iterator[Path]:
    destination_path = _absolute(destination)
    _prepare_parent(destination_path)
    if destination_path.exists() or destination_path.is_symlink():
        raise HardFailure(f"artifact already exists: {destination_path}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.", suffix=".staging", dir=destination_path.parent))
    try:
        yield staging
        if destination_path.exists() or destination_path.is_symlink():
            raise HardFailure(f"artifact already exists: {destination_path}")
        staging.rename(destination_path)
        _fsync_directory(destination_path.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@dataclass(frozen=True)
class ExperimentLayout:
    root: Path

    def __init__(self, root: str | Path) -> None:
        root_path = _absolute(root)
        _reject_symlink_components(root_path)
        object.__setattr__(self, "root", root_path)

    def member_root(self, index: int) -> Path:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise HardFailure("member_index must be a non-negative integer")
        return safe_child(self.root, "members", f"member_{index:03d}")

    def prediction_member(self, split: str, index: int, mode: str) -> Path:
        if split not in {"val", "test"}:
            raise HardFailure("split must be val or test")
        if mode not in {"raw", "ema"}:
            raise HardFailure("parameter mode must be raw or ema")
        member = self.member_root(index).name
        return safe_child(self.root, "predictions", split, "members", member, f"{mode}.npz")

    def analysis_root(self, split: str, mode: str) -> Path:
        if split not in {"val", "test"}:
            raise HardFailure("split must be val or test")
        if mode not in {"raw", "ema"}:
            raise HardFailure("parameter mode must be raw or ema")
        return safe_child(self.root, "analysis", split, mode)

