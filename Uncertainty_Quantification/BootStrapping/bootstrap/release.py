"""Allowlist-only source release construction."""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

from .errors import HardFailure


_ROOT_FILES = {"__init__.py", "README.md", "publication_files.txt"}
_ROOT_DIRS = {"bootstrap", "scripts", "configs"}
_BINARY_SUFFIXES = {".pt", ".model", ".npz", ".npy", ".png"}


def _allowed(package: Path, path: Path) -> bool:
    relative = path.relative_to(package)
    if len(relative.parts) == 1:
        return relative.name in _ROOT_FILES
    if relative.parts[0] not in _ROOT_DIRS:
        return False
    if any(part in {"__pycache__", "tests", "outputs", "internal_migration"} for part in relative.parts):
        return False
    return path.suffix not in _BINARY_SUFFIXES and not path.name.endswith(".pyc")


def build_release(package_root: str | Path, destination: str | Path) -> Path:
    package = Path(package_root).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not package.is_dir() or package.is_symlink():
        raise HardFailure(f"package root is invalid: {package}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for path in sorted(candidate for candidate in package.rglob("*") if candidate.is_file() and _allowed(package, candidate)):
                archive.add(path, arcname=(Path(package.name) / path.relative_to(package)).as_posix(), recursive=False)
        os.replace(temporary, target)
    except (OSError, tarfile.TarError) as error:
        raise HardFailure(f"could not build release archive: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return target
