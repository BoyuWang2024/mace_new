"""Atomic persistence helpers for ConfidenceHead artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def _temporary_file(destination: Path, *, mode: str):
    return tempfile.NamedTemporaryFile(
        mode=mode,
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
        encoding="utf-8" if "t" in mode else None,
    )


def atomic_json_dump(destination: Path, value: Any) -> None:
    """Atomically write JSON data to a file in the destination directory."""
    path = Path(destination)
    temporary_path: Path | None = None
    try:
        with _temporary_file(path, mode="w+t") as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_torch_save(destination: Path, value: Any) -> None:
    """Atomically persist a PyTorch artifact in the destination directory."""
    path = Path(destination)
    temporary_path: Path | None = None
    try:
        with _temporary_file(path, mode="w+b") as handle:
            temporary_path = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_torch_artifact(path: Path) -> dict[str, Any]:
    """Safely load a dictionary artifact onto CPU memory."""
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("torch artifact must contain a dictionary")
    return artifact
