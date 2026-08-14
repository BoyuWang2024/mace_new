"""Explicitly trusted, CPU-only checkpoint loading and capability audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .artifacts import sha256_file
from .errors import HardFailure


@dataclass(frozen=True)
class CheckpointAudit:
    sha256: str
    size_bytes: int
    inference_capable: bool
    resume_capable: bool
    raw_available: bool
    ema_available: bool


def load_trusted_checkpoint(path: str | Path, *, expected_sha256: str | None = None) -> Any:
    source = Path(path).expanduser().resolve()
    digest = sha256_file(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise HardFailure(f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise HardFailure(f"could not load trusted checkpoint {source}: {error}") from error


def audit_checkpoint(path: str | Path, *, trusted: bool) -> CheckpointAudit:
    if not trusted:
        raise HardFailure("checkpoint must be explicitly trusted before full-object loading")
    source = Path(path).expanduser().resolve()
    digest = sha256_file(source)
    document = load_trusted_checkpoint(source, expected_sha256=digest)
    is_module = isinstance(document, torch.nn.Module)
    mapping: Mapping[str, Any] = document if isinstance(document, Mapping) else {}
    raw_available = is_module or "model" in mapping or "model_state_dict" in mapping
    ema_available = "ema" in mapping or "ema_state_dict" in mapping
    resume_capable = raw_available and "optimizer" in mapping and "epoch" in mapping
    return CheckpointAudit(
        sha256=digest,
        size_bytes=source.stat().st_size,
        inference_capable=raw_available,
        resume_capable=resume_capable,
        raw_available=raw_available,
        ema_available=ema_available,
    )
