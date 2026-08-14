"""Deterministic legacy analysis key isolation without numeric computation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...bootstrap.errors import HardFailure


_KEY_MAP = {
    "force_vector": "legacy_force_vector",
    "force_structure_q95": "legacy_force_structure_q95",
}


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            mapped = _KEY_MAP.get(key, key)
            if mapped in result:
                raise HardFailure(f"legacy analysis key collision after normalization: {mapped}")
            result[mapped] = _normalize(child)
        return result
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def normalize_legacy_document(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HardFailure(f"could not load legacy analysis JSON {source}: {error}") from error
    if not isinstance(value, dict):
        raise HardFailure(f"legacy analysis JSON root must be an object: {source}")
    return _normalize(value)


def denormalize_legacy_document(value: Any) -> Any:
    inverse = {mapped: old for old, mapped in _KEY_MAP.items()}
    if isinstance(value, dict):
        return {inverse.get(key, key): denormalize_legacy_document(child) for key, child in value.items()}
    if isinstance(value, list):
        return [denormalize_legacy_document(item) for item in value]
    return value
