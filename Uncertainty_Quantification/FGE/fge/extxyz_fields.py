"""ASE-compatible lookup for configurable extxyz fields."""

from __future__ import annotations

from typing import Any


def ase_value(atoms: Any, key: str, *, array: bool) -> Any:
    """Read a value even when ASE promotes a standard key to calc.results."""
    container = atoms.arrays if array else atoms.info
    value = container.get(key)
    if value is None and atoms.calc is not None:
        value = atoms.calc.results.get(key)
    return value
