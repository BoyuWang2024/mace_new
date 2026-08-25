"""Lookup helpers for standard and custom extxyz observables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .extxyz_fields import ase_value


def _calculator_result(atoms: Any, key: str) -> Any:
    calculator = getattr(atoms, "calc", None)
    results = getattr(calculator, "results", None)
    if isinstance(results, Mapping):
        value = results.get(key)
        if value is not None:
            return value
    return None


def read_energy(atoms: Any, keys: Mapping[str, str]) -> Any:
    key = keys["energy"]
    value = _calculator_result(atoms, key)
    return value if value is not None else ase_value(atoms, key, array=False)


def read_forces(atoms: Any, keys: Mapping[str, str]) -> Any:
    key = keys["forces"]
    value = _calculator_result(atoms, key)
    return value if value is not None else ase_value(atoms, key, array=True)


def read_stress(atoms: Any, keys: Mapping[str, str]) -> Any:
    key = keys["stress"]
    value = _calculator_result(atoms, key)
    return value if value is not None else ase_value(atoms, key, array=False)


def read_atomization_energy(atoms: Any, key: str = "atomization_energy") -> Any:
    return ase_value(atoms, key, array=False)
