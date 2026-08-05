"""Lookup helpers for standard and custom extxyz observables."""

from __future__ import annotations

from typing import Any, Mapping

from .extxyz_fields import ase_value


def read_energy(atoms: Any, keys: Mapping[str, str]) -> Any:
    return ase_value(atoms, keys["energy"], array=False)


def read_forces(atoms: Any, keys: Mapping[str, str]) -> Any:
    return ase_value(atoms, keys["forces"], array=True)


def read_stress(atoms: Any, keys: Mapping[str, str]) -> Any:
    return ase_value(atoms, keys["stress"], array=False)
