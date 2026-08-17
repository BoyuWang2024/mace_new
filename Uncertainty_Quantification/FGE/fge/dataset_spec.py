"""Source-neutral dataset descriptions for derived FGE inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .errors import HardFailure


_LABEL_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_EXTXYZ_SUFFIXES = {".xyz", ".extxyz"}


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: Path
    energy_key: str = "energy"
    forces_key: str = "forces"
    stress_key: str = "stress"
    head_name: str = "default"
    compute_stress: bool = False
    batch_size: int = 64
    shard_size: int = 256

    def __post_init__(self) -> None:
        path = Path(self.path)
        object.__setattr__(self, "path", path)
        if _LABEL_PATTERN.fullmatch(self.label) is None:
            raise HardFailure("dataset label must use lowercase letters, digits, '_' or '-'")
        if path.suffix.lower() not in _EXTXYZ_SUFFIXES:
            raise HardFailure("dataset input must use an .xyz or .extxyz container")
        if not path.is_file():
            raise HardFailure(f"dataset input does not exist or is not a file: {path}")
        for name in ("batch_size", "shard_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise HardFailure(f"{name} must be a positive integer")

    @property
    def observables(self) -> tuple[str, ...]:
        if self.compute_stress:
            return ("energy", "forces", "stress")
        return ("energy", "forces")

    @property
    def required(self) -> frozenset[str]:
        return frozenset(self.observables)

    @property
    def keys(self) -> dict[str, str]:
        return {
            "energy": self.energy_key,
            "forces": self.forces_key,
            "stress": self.stress_key,
            "head": self.head_name,
        }

    def neutral_manifest(self) -> dict[str, object]:
        return {
            "schema_version": "fge.dataset.v1",
            "dataset": self.label,
            "observables": list(self.observables),
            "keys": self.keys,
            "batch_size": self.batch_size,
            "shard_size": self.shard_size,
        }
