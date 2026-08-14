"""Canonical target and prediction arrays stored as immutable NPZ files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .artifacts import atomic_write_npz
from .errors import HardFailure


@dataclass(frozen=True)
class TargetArrays:
    structure_ids: NDArray[np.str_]
    num_atoms: NDArray[np.int64]
    atom_offsets: NDArray[np.int64]
    energy: NDArray[np.floating]
    forces: NDArray[np.floating]
    stress: NDArray[np.floating]


@dataclass(frozen=True)
class PredictionArrays:
    energy: NDArray[np.floating]
    forces: NDArray[np.floating]
    stress: NDArray[np.floating]


def _array(value: object, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind == "O":
        raise HardFailure(f"{name} must not use object dtype")
    return result


def validate_targets(targets: TargetArrays) -> None:
    structures = int(targets.structure_ids.shape[0])
    if targets.structure_ids.ndim != 1 or len(set(targets.structure_ids.tolist())) != structures:
        raise HardFailure("target structure_ids must be a unique rank-one array")
    if targets.num_atoms.shape != (structures,) or targets.atom_offsets.shape != (structures + 1,):
        raise HardFailure("target atom layout shape mismatch")
    if not np.array_equal(targets.atom_offsets, np.concatenate(([0], np.cumsum(targets.num_atoms)))):
        raise HardFailure("target atom_offsets do not match num_atoms")
    atoms = int(targets.atom_offsets[-1])
    if targets.energy.shape != (structures,):
        raise HardFailure("target energy shape mismatch")
    if targets.forces.shape != (atoms, 3):
        raise HardFailure("target forces shape must be [total_atoms, 3]")
    if targets.stress.shape != (structures, 6):
        raise HardFailure("target stress shape must be [structures, 6]")


def validate_predictions(prediction: PredictionArrays, targets: TargetArrays) -> None:
    validate_targets(targets)
    if prediction.energy.shape != targets.energy.shape:
        raise HardFailure("prediction energy shape mismatch")
    if prediction.forces.shape != targets.forces.shape:
        raise HardFailure("prediction forces shape mismatch")
    if prediction.stress.shape != targets.stress.shape:
        raise HardFailure("prediction stress shape mismatch")
    for name in ("energy", "forces", "stress"):
        if not np.isfinite(getattr(prediction, name)).all():
            raise HardFailure(f"prediction {name} contains non-finite values")


def write_target_arrays(path: str | Path, targets: TargetArrays) -> Path:
    normalized = TargetArrays(*(_array(getattr(targets, name), name) for name in ("structure_ids", "num_atoms", "atom_offsets", "energy", "forces", "stress")))
    validate_targets(normalized)
    return atomic_write_npz(path, **{name: getattr(normalized, name) for name in normalized.__dataclass_fields__})


def write_prediction_arrays(path: str | Path, prediction: PredictionArrays) -> Path:
    normalized = PredictionArrays(*(_array(getattr(prediction, name), name) for name in ("energy", "forces", "stress")))
    return atomic_write_npz(path, energy=normalized.energy, forces=normalized.forces, stress=normalized.stress)


def _load(path: str | Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != set(required):
                raise HardFailure(f"NPZ fields differ for {source}")
            return {name: np.array(archive[name], copy=True) for name in required}
    except (OSError, ValueError) as error:
        raise HardFailure(f"could not load NPZ {source}: {error}") from error


def load_target_arrays(path: str | Path) -> TargetArrays:
    fields = ("structure_ids", "num_atoms", "atom_offsets", "energy", "forces", "stress")
    result = TargetArrays(**_load(path, fields))
    validate_targets(result)
    return result


def load_prediction_arrays(path: str | Path) -> PredictionArrays:
    fields = ("energy", "forces", "stress")
    return PredictionArrays(**_load(path, fields))
