"""Create an auditable element-compatible derivative of a labeled extxyz file."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import ase.io
import numpy as np
from ase import Atoms

from ..artifacts import atomic_json_dump
from ..data import ENERGY_KEY, FORCES_KEY
from ..identity import sha256_file


PREPARED_DATASET_SCHEMA_VERSION = 1


class PreparedDatasetError(RuntimeError):
    """Compatible dataset evidence is partial, conflicting, or invalid."""


@dataclass(frozen=True)
class PreparedDataset:
    dataset_path: Path
    exclusions_path: Path
    source_index_path: Path
    manifest_path: Path
    source_indices: tuple[int, ...]
    exclusions: tuple[dict[str, object], ...]
    manifest: dict[str, object]

    @property
    def output_paths(self) -> tuple[Path, ...]:
        return (
            self.dataset_path,
            self.exclusions_path,
            self.source_index_path,
            self.manifest_path,
        )


def _paths(output_path: Path) -> tuple[Path, Path, Path, Path]:
    output = Path(output_path).expanduser().resolve()
    stem = output.stem
    base = stem[: -len("-compatible")] if stem.endswith("-compatible") else stem
    return (
        output,
        output.with_name(f"{base}-exclusions.csv"),
        output.with_name(f"{base}-source-index.csv"),
        output.with_name(f"{base}-dataset-manifest.json"),
    )


def _structures(path: Path) -> list[Atoms]:
    try:
        values = ase.io.read(path, index=":")
    except Exception as error:
        raise PreparedDatasetError(f"could not read source dataset: {error}") from error
    if isinstance(values, Atoms):
        values = [values]
    if not values:
        raise PreparedDatasetError("source dataset contains no structures")
    for index, atoms in enumerate(values):
        if not len(atoms):
            raise PreparedDatasetError(f"source structure {index} contains no atoms")
        if ENERGY_KEY not in atoms.info or FORCES_KEY not in atoms.arrays:
            raise PreparedDatasetError(f"source structure {index} is missing labels")
        energy = np.asarray(atoms.info[ENERGY_KEY])
        forces = np.asarray(atoms.arrays[FORCES_KEY])
        if energy.shape != () or forces.shape != (len(atoms), 3):
            raise PreparedDatasetError(f"source structure {index} label shape differs")
        if not np.isfinite(energy).all() or not np.isfinite(forces).all():
            raise PreparedDatasetError(f"source structure {index} labels are non-finite")
    return values


def _csv_bytes(fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_extxyz(path: Path, structures: Sequence[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(dir=path.parent, suffix=".xyz")
    os.close(descriptor)
    temporary = Path(raw)
    try:
        ase.io.write(temporary, list(structures), format="extxyz")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _reuse(
    outputs: tuple[Path, Path, Path, Path],
    *,
    source_path: Path,
    source_sha256: str,
    unsupported: tuple[int, ...],
) -> PreparedDataset:
    dataset, exclusions_path, source_index_path, manifest_path = outputs
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise PreparedDatasetError(f"dataset manifest is unreadable: {error}") from error
    expected = {
        "schema_version": PREPARED_DATASET_SCHEMA_VERSION,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "unsupported_atomic_numbers": list(unsupported),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise PreparedDatasetError("existing compatible dataset identity differs")
    hashes = manifest.get("artifact_sha256")
    artifact_paths = (dataset, exclusions_path, source_index_path)
    if type(hashes) is not dict or hashes != {
        path.name: sha256_file(path) for path in artifact_paths
    }:
        raise PreparedDatasetError("existing compatible dataset artifact hash differs")
    source_rows = _load_rows(source_index_path)
    exclusion_rows = _load_rows(exclusions_path)
    source_indices = tuple(int(row["source_index"]) for row in source_rows)
    exclusions: tuple[dict[str, object], ...] = tuple(
        {
            "source_index": int(row["source_index"]),
            "chemical_formula": row["chemical_formula"],
            "unsupported_atomic_numbers": row["unsupported_atomic_numbers"],
            "reason": row["reason"],
        }
        for row in exclusion_rows
    )
    return PreparedDataset(
        dataset, exclusions_path, source_index_path, manifest_path,
        source_indices, exclusions, manifest,
    )


def prepare_compatible_dataset(
    *,
    source_path: Path,
    output_path: Path,
    unsupported_atomic_numbers: Sequence[int],
    expected_source_sha256: str,
) -> PreparedDataset:
    """Remove each whole structure containing any unsupported atomic number."""
    source = Path(source_path).expanduser().resolve()
    source_sha = sha256_file(source)
    if source_sha != expected_source_sha256.lower():
        raise PreparedDatasetError(
            f"source SHA-256 mismatch: expected {expected_source_sha256}, got {source_sha}"
        )
    unsupported = tuple(sorted({int(number) for number in unsupported_atomic_numbers}))
    if not unsupported or any(number < 1 for number in unsupported):
        raise PreparedDatasetError("unsupported atomic numbers must be positive")
    outputs = _paths(output_path)
    evidence = tuple(path.exists() for path in outputs)
    if any(evidence):
        if not all(evidence):
            raise PreparedDatasetError("compatible dataset evidence is partial")
        return _reuse(
            outputs,
            source_path=source,
            source_sha256=source_sha,
            unsupported=unsupported,
        )

    structures = _structures(source)
    unsupported_set = set(unsupported)
    compatible: list[Atoms] = []
    source_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for source_index, atoms in enumerate(structures):
        found = sorted(set(map(int, atoms.get_atomic_numbers())) & unsupported_set)
        if found:
            exclusions.append(
                {
                    "source_index": source_index,
                    "chemical_formula": atoms.get_chemical_formula(),
                    "unsupported_atomic_numbers": ";".join(map(str, found)),
                    "reason": "unsupported_checkpoint_element",
                }
            )
            continue
        source_rows.append(
            {"compatible_index": len(compatible), "source_index": source_index}
        )
        compatible.append(atoms)
    if not compatible:
        raise PreparedDatasetError("filter removed every source structure")

    dataset, exclusions_path, source_index_path, manifest_path = outputs
    _atomic_extxyz(dataset, compatible)
    _atomic_bytes(
        exclusions_path,
        _csv_bytes(
            (
                "source_index",
                "chemical_formula",
                "unsupported_atomic_numbers",
                "reason",
            ),
            exclusions,
        ),
    )
    _atomic_bytes(
        source_index_path,
        _csv_bytes(("compatible_index", "source_index"), source_rows),
    )
    manifest: dict[str, Any] = {
        "schema_version": PREPARED_DATASET_SCHEMA_VERSION,
        "source_path": str(source),
        "source_sha256": source_sha,
        "unsupported_atomic_numbers": list(unsupported),
        "source_structures": len(structures),
        "source_atoms": sum(len(atoms) for atoms in structures),
        "compatible_structures": len(compatible),
        "compatible_atoms": sum(len(atoms) for atoms in compatible),
        "excluded_structures": len(exclusions),
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in (dataset, exclusions_path, source_index_path)
        },
    }
    atomic_json_dump(manifest_path, manifest)
    return PreparedDataset(
        dataset,
        exclusions_path,
        source_index_path,
        manifest_path,
        tuple(int(row["source_index"]) for row in source_rows),
        tuple(exclusions),
        manifest,
    )
