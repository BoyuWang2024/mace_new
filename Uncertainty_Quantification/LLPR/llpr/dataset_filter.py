"""Filter extxyz structures that contain atoms without a cutoff neighbour."""

from __future__ import annotations

import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any
from uuid import uuid4

from ase import Atoms
from ase.io import iread
from ase.io import read as ase_read
from ase.io import write as ase_write
from matscipy.neighbours import neighbour_list

from .artifacts import sha256_file


FILTER_PREDICATE_VERSION = "distinct-base-atom-within-cutoff-v2"
FILTER_AUDIT_SCHEMA_VERSION = "mad-neighbor-filter-audit-v2"


def _missing_neighbor_analysis(
    atoms: Atoms, cutoff: float
) -> tuple[tuple[int, ...], int]:
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    neighbour_atoms = atoms.copy()
    neighbour_atoms.set_cell(neighbour_atoms.cell.complete())
    i_indices, j_indices = neighbour_list("ij", neighbour_atoms, cutoff)
    distinct_mask = i_indices != j_indices
    ignored_self_image_edges = int((~distinct_mask).sum())
    atoms_with_neighbours = {
        int(index) for index in i_indices[distinct_mask]
    } | {
        int(index) for index in j_indices[distinct_mask]
    }
    missing = tuple(
        index for index in range(len(atoms)) if index not in atoms_with_neighbours
    )
    return missing, ignored_self_image_edges


def missing_neighbor_indices(atoms: Atoms, cutoff: float) -> tuple[int, ...]:
    """Return atoms without a distinct-base-atom neighbour within ``cutoff``."""
    return _missing_neighbor_analysis(atoms, cutoff)[0]


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0



def _derive_source_filter_audit(
    source_path: Path, cutoff: float
) -> tuple[int, list[int], list[dict[str, Any]], int]:
    excluded: list[dict[str, Any]] = []
    retained_indices: list[int] = []
    ignored_self_image_edges = 0
    total_structures = 0

    for source_index, atoms in enumerate(iread(source_path, index=":", format="extxyz")):
        total_structures += 1
        existing_source_index = atoms.info.get("source_index")
        if existing_source_index is not None and existing_source_index != source_index:
            raise ValueError(
                f"structure {source_index} has conflicting source_index "
                f"{existing_source_index}"
            )
        missing_indices, ignored_edges = _missing_neighbor_analysis(atoms, cutoff)
        ignored_self_image_edges += ignored_edges
        if missing_indices:
            excluded.append(_audit_record(atoms, source_index, missing_indices))
        else:
            retained_indices.append(source_index)

    return total_structures, retained_indices, excluded, ignored_self_image_edges


def _valid_cached_report(
    source_path: Path, output_path: Path, audit_path: Path, cutoff: float
) -> dict[str, Any] | None:
    if not output_path.is_file() or not audit_path.is_file():
        return None
    try:
        report = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict):
        return None
    required_keys = {
        "audit_schema_version",
        "predicate_version",
        "source_sha256",
        "output_sha256",
        "cutoff",
        "total_structures",
        "retained_structures",
        "excluded_structures",
        "ignored_self_image_edges",
        "excluded",
    }
    if set(report) != required_keys:
        return None
    recorded_cutoff = report["cutoff"]
    if (
        not isinstance(recorded_cutoff, (int, float))
        or isinstance(recorded_cutoff, bool)
        or not math.isfinite(recorded_cutoff)
        or report["audit_schema_version"] != FILTER_AUDIT_SCHEMA_VERSION
        or report["predicate_version"] != FILTER_PREDICATE_VERSION
        or recorded_cutoff != cutoff
        or report["source_sha256"] != sha256_file(source_path)
        or report["output_sha256"] != sha256_file(output_path)
    ):
        return None
    total = report["total_structures"]
    retained = report["retained_structures"]
    excluded_count = report["excluded_structures"]
    ignored_edges = report["ignored_self_image_edges"]
    excluded = report["excluded"]
    if (
        not all(
            _is_nonnegative_integer(value)
            for value in (total, retained, excluded_count, ignored_edges)
        )
        or retained + excluded_count != total
        or not isinstance(excluded, list)
        or len(excluded) != excluded_count
    ):
        return None
    excluded_indices: list[int] = []
    expected_exclusion_keys = {
        "source_index",
        "structure_id",
        "num_atoms",
        "missing_neighbor_indices",
        "pbc",
        "elements",
        "reason",
    }
    for record in excluded:
        if not isinstance(record, dict) or set(record) != expected_exclusion_keys:
            return None
        index = record["source_index"]
        if (
            not isinstance(index, Integral)
            or isinstance(index, bool)
            or not 0 <= index < total
            or not isinstance(record["num_atoms"], Integral)
            or isinstance(record["num_atoms"], bool)
            or record["num_atoms"] <= 0
            or not isinstance(record["structure_id"], str)
            or not isinstance(record["missing_neighbor_indices"], list)
            or not record["missing_neighbor_indices"]
            or any(
                not isinstance(atom_index, Integral)
                or isinstance(atom_index, bool)
                or not 0 <= atom_index < record["num_atoms"]
                for atom_index in record["missing_neighbor_indices"]
            )
            or record["missing_neighbor_indices"]
            != sorted(set(record["missing_neighbor_indices"]))
            or not isinstance(record["pbc"], list)
            or len(record["pbc"]) != 3
            or any(not isinstance(value, bool) for value in record["pbc"])
            or not isinstance(record["elements"], list)
            or not record["elements"]
            or record["elements"] != sorted(set(record["elements"]))
            or any(not isinstance(element, str) for element in record["elements"])
            or record["reason"] != "missing_neighbor_within_cutoff"
        ):
            return None
        excluded_indices.append(int(index))
    if excluded_indices != sorted(set(excluded_indices)):
        return None
    (
        expected_total,
        expected_retained_indices,
        expected_excluded,
        expected_ignored_edges,
    ) = _derive_source_filter_audit(source_path, cutoff)
    if (
        total != expected_total
        or retained != len(expected_retained_indices)
        or excluded_count != len(expected_excluded)
        or ignored_edges != expected_ignored_edges
        or excluded != expected_excluded
    ):
        return None
    try:
        retained_atoms = ase_read(output_path, index=":", format="extxyz")
    except Exception:
        return None
    retained_indices = [atoms.info.get("source_index") for atoms in retained_atoms]
    if (
        len(retained_indices) != retained
        or any(
            not isinstance(index, Integral) or isinstance(index, bool)
            for index in retained_indices
        )
    ):
        return None
    retained_indices = [int(index) for index in retained_indices]
    if (
        retained_indices != expected_retained_indices
        or retained_indices != sorted(set(retained_indices))
        or set(retained_indices) | set(excluded_indices) != set(range(total))
        or set(retained_indices) & set(excluded_indices)
    ):
        return None
    return report


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _backup_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.bak")


def _replace_file(source: Path, destination: Path) -> Path:
    return source.replace(destination)


def _restore_file(target: Path, backup: Path | None, published: bool) -> None:
    if backup is not None:
        target.unlink(missing_ok=True)
        _replace_file(backup, target)
    elif published:
        target.unlink(missing_ok=True)


def _publish_pair(
    temporary_output: Path,
    temporary_audit: Path,
    output_path: Path,
    audit_path: Path,
) -> None:
    output_backup = _backup_path(output_path) if output_path.exists() else None
    audit_backup = _backup_path(audit_path) if audit_path.exists() else None
    output_backup_staged = False
    audit_backup_staged = False
    output_published = False
    audit_published = False

    try:
        if output_backup is not None:
            _replace_file(output_path, output_backup)
            output_backup_staged = True
        if audit_backup is not None:
            _replace_file(audit_path, audit_backup)
            audit_backup_staged = True
        _replace_file(temporary_output, output_path)
        output_published = True
        _replace_file(temporary_audit, audit_path)
        audit_published = True
    except Exception:
        _restore_file(audit_path, audit_backup if audit_backup_staged else None, audit_published)
        _restore_file(output_path, output_backup if output_backup_staged else None, output_published)
        raise
    finally:
        if output_backup_staged:
            output_backup.unlink(missing_ok=True)
        if audit_backup_staged:
            audit_backup.unlink(missing_ok=True)


def _audit_record(
    atoms: Atoms, source_index: int, missing_indices: tuple[int, ...]
) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "structure_id": str(atoms.info.get("structure_id", source_index)),
        "num_atoms": len(atoms),
        "missing_neighbor_indices": list(missing_indices),
        "pbc": [bool(value) for value in atoms.pbc],
        "elements": sorted(set(atoms.get_chemical_symbols())),
        "reason": "missing_neighbor_within_cutoff",
    }


def filter_neighborless_extxyz(
    source: str | Path,
    output: str | Path,
    audit: str | Path,
    *,
    cutoff: float,
) -> dict[str, Any]:
    """Stream-filter an extxyz file and atomically emit its audit report."""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    source_path = Path(source)
    output_path = Path(output)
    audit_path = Path(audit)
    cached_report = _valid_cached_report(source_path, output_path, audit_path, cutoff)
    if cached_report is not None:
        return cached_report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = _temporary_path(output_path)
    temporary_audit = _temporary_path(audit_path)
    excluded: list[dict[str, Any]] = []
    total_structures = 0
    retained_structures = 0
    ignored_self_image_edges = 0

    try:
        for source_index, atoms in enumerate(
            iread(source_path, index=":", format="extxyz")
        ):
            total_structures += 1
            existing_source_index = atoms.info.get("source_index")
            if existing_source_index is not None and existing_source_index != source_index:
                raise ValueError(
                    f"structure {source_index} has conflicting source_index {existing_source_index}"
                )
            missing_indices, ignored_edges = _missing_neighbor_analysis(atoms, cutoff)
            ignored_self_image_edges += ignored_edges
            if missing_indices:
                excluded.append(_audit_record(atoms, source_index, missing_indices))
                continue
            output_atoms = atoms.copy()
            output_atoms.info["source_index"] = source_index
            ase_write(
                temporary_output,
                output_atoms,
                format="extxyz",
                append=retained_structures > 0,
            )
            retained_structures += 1

        if retained_structures == 0:
            temporary_output.touch()
        verified_structures = sum(
            1 for _ in iread(temporary_output, index=":", format="extxyz")
        )
        if verified_structures != retained_structures:
            raise ValueError("temporary extxyz verification count mismatch")

        report: dict[str, Any] = {
            "audit_schema_version": FILTER_AUDIT_SCHEMA_VERSION,
            "predicate_version": FILTER_PREDICATE_VERSION,
            "source_sha256": sha256_file(source_path),
            "output_sha256": sha256_file(temporary_output),
            "cutoff": cutoff,
            "ignored_self_image_edges": ignored_self_image_edges,
            "total_structures": total_structures,
            "retained_structures": retained_structures,
            "excluded_structures": len(excluded),
            "excluded": excluded,
        }
        temporary_audit.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if json.loads(temporary_audit.read_text(encoding="utf-8")) != report:
            raise ValueError("temporary audit verification mismatch")

        _publish_pair(temporary_output, temporary_audit, output_path, audit_path)
        return report
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_audit.unlink(missing_ok=True)
