from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read, write

from Uncertainty_Quantification.LLPR.llpr.artifacts import sha256_file
from Uncertainty_Quantification.LLPR.llpr.dataset_filter import (
    filter_neighborless_extxyz,
    missing_neighbor_indices,
)


def _write_source(path: Path) -> None:
    retained = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    retained.info["structure_id"] = "keep"
    excluded = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [20, 0, 0]])
    excluded.info["structure_id"] = "drop"
    write(path, [retained, excluded], format="extxyz")


def test_missing_neighbor_excludes_whole_structure() -> None:
    atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [20, 0, 0]])

    assert missing_neighbor_indices(atoms, 6.0) == (2,)


def test_missing_neighbor_detects_single_atom() -> None:
    assert missing_neighbor_indices(Atoms("H", positions=[[0, 0, 0]]), 6.0) == (0,)


def test_missing_neighbor_excludes_periodic_single_atom() -> None:
    atoms = Atoms(
        "H",
        positions=[[0, 0, 0]],
        cell=[5, 5, 5],
        pbc=True,
    )

    assert missing_neighbor_indices(atoms, 6.0) == (0,)


def test_filter_writes_hash_complete_audit(tmp_path: Path) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)

    report = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert report["source_sha256"] == sha256_file(source)
    assert report["output_sha256"] == sha256_file(output)
    assert report["total_structures"] == (
        report["retained_structures"] + report["excluded_structures"]
    )
    assert report["total_structures"] == 2
    assert report["retained_structures"] == 1
    assert report["excluded_structures"] == 1
    assert [atoms.info["structure_id"] for atoms in read(output, ":")] == ["keep"]

    audit_data = json.loads(audit.read_text(encoding="utf-8"))
    assert audit_data == report
    assert audit_data["excluded"] == [
        {
            "source_index": 1,
            "structure_id": "drop",
            "num_atoms": 3,
            "missing_neighbor_indices": [2],
            "pbc": [False, False, False],
            "elements": ["H"],
            "reason": "missing_neighbor_within_cutoff",
        }
    ]


def test_filter_is_deterministic_on_rerun(tmp_path: Path) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)

    first = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)
    first_output_sha256 = sha256_file(output)
    first_audit_sha256 = sha256_file(audit)
    second = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert second == first
    assert sha256_file(output) == first_output_sha256
    assert sha256_file(audit) == first_audit_sha256


def test_filter_cleans_temporary_output_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(
        "Uncertainty_Quantification.LLPR.llpr.dataset_filter.ase_write", fail_write
    )

    with pytest.raises(OSError, match="simulated write failure"):
        filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert not output.exists()
    assert not audit.exists()
    assert list(tmp_path.glob(".filtered.extxyz.*.tmp")) == []


def test_missing_neighbor_respects_partial_periodic_boundary_neighbors() -> None:
    atoms = Atoms(
        "H2",
        positions=[[0.2, 0, 0], [9.8, 0, 0]],
        cell=[10, 20, 20],
        pbc=[True, False, False],
    )

    assert missing_neighbor_indices(atoms, 1.0) == ()


def test_filter_restores_both_previous_artifacts_when_audit_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)
    output.write_text("old output", encoding="utf-8")
    audit.write_text("old audit", encoding="utf-8")

    def fail_second_publish(source_path: Path, destination: Path) -> Path:
        if destination == audit and source_path.name.endswith(".tmp"):
            raise OSError("simulated audit publish failure")
        return source_path.replace(destination)

    monkeypatch.setattr(
        "Uncertainty_Quantification.LLPR.llpr.dataset_filter._replace_file",
        fail_second_publish,
    )

    with pytest.raises(OSError, match="simulated audit publish failure"):
        filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert output.read_text(encoding="utf-8") == "old output"
    assert audit.read_text(encoding="utf-8") == "old audit"
    assert list(tmp_path.glob(".filtered.extxyz.*.tmp")) == []
    assert list(tmp_path.glob(".audit.json.*.tmp")) == []
    assert list(tmp_path.glob(".filtered.extxyz.*.bak")) == []
    assert list(tmp_path.glob(".audit.json.*.bak")) == []
