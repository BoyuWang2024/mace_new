from __future__ import annotations

import csv
import json
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms

from confidence_head.identity import sha256_file
from confidence_head.workflows.prepare_external_dataset import (
    PreparedDatasetError,
    prepare_compatible_dataset,
)


def _labeled(symbols: str) -> Atoms:
    atoms = Atoms(symbols, positions=np.zeros((len(Atoms(symbols)), 3)))
    atoms.info["REF_energy"] = float(len(atoms))
    atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
    return atoms


def _source(path: Path) -> Path:
    ase.io.write(
        path,
        [_labeled("H"), _labeled("PoH"), _labeled("He"), _labeled("Rn")],
        format="extxyz",
    )
    return path


def test_prepare_compatible_dataset_removes_complete_unsupported_structures(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "mad.xyz")

    result = prepare_compatible_dataset(
        source_path=source,
        output_path=tmp_path / "mad-compatible.xyz",
        unsupported_atomic_numbers=(84, 86),
        expected_source_sha256=sha256_file(source),
    )

    kept = ase.io.read(result.dataset_path, index=":")
    assert [atoms.get_chemical_formula() for atoms in kept] == ["H", "He"]
    assert result.source_indices == (0, 2)
    assert [row["source_index"] for row in result.exclusions] == [1, 3]
    assert result.manifest["source_structures"] == 4
    assert result.manifest["compatible_structures"] == 2
    assert result.manifest["excluded_structures"] == 2
    assert result.manifest["compatible_atoms"] == 2

    with result.exclusions_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["unsupported_atomic_numbers"] == "84"
    assert rows[1]["unsupported_atomic_numbers"] == "86"


def test_prepare_compatible_dataset_reuses_matching_complete_artifacts(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "mad.xyz")
    arguments = {
        "source_path": source,
        "output_path": tmp_path / "mad-compatible.xyz",
        "unsupported_atomic_numbers": (84, 86),
        "expected_source_sha256": sha256_file(source),
    }
    first = prepare_compatible_dataset(**arguments)
    before = {path: path.stat().st_mtime_ns for path in first.output_paths}

    second = prepare_compatible_dataset(**arguments)

    assert second == first
    assert before == {path: path.stat().st_mtime_ns for path in first.output_paths}


def test_prepare_compatible_dataset_rejects_partial_evidence(tmp_path: Path) -> None:
    source = _source(tmp_path / "mad.xyz")
    output = tmp_path / "mad-compatible.xyz"
    output.write_text("partial\n", encoding="utf-8")

    with pytest.raises(PreparedDatasetError, match="partial"):
        prepare_compatible_dataset(
            source_path=source,
            output_path=output,
            unsupported_atomic_numbers=(84, 86),
            expected_source_sha256=sha256_file(source),
        )
