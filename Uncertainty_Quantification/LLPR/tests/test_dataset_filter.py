from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from Uncertainty_Quantification.LLPR.llpr import dataset_filter
from Uncertainty_Quantification.LLPR.llpr.artifacts import sha256_file
from Uncertainty_Quantification.LLPR.llpr.dataset_filter import (
    FILTER_PREDICATE_VERSION,
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


def test_multiatom_periodic_self_image_does_not_rescue_isolated_atom() -> None:
    atoms = Atoms(
        "H3",
        positions=[[0, 0, 0], [1, 0, 0], [9, 9, 0]],
        cell=[2, 20, 20],
        pbc=[True, False, False],
    )

    assert missing_neighbor_indices(atoms, 6.0) == (2,)


def test_periodic_image_of_different_base_atom_is_a_neighbor() -> None:
    atoms = Atoms(
        "H2",
        positions=[[0, 0, 0], [9, 0, 0]],
        cell=[10, 20, 20],
        pbc=[True, False, False],
    )

    assert missing_neighbor_indices(atoms, 6.0) == ()


def test_filter_writes_hash_complete_audit(tmp_path: Path) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)

    report = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert report["source_sha256"] == sha256_file(source)
    assert report["output_sha256"] == sha256_file(output)
    assert report["predicate_version"] == FILTER_PREDICATE_VERSION
    assert report["ignored_self_image_edges"] == 0
    assert report["total_structures"] == (
        report["retained_structures"] + report["excluded_structures"]
    )
    assert report["total_structures"] == 2
    assert report["retained_structures"] == 1
    assert report["excluded_structures"] == 1
    retained = read(output, ":")
    assert [atoms.info["structure_id"] for atoms in retained] == ["keep"]
    assert [atoms.info["source_index"] for atoms in retained] == [0]

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


def test_filter_preserves_reference_results_and_structure_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    retained = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    retained.info.update(
        structure_id="labelled-keep",
        provenance="reference labels must survive verbatim",
    )
    retained.new_array("site_id", np.array([17, 23], dtype=np.int64))
    retained.calc = SinglePointCalculator(
        retained,
        energy=-12.345678901234,
        free_energy=-12.456789012345,
        forces=np.array(
            [[0.125, -0.25, 0.375], [-0.125, 0.25, -0.375]],
            dtype=np.float64,
        ),
        stress=np.array([1.25, 2.5, 3.75, 0.125, 0.25, 0.5], dtype=np.float64),
    )
    excluded = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [20, 0, 0]])
    excluded.info["structure_id"] = "labelled-drop"
    excluded.calc = SinglePointCalculator(
        excluded,
        energy=99.0,
        forces=np.full((3, 3), 9.0, dtype=np.float64),
    )
    write(source, [retained, excluded], format="extxyz")

    source_bytes = source.read_bytes()
    source_frames = read(source, ":", format="extxyz")
    source_info = [deepcopy(atoms.info) for atoms in source_frames]
    source_arrays = [
        {name: values.copy() for name, values in atoms.arrays.items()}
        for atoms in source_frames
    ]
    source_results = [
        {name: np.asarray(value).copy() for name, value in atoms.calc.results.items()}
        for atoms in source_frames
    ]

    first_report = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert source.read_bytes() == source_bytes
    for frame_index, atoms in enumerate(source_frames):
        assert atoms.info == source_info[frame_index]
        assert atoms.arrays.keys() == source_arrays[frame_index].keys()
        for name, expected in source_arrays[frame_index].items():
            np.testing.assert_array_equal(atoms.arrays[name], expected)
        assert atoms.calc is not None
        assert atoms.calc.results.keys() == source_results[frame_index].keys()
        for name, expected in source_results[frame_index].items():
            np.testing.assert_array_equal(atoms.calc.results[name], expected)

    filtered = read(output, ":", format="extxyz")
    assert len(filtered) == 1
    filtered_atoms = filtered[0]
    assert filtered_atoms.info["source_index"] == 0
    assert filtered_atoms.info["structure_id"] == "labelled-keep"
    assert filtered_atoms.info["provenance"] == retained.info["provenance"]
    np.testing.assert_array_equal(filtered_atoms.arrays["site_id"], [17, 23])
    assert filtered_atoms.calc is not None
    assert filtered_atoms.calc.results.keys() == source_results[0].keys()
    for name, expected in source_results[0].items():
        actual = filtered_atoms.calc.results[name]
        assert np.asarray(actual).dtype.kind == "f"
        assert np.asarray(actual).shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    assert all(atoms.info["structure_id"] != "labelled-drop" for atoms in filtered)

    def fail_publish(*args: object, **kwargs: object) -> None:
        raise AssertionError("canonical labelled output must be a strict cache hit")

    monkeypatch.setattr(dataset_filter, "_publish_pair", fail_publish)
    assert filter_neighborless_extxyz(source, output, audit, cutoff=6.0) == first_report


def test_calculator_result_copy_has_no_mutable_aliases() -> None:
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=np.array(-2.0, dtype=np.float32),
        free_energy=np.array(-2.1, dtype=np.float64),
        magmom=np.array(1.25, dtype=np.float32),
        forces=np.array(
            [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=np.float64
        ),
        stress=np.arange(6, dtype=np.float64),
        born_effective_charges=np.arange(18, dtype=np.float64).reshape(2, 3, 3),
    )
    atoms.calc.results["dielectric_tensor"] = [
        [1.0, 0.1, 0.2],
        [0.1, 2.0, 0.3],
        [0.2, 0.3, 3.0],
    ]
    original_values = {
        name: np.asarray(value).copy() for name, value in atoms.calc.results.items()
    }

    copied = dataset_filter._copy_atoms_with_calculator_results(atoms)

    assert copied is not atoms
    assert copied.calc is not None
    assert copied.calc is not atoms.calc
    assert copied.calc.results is not atoms.calc.results
    assert copied.calc.results.keys() == atoms.calc.results.keys()
    assert copied.arrays.keys() == atoms.arrays.keys()
    for name in atoms.arrays:
        assert not np.shares_memory(copied.arrays[name], atoms.arrays[name])
    for name in ("energy", "free_energy", "magmom"):
        assert copied.calc.results[name] is not atoms.calc.results[name]
        assert not isinstance(copied.calc.results[name], np.ndarray)
        np.testing.assert_array_equal(copied.calc.results[name], original_values[name])
    for name in ("forces", "stress", "born_effective_charges", "dielectric_tensor"):
        assert copied.calc.results[name] is not atoms.calc.results[name]
        assert not np.shares_memory(
            copied.calc.results[name], atoms.calc.results[name]
        )
        assert copied.calc.results[name].dtype == np.asarray(
            atoms.calc.results[name]
        ).dtype
        assert copied.calc.results[name].shape == np.asarray(
            atoms.calc.results[name]
        ).shape
        np.testing.assert_array_equal(copied.calc.results[name], original_values[name])

    atoms.calc.results["energy"][...] = -9.0
    atoms.calc.results["forces"][0, 0] = 91.0
    atoms.calc.results["born_effective_charges"][0, 0, 0] = 92.0
    atoms.calc.results["dielectric_tensor"][0][0] = 93.0
    np.testing.assert_array_equal(copied.calc.results["energy"], original_values["energy"])
    np.testing.assert_array_equal(copied.calc.results["forces"], original_values["forces"])
    np.testing.assert_array_equal(
        copied.calc.results["born_effective_charges"],
        original_values["born_effective_charges"],
    )
    np.testing.assert_array_equal(
        copied.calc.results["dielectric_tensor"], original_values["dielectric_tensor"]
    )

    copied.calc.results["energy"] = -8.0
    copied.calc.results["stress"][0] = 81.0
    copied.calc.results["born_effective_charges"][1, 2, 2] = 82.0
    assert atoms.calc.results["energy"] == -9.0
    np.testing.assert_array_equal(atoms.calc.results["stress"], original_values["stress"])
    assert atoms.calc.results["born_effective_charges"][1, 2, 2] == (
        original_values["born_effective_charges"][1, 2, 2]
    )
    copied.calc.results["dielectric_tensor"][2, 2] = 83.0
    assert atoms.calc.results["dielectric_tensor"][2][2] == (
        original_values["dielectric_tensor"][2, 2]
    )
    copied.positions[0, 0] = 84.0
    assert atoms.positions[0, 0] == 0.0


def test_calculator_result_copy_rejects_unknown_property() -> None:
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-2.0)
    atoms.calc.results["unknown_reference_label"] = np.array([1.0])

    with pytest.raises(
        ValueError,
        match="unsupported calculator result property 'unknown_reference_label'",
    ):
        dataset_filter._copy_atoms_with_calculator_results(atoms)


def test_calculator_result_copy_rejects_value_that_cannot_be_copied() -> None:
    class UncopyableResult:
        def __deepcopy__(self, memo: dict[int, object]) -> object:
            raise TypeError("cannot copy this value")

    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-2.0)
    atoms.calc.results["energy"] = UncopyableResult()

    with pytest.raises(
        ValueError,
        match="calculator result 'energy' cannot be copied safely",
    ):
        dataset_filter._copy_atoms_with_calculator_results(atoms)


@pytest.mark.parametrize(
    "field, value",
    [
        ("predicate_version", "distinct-base-atom-within-cutoff-v1"),
        ("cutoff", 5.0),
        ("source_sha256", "0" * 64),
    ],
)
def test_filter_recomputes_when_existing_audit_is_not_a_valid_v2_cache(
    tmp_path: Path, field: str, value: object
) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)
    original = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)
    stale_audit = dict(original)
    stale_audit[field] = value
    audit.write_text(json.dumps(stale_audit), encoding="utf-8")

    report = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert report == json.loads(audit.read_text(encoding="utf-8"))
    assert report["source_sha256"] == sha256_file(source)
    assert report["predicate_version"] == FILTER_PREDICATE_VERSION
    assert report["cutoff"] == 6.0


def test_filter_cache_hit_rederives_source_without_republishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)
    expected = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)
    output_mtime_ns = output.stat().st_mtime_ns
    original_iread = dataset_filter.iread
    source_restreams = 0

    def count_source_restream(*args: object, **kwargs: object) -> object:
        nonlocal source_restreams
        if args[0] == source:
            source_restreams += 1
        return original_iread(*args, **kwargs)

    monkeypatch.setattr(dataset_filter, "iread", count_source_restream)

    def fail_publish(*args: object, **kwargs: object) -> None:
        raise AssertionError("valid cache hit must not republish artifacts")

    monkeypatch.setattr(
        "Uncertainty_Quantification.LLPR.llpr.dataset_filter._publish_pair",
        fail_publish,
    )
    assert filter_neighborless_extxyz(source, output, audit, cutoff=6.0) == expected




def test_filter_recomputes_when_source_changes_after_initial_cache_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)
    filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    original_sha256_file = dataset_filter.sha256_file
    source_was_changed = False

    def mutate_source_after_initial_sha(path: str | Path) -> str:
        nonlocal source_was_changed
        digest = original_sha256_file(path)
        if Path(path) == source and not source_was_changed:
            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            source_was_changed = True
        return digest

    monkeypatch.setattr(dataset_filter, "sha256_file", mutate_source_after_initial_sha)
    report = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert source_was_changed
    assert report["source_sha256"] == original_sha256_file(source)
    assert json.loads(audit.read_text(encoding="utf-8"))["source_sha256"] == report[
        "source_sha256"
    ]


def test_filter_recomputes_after_coordinated_output_and_audit_tampering(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)
    filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    forged_output = read(source, ":")[1]
    forged_output.info["source_index"] = 1
    write(output, [forged_output], format="extxyz")
    forged_audit = json.loads(audit.read_text(encoding="utf-8"))
    forged_audit["output_sha256"] = sha256_file(output)
    forged_audit["excluded"] = [
        {
            "source_index": 0,
            "structure_id": "keep",
            "num_atoms": 2,
            "missing_neighbor_indices": [0],
            "pbc": [False, False, False],
            "elements": ["H"],
            "reason": "missing_neighbor_within_cutoff",
        }
    ]
    audit.write_text(json.dumps(forged_audit), encoding="utf-8")

    report = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert report["excluded"] == [
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
    assert [atoms.info["source_index"] for atoms in read(output, ":")] == [0]



def test_filter_recomputes_after_retained_content_and_audit_tampering(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)
    filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    tampered_output = read(output, ":")[0]
    tampered_output.positions[1, 0] = 4.0
    write(output, [tampered_output], format="extxyz")
    tampered_audit = json.loads(audit.read_text(encoding="utf-8"))
    tampered_audit["output_sha256"] = sha256_file(output)
    audit.write_text(json.dumps(tampered_audit), encoding="utf-8")

    report = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    restored = read(output, ":")
    assert restored[0].positions[1, 0] == 1.0
    assert report["output_sha256"] == sha256_file(output)


def test_filter_audits_periodic_singleton_self_image_edges(tmp_path: Path) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    write(source, [atoms], format="extxyz")

    report = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert report["retained_structures"] == 0
    assert report["excluded_structures"] == 1
    assert report["ignored_self_image_edges"] == 6
    assert json.loads(audit.read_text(encoding="utf-8"))["ignored_self_image_edges"] == 6


def test_filter_recomputes_when_cached_output_sha_is_tampered(tmp_path: Path) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)
    filter_neighborless_extxyz(source, output, audit, cutoff=6.0)
    output.write_text("tampered output", encoding="utf-8")

    report = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert report["output_sha256"] == sha256_file(output)
    assert report["excluded"] == [
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


def test_filter_rejects_conflicting_existing_source_index(tmp_path: Path) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    retained = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    retained.info["source_index"] = 7
    write(source, [retained], format="extxyz")

    with pytest.raises(
        ValueError,
        match="structure 0 has conflicting source_index 7",
    ):
        filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert not output.exists()
    assert not audit.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("*.bak")) == []


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


def test_filter_preserves_previous_artifacts_when_output_backup_staging_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)
    output.write_text("old output", encoding="utf-8")
    audit.write_text("old audit", encoding="utf-8")

    def fail_output_backup(source_path: Path, destination: Path) -> Path:
        if source_path == output and destination.name.endswith(".bak"):
            raise OSError("simulated output backup failure")
        return source_path.replace(destination)

    monkeypatch.setattr(
        "Uncertainty_Quantification.LLPR.llpr.dataset_filter._replace_file",
        fail_output_backup,
    )

    with pytest.raises(OSError, match="simulated output backup failure"):
        filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert output.read_text(encoding="utf-8") == "old output"
    assert audit.read_text(encoding="utf-8") == "old audit"
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("*.bak")) == []


def test_filter_preserves_previous_artifacts_when_audit_backup_staging_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.extxyz"
    output = tmp_path / "filtered.extxyz"
    audit = tmp_path / "audit.json"
    _write_source(source)
    output.write_text("old output", encoding="utf-8")
    audit.write_text("old audit", encoding="utf-8")

    def fail_audit_backup(source_path: Path, destination: Path) -> Path:
        if source_path == audit and destination.name.endswith(".bak"):
            raise OSError("simulated audit backup failure")
        return source_path.replace(destination)

    monkeypatch.setattr(
        "Uncertainty_Quantification.LLPR.llpr.dataset_filter._replace_file",
        fail_audit_backup,
    )

    with pytest.raises(OSError, match="simulated audit backup failure"):
        filter_neighborless_extxyz(source, output, audit, cutoff=6.0)

    assert output.read_text(encoding="utf-8") == "old output"
    assert audit.read_text(encoding="utf-8") == "old audit"
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("*.bak")) == []
