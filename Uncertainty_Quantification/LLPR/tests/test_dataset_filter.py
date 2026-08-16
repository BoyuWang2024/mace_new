from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase import Atoms
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
