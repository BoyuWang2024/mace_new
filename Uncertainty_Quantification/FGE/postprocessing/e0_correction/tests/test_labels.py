from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from ase import Atoms

from Uncertainty_Quantification.FGE.fge.derived_artifacts import (
    build_prediction_shard_signature,
)
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.postprocessing.e0_correction.labels import (
    assert_disjoint_splits,
    build_composition_matrix,
    decontaminate_validation_split,
    iter_aligned_test_label_shards,
    iter_filtered_structures,
    load_validation_labels,
    structure_identity,
    supported_atomic_numbers,
)
from Uncertainty_Quantification.FGE.tests.test_prediction import (
    canonical_prediction_fixture,
)


ATOMIC_NUMBERS = (8, 1)
LABEL_KEYS = {
    "energy": "energy",
    "forces": "forces",
    "atomization_energy": "atomization_energy",
}
TEST_LABEL_KEYS = {
    "energy_reference",
    "atomization_energy",
    "forces_reference",
    "composition",
    "n_atoms",
    "structure_ptr",
    "structure_ids",
}
VALIDATION_LABEL_KEYS = {
    "energy_reference",
    "forces_reference",
    "composition",
    "n_atoms",
    "structure_ptr",
    "structure_ids",
    "source_indices",
    "structures",
}
_MISSING = object()

StructureRecord = tuple[int, Atoms]
PredictionShard = tuple[dict[str, object], dict[str, object]]


def _structure(
    numbers: Sequence[int] = (1,),
    *,
    positions: Any = None,
    cell: Any = None,
    pbc: Any = None,
    energy: Any = 1.0,
    forces: Any = None,
    atomization_energy: Any = -0.5,
    stale_energy: Any = _MISSING,
    stale_forces: Any = _MISSING,
    configuration_id: Any = _MISSING,
    extra_info: Mapping[str, Any] | None = None,
) -> Atoms:
    count = len(numbers)
    if positions is None:
        positions = np.arange(count * 3, dtype=np.float64).reshape(count, 3) / 10.0
    if cell is None:
        cell = np.diag([8.0, 9.0, 10.0])
    if pbc is None:
        pbc = (False, False, False)
    if forces is None:
        forces = (
            np.arange(1, count * 3 + 1, dtype=np.float64).reshape(count, 3)
            / 10.0
        )

    atoms = Atoms(
        numbers=numbers,
        positions=np.asarray(positions, dtype=np.float64),
        cell=np.asarray(cell, dtype=np.float64),
        pbc=np.asarray(pbc, dtype=bool),
    )
    if stale_energy is not _MISSING:
        atoms.info["energy"] = stale_energy
    if stale_forces is not _MISSING:
        atoms.arrays["forces"] = np.asarray(stale_forces)
    if atomization_energy is not _MISSING:
        atoms.info["atomization_energy"] = atomization_energy
    if configuration_id is not _MISSING:
        atoms.info["configuration_id"] = configuration_id
    if extra_info is not None:
        atoms.info.update(extra_info)

    results: dict[str, Any] = {}
    if energy is not _MISSING:
        results["energy"] = energy
    if forces is not _MISSING:
        results["forces"] = forces
    atoms.calc = SimpleNamespace(results=results)
    return atoms


def _filtered(
    structures: Sequence[Atoms], atomic_numbers: Sequence[int] = ATOMIC_NUMBERS
) -> tuple[StructureRecord, ...]:
    return tuple(iter_filtered_structures(iter(structures), atomic_numbers))


def _canonical_payload(n_atoms: Sequence[int]) -> dict[str, object]:
    payload = canonical_prediction_fixture()
    counts = torch.tensor(tuple(n_atoms), dtype=torch.int64)
    structures = len(n_atoms)
    atoms = int(counts.sum().item())
    payload["energy_members"] = torch.arange(
        2 * structures, dtype=torch.float64
    ).reshape(2, structures)
    payload["forces_members"] = torch.zeros((2, atoms, 3), dtype=torch.float64)
    # Deliberately stale zeros: labels must be corrected from extxyz readers.
    payload["energy_reference"] = torch.zeros(structures, dtype=torch.float64)
    payload["forces_reference"] = torch.zeros((atoms, 3), dtype=torch.float64)
    payload["n_atoms"] = counts
    payload["atom_to_structure"] = torch.repeat_interleave(
        torch.arange(structures, dtype=torch.int64), counts
    )
    payload["structure_ptr"] = torch.cat(
        (torch.zeros(1, dtype=torch.int64), counts.cumsum(0))
    )
    return payload


def _prediction_shard(
    *,
    index: int,
    structure_start: int,
    atom_start: int,
    n_atoms: Sequence[int],
) -> PredictionShard:
    payload = _canonical_payload(n_atoms)
    signature = build_prediction_shard_signature(
        dataset="mad_r2scan_test",
        observables=("energy", "forces"),
        member_ids=("member_01", "member_02"),
        shard_index=index,
        structure_start=structure_start,
        structure_stop=structure_start + len(n_atoms),
        atom_start=atom_start,
        atom_stop=atom_start + sum(n_atoms),
        batch_size=2,
    )
    return payload, signature


def _alignment_case() -> tuple[
    tuple[Atoms, ...], tuple[StructureRecord, ...], list[PredictionShard]
]:
    first = _structure(
        (1, 1),
        energy=10.0,
        forces=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        atomization_energy=-1.0,
        stale_energy=-1000.0,
        stale_forces=np.full((2, 3), -1000.0),
        configuration_id="structure_0001",
    )
    rejected_mixed = _structure(
        (1, 17),
        energy=999.0,
        forces=np.full((2, 3), 999.0),
        atomization_energy=999.0,
        configuration_id="unsupported_mixed",
    )
    second = _structure(
        (8,),
        energy=20.0,
        forces=np.array([[0.0, -0.5, 0.0]]),
        atomization_energy=-2.0,
        stale_energy=-2000.0,
        stale_forces=np.full((1, 3), -2000.0),
        configuration_id="structure_0002",
    )
    third = _structure(
        (1, 1, 1),
        energy=30.0,
        forces=np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.75, 0.0], [0.0, 0.0, 0.0]]
        ),
        atomization_energy=-3.0,
        stale_energy=-3000.0,
        stale_forces=np.full((3, 3), -3000.0),
        configuration_id="structure_0003",
    )
    structures = (first, rejected_mixed, second, third)
    records = _filtered(structures)
    shards = [
        _prediction_shard(index=0, structure_start=0, atom_start=0, n_atoms=(2, 1)),
        _prediction_shard(index=1, structure_start=2, atom_start=3, n_atoms=(3,)),
    ]
    return structures, records, shards


def _one_test_label_result(atoms: Atoms) -> dict[str, object]:
    records = _filtered((atoms,))
    shard = _prediction_shard(
        index=0,
        structure_start=0,
        atom_start=0,
        n_atoms=(len(atoms),),
    )
    return tuple(
        iter_aligned_test_label_shards(
            (shard,), records, atomic_numbers=ATOMIC_NUMBERS, keys=LABEL_KEYS
        )
    )[0]


def test_supported_atomic_numbers_preserve_explicit_checkpoint_order() -> None:
    assert supported_atomic_numbers(np.array([8, 1], dtype=np.int64)) == (8, 1)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([], id="empty"),
        pytest.param([1, 1], id="duplicate"),
        pytest.param([0, 1], id="non-positive"),
        pytest.param([1.0, 8], id="non-integer-dtype"),
    ],
)
def test_supported_atomic_numbers_reject_invalid_checkpoint_tables(
    values: Sequence[Any],
) -> None:
    with pytest.raises(HardFailure):
        supported_atomic_numbers(values)


def test_filtering_removes_whole_structures_and_preserves_source_order() -> None:
    supported_first = _structure((1, 8), configuration_id="first")
    unsupported_mixed = _structure((1, 17), configuration_id="mixed")
    supported_last = _structure((8,), configuration_id="last")

    records = _filtered((supported_first, unsupported_mixed, supported_last))

    assert tuple(source_index for source_index, _ in records) == (0, 2)
    assert records[0][1] is supported_first
    assert records[1][1] is supported_last
    assert unsupported_mixed.numbers.tolist() == [1, 17]
    assert len(unsupported_mixed) == 2


def test_filtering_rejects_a_zero_atom_structure() -> None:
    empty = _structure((), forces=np.empty((0, 3)))

    with pytest.raises(HardFailure):
        tuple(iter_filtered_structures((empty,), ATOMIC_NUMBERS))


def test_composition_matrix_uses_checkpoint_column_order_and_float64_counts() -> None:
    structures = (
        _structure((1, 1, 8)),
        _structure((8, 8)),
        _structure((1,)),
    )

    composition = build_composition_matrix(structures, ATOMIC_NUMBERS)

    assert isinstance(composition, np.ndarray)
    assert composition.dtype == np.float64
    np.testing.assert_array_equal(
        composition,
        np.array([[1.0, 2.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    )
    assert np.isfinite(composition).all()
    assert (composition >= 0).all()
    np.testing.assert_array_equal(composition, np.floor(composition))


def test_composition_matrix_rejects_an_unlisted_element() -> None:
    with pytest.raises(HardFailure):
        build_composition_matrix((_structure((1, 17)),), ATOMIC_NUMBERS)


def test_multishard_test_labels_are_aligned_and_calculator_first() -> None:
    structures, records, shards = _alignment_case()
    assert all(
        torch.count_nonzero(payload["energy_reference"]).item() == 0
        and torch.count_nonzero(payload["forces_reference"]).item() == 0
        for payload, _ in shards
    )

    labels = tuple(
        iter_aligned_test_label_shards(
            shards, records, atomic_numbers=ATOMIC_NUMBERS, keys=LABEL_KEYS
        )
    )

    assert len(labels) == 2
    assert all(set(shard) == TEST_LABEL_KEYS for shard in labels)
    first, second = labels
    np.testing.assert_array_equal(first["energy_reference"], [10.0, 20.0])
    np.testing.assert_array_equal(first["atomization_energy"], [-1.0, -2.0])
    np.testing.assert_array_equal(
        first["forces_reference"],
        np.concatenate(
            (structures[0].calc.results["forces"], structures[2].calc.results["forces"])
        ),
    )
    np.testing.assert_array_equal(first["composition"], [[0.0, 2.0], [1.0, 0.0]])
    np.testing.assert_array_equal(first["n_atoms"], [2, 1])
    np.testing.assert_array_equal(first["structure_ptr"], [0, 2, 3])
    assert first["structure_ids"] == ("structure_0001", "structure_0002")

    np.testing.assert_array_equal(second["energy_reference"], [30.0])
    np.testing.assert_array_equal(second["atomization_energy"], [-3.0])
    np.testing.assert_array_equal(
        second["forces_reference"], structures[3].calc.results["forces"]
    )
    np.testing.assert_array_equal(second["composition"], [[0.0, 3.0]])
    np.testing.assert_array_equal(second["n_atoms"], [3])
    np.testing.assert_array_equal(second["structure_ptr"], [0, 3])
    assert second["structure_ids"] == ("structure_0003",)

    for shard in labels:
        for name in (
            "energy_reference",
            "atomization_energy",
            "forces_reference",
            "composition",
        ):
            assert isinstance(shard[name], np.ndarray)
            assert shard[name].dtype == np.float64
            assert np.isfinite(shard[name]).all()
        for name in ("n_atoms", "structure_ptr"):
            assert isinstance(shard[name], np.ndarray)
            assert shard[name].dtype == np.int64

    corrected_energy = np.concatenate([shard["energy_reference"] for shard in labels])
    corrected_forces = np.concatenate([shard["forces_reference"] for shard in labels])
    assert not np.all(corrected_energy == 0.0)
    assert not np.all(corrected_forces == 0.0)


def test_legitimate_all_zero_test_labels_do_not_trigger_the_zero_guard() -> None:
    atoms = _structure(
        (1,),
        energy=0.0,
        forces=np.zeros((1, 3)),
        atomization_energy=0.0,
        configuration_id="zero_labels",
    )

    labels = _one_test_label_result(atoms)

    assert np.all(labels["energy_reference"] == 0.0)
    assert np.all(labels["forces_reference"] == 0.0)
    assert np.all(labels["atomization_energy"] == 0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("energy", _MISSING, id="missing-energy"),
        pytest.param("energy", np.array([1.0, 2.0]), id="energy-shape"),
        pytest.param("energy", float("nan"), id="energy-nan"),
        pytest.param("forces", _MISSING, id="missing-forces"),
        pytest.param("forces", np.zeros((2, 3)), id="forces-shape"),
        pytest.param(
            "forces",
            np.array([[float("inf"), 0.0, 0.0]]),
            id="forces-inf",
        ),
        pytest.param(
            "atomization_energy", _MISSING, id="missing-atomization"
        ),
        pytest.param(
            "atomization_energy",
            np.array([-1.0, -2.0]),
            id="atomization-shape",
        ),
        pytest.param(
            "atomization_energy", float("-inf"), id="atomization-inf"
        ),
    ],
)
def test_test_label_reader_hard_fails_on_invalid_required_fields(
    field: str, value: Any
) -> None:
    values = {
        "energy": 1.0,
        "forces": np.ones((1, 3)),
        "atomization_energy": -0.5,
    }
    values[field] = value
    atoms = _structure((1,), configuration_id="invalid", **values)

    with pytest.raises(HardFailure):
        _one_test_label_result(atoms)


@pytest.mark.parametrize(
    "atomization_energy",
    [
        pytest.param(_MISSING, id="missing"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(np.array([1.0, 2.0]), id="wrong-shape"),
    ],
)
def test_validation_labels_ignore_atomization_and_preserve_forward_structures(
    atomization_energy: Any,
) -> None:
    first = _structure(
        (1, 8),
        energy=4.0,
        forces=np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]),
        atomization_energy=atomization_energy,
        stale_energy=-400.0,
        stale_forces=np.full((2, 3), -400.0),
        configuration_id="val_0001",
    )
    rejected = _structure((1, 17), configuration_id="val_rejected")
    second = _structure(
        (8,),
        energy=5.0,
        forces=np.array([[0.0, 0.0, -0.25]]),
        atomization_energy=atomization_energy,
        stale_energy=-500.0,
        stale_forces=np.full((1, 3), -500.0),
        configuration_id="val_0002",
    )
    records = _filtered((first, rejected, second))

    labels = load_validation_labels(
        records, atomic_numbers=ATOMIC_NUMBERS, keys=LABEL_KEYS
    )

    assert set(labels) == VALIDATION_LABEL_KEYS
    assert labels["structures"][0] is first
    assert labels["structures"][1] is second
    assert labels["source_indices"] == (0, 2)
    assert labels["structure_ids"] == ("val_0001", "val_0002")
    np.testing.assert_array_equal(labels["energy_reference"], [4.0, 5.0])
    np.testing.assert_array_equal(
        labels["forces_reference"],
        np.concatenate((first.calc.results["forces"], second.calc.results["forces"])),
    )
    np.testing.assert_array_equal(
        labels["composition"], [[1.0, 1.0], [1.0, 0.0]]
    )
    np.testing.assert_array_equal(labels["n_atoms"], [2, 1])
    np.testing.assert_array_equal(labels["structure_ptr"], [0, 2, 3])
    assert labels["energy_reference"].dtype == np.float64
    assert labels["forces_reference"].dtype == np.float64
    assert labels["composition"].dtype == np.float64
    assert labels["n_atoms"].dtype == np.int64
    assert labels["structure_ptr"].dtype == np.int64
    assert "atomization_energy" not in labels


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("energy", _MISSING, id="missing-energy"),
        pytest.param("energy", float("inf"), id="energy-inf"),
        pytest.param("forces", _MISSING, id="missing-forces"),
        pytest.param("forces", np.zeros((2, 3)), id="forces-shape"),
        pytest.param(
            "forces",
            np.array([[0.0, float("nan"), 0.0]]),
            id="forces-nan",
        ),
    ],
)
def test_validation_reader_hard_fails_on_invalid_energy_or_forces(
    field: str, value: Any
) -> None:
    values = {"energy": 1.0, "forces": np.ones((1, 3))}
    values[field] = value
    atoms = _structure(
        (1,),
        atomization_energy=float("nan"),
        configuration_id="invalid_val",
        **values,
    )

    with pytest.raises(HardFailure):
        load_validation_labels(
            _filtered((atoms,)), atomic_numbers=ATOMIC_NUMBERS, keys=LABEL_KEYS
        )


def test_validation_reader_rejects_an_empty_filtered_split() -> None:
    with pytest.raises(HardFailure):
        load_validation_labels((), atomic_numbers=ATOMIC_NUMBERS, keys=LABEL_KEYS)


def test_configuration_id_is_the_preferred_nonempty_identity() -> None:
    first_geometry = _structure(
        (1,), configuration_id="configuration_0042", energy=1.0
    )
    changed_geometry_and_labels = _structure(
        (8,),
        positions=np.array([[7.0, 8.0, 9.0]]),
        configuration_id="configuration_0042",
        energy=999.0,
        forces=np.full((1, 3), 999.0),
        atomization_energy=999.0,
    )

    assert structure_identity(first_geometry) == "configuration_0042"
    assert structure_identity(changed_geometry_and_labels) == "configuration_0042"


@pytest.mark.parametrize("invalid_id", ["", "   ", None, 42])
def test_present_configuration_id_must_be_a_nonempty_stable_string(
    invalid_id: Any,
) -> None:
    with pytest.raises(HardFailure):
        structure_identity(_structure((1,), configuration_id=invalid_id))


def test_fallback_identity_excludes_labels_and_paths() -> None:
    positions = np.array(
        [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=np.float64
    )
    cell = np.array(
        [[4.0, 0.1, 0.0], [0.0, 5.0, 0.2], [0.3, 0.0, 6.0]],
        dtype=np.float64,
    )
    base = _structure(
        (1, 8), positions=positions, cell=cell, pbc=(True, False, True)
    )
    labels_and_paths_changed = _structure(
        (1, 8),
        positions=positions,
        cell=cell,
        pbc=(True, False, True),
        energy=-999.0,
        forces=np.full((2, 3), -999.0),
        atomization_energy=-999.0,
        extra_info={
            "source_path": "/different/source.extxyz",
            "source_index": 987654,
        },
    )

    identity = structure_identity(base)

    assert re.fullmatch(r"[0-9a-f]{64}", identity)
    assert structure_identity(labels_and_paths_changed) == identity


def test_fallback_identity_is_raw_byte_sensitive_without_decimal_rounding() -> None:
    positions = np.array(
        [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=np.float64
    )
    cell = np.diag([4.0, 5.0, 6.0]).astype(np.float64)
    base = _structure(
        (1, 8),
        positions=positions,
        cell=cell,
        pbc=(False, False, False),
    )
    next_position = positions.copy()
    next_position[0, 0] = np.nextafter(next_position[0, 0], np.inf)
    next_cell = cell.copy()
    next_cell[2, 2] = np.nextafter(next_cell[2, 2], np.inf)
    variants = (
        _structure(
            (8, 1),
            positions=positions,
            cell=cell,
            pbc=(False, False, False),
        ),
        _structure(
            (1, 8),
            positions=next_position,
            cell=cell,
            pbc=(False, False, False),
        ),
        _structure(
            (1, 8),
            positions=positions,
            cell=next_cell,
            pbc=(False, False, False),
        ),
        _structure(
            (1, 8),
            positions=positions,
            cell=cell,
            pbc=(True, False, False),
        ),
    )

    identity = structure_identity(base)

    assert all(structure_identity(variant) != identity for variant in variants)


def test_decontamination_removes_val_overlap_and_keeps_test_unchanged() -> None:
    test_records = _filtered(
        (
            _structure((1,), configuration_id="shared_0001"),
            _structure((8,), configuration_id="shared_0002"),
        )
    )
    validation_records = _filtered(
        (
            _structure((1, 1), configuration_id="shared_0001"),
            _structure((1, 8), configuration_id="val_unique"),
            _structure((8, 8), configuration_id="shared_0002"),
        )
    )
    test_snapshot = tuple((index, atoms) for index, atoms in test_records)

    cleaned = decontaminate_validation_split(
        test_records, validation_records, expected_removed_count=2
    )

    assert cleaned == (validation_records[1],)
    assert test_records == test_snapshot
    assert tuple(index for index, _ in cleaned) == (1,)
    assert_disjoint_splits(test_records, cleaned)
    with pytest.raises(HardFailure):
        assert_disjoint_splits(test_records, validation_records)


def test_expected_overlap_count_is_optional_and_never_hardcodes_formal_count() -> None:
    test_records = _filtered((_structure((1,), configuration_id="shared"),))
    validation_records = _filtered(
        (
            _structure((8,), configuration_id="shared"),
            _structure((1,), configuration_id="unique"),
        )
    )

    cleaned = decontaminate_validation_split(test_records, validation_records)

    assert cleaned == (validation_records[1],)
    with pytest.raises(HardFailure):
        decontaminate_validation_split(
            test_records, validation_records, expected_removed_count=2
        )


def test_changing_test_labels_does_not_change_fallback_identity_exclusions() -> None:
    geometry = {
        "numbers": (1, 8),
        "positions": np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        "cell": np.diag([4.0, 5.0, 6.0]),
        "pbc": (True, True, False),
    }
    test_original = _structure(
        **geometry,
        energy=1.0,
        forces=np.ones((2, 3)),
        atomization_energy=-1.0,
    )
    test_relabelled = _structure(
        **geometry,
        energy=12345.0,
        forces=np.full((2, 3), -54321.0),
        atomization_energy=777.0,
    )
    validation_overlap = _structure(
        **geometry,
        energy=-2.0,
        forces=np.zeros((2, 3)),
        atomization_energy=float("nan"),
    )
    validation_unique = _structure(
        (1,),
        positions=np.array([[9.0, 9.0, 9.0]]),
        configuration_id="val_unique",
    )
    validation_records = _filtered((validation_overlap, validation_unique))

    original_cleaned = decontaminate_validation_split(
        _filtered((test_original,)), validation_records
    )
    relabelled_cleaned = decontaminate_validation_split(
        _filtered((test_relabelled,)), validation_records
    )

    assert tuple(index for index, _ in original_cleaned) == (1,)
    assert tuple(index for index, _ in relabelled_cleaned) == (1,)


def test_label_loaders_reject_duplicate_identity_within_each_split() -> None:
    duplicate_records = _filtered(
        (
            _structure((1,), configuration_id="duplicate"),
            _structure((8,), configuration_id="duplicate"),
        )
    )

    with pytest.raises(HardFailure):
        load_validation_labels(
            duplicate_records, atomic_numbers=ATOMIC_NUMBERS, keys=LABEL_KEYS
        )

    shard = _prediction_shard(
        index=0,
        structure_start=0,
        atom_start=0,
        n_atoms=(1, 1),
    )
    iterator = iter_aligned_test_label_shards(
        (shard,),
        duplicate_records,
        atomic_numbers=ATOMIC_NUMBERS,
        keys=LABEL_KEYS,
    )
    with pytest.raises(HardFailure):
        next(iterator)


def test_a_late_noncanonical_payload_hard_fails_during_full_consumption() -> None:
    _, records, shards = _alignment_case()
    shards[1][0]["energy_reference"] = shards[1][0]["energy_reference"].float()

    with pytest.raises(HardFailure):
        tuple(
            iter_aligned_test_label_shards(
                shards,
                records,
                atomic_numbers=ATOMIC_NUMBERS,
                keys=LABEL_KEYS,
            )
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "reversed-shards",
        "duplicate-shard-index",
        "global-structure-gap",
        "global-atom-gap",
        "payload-vs-extxyz-atom-count",
        "signature-member-mismatch",
        "signature-observable-mismatch",
        "missing-extxyz-structure",
        "trailing-extxyz-structure",
    ],
)
def test_alignment_mismatches_hard_fail_during_full_consumption(
    mutation: str,
) -> None:
    _, records, shards = _alignment_case()
    if mutation == "reversed-shards":
        shards.reverse()
    elif mutation == "duplicate-shard-index":
        shards[1][1]["shard_index"] = 0
    elif mutation == "global-structure-gap":
        shards[1][1]["structure_start"] = 3
        shards[1][1]["structure_stop"] = 4
    elif mutation == "global-atom-gap":
        shards[1][1]["atom_start"] = 4
        shards[1][1]["atom_stop"] = 7
    elif mutation == "payload-vs-extxyz-atom-count":
        payload = shards[0][0]
        payload["n_atoms"] = torch.tensor([1, 2], dtype=torch.int64)
        payload["atom_to_structure"] = torch.tensor(
            [0, 1, 1], dtype=torch.int64
        )
        payload["structure_ptr"] = torch.tensor(
            [0, 1, 3], dtype=torch.int64
        )
    elif mutation == "signature-member-mismatch":
        shards[1][1]["member_ids"] = [
            "member_01",
            "member_02",
            "member_03",
        ]
    elif mutation == "signature-observable-mismatch":
        shards[1][1]["observables"] = ["energy", "forces", "stress"]
    elif mutation == "missing-extxyz-structure":
        records = records[:-1]
    else:
        records = records + (
            (4, _structure((1,), configuration_id="unexpected_trailing")),
        )

    with pytest.raises(HardFailure):
        tuple(
            iter_aligned_test_label_shards(
                shards,
                records,
                atomic_numbers=ATOMIC_NUMBERS,
                keys=LABEL_KEYS,
            )
        )
