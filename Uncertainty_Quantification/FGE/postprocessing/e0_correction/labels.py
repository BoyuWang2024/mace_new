"""Strict label extraction, structure identity, and split alignment for E0 correction."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Iterator, Mapping, Sequence
from numbers import Integral
from typing import Any, Callable

import numpy as np

from Uncertainty_Quantification.FGE.fge.derived_artifacts import (
    build_prediction_shard_signature,
)
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.extxyz_standard import (
    read_atomization_energy,
    read_energy,
    read_forces,
)
from Uncertainty_Quantification.FGE.fge.prediction import (
    validate_prediction_payload,
)


_SIGNATURE_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "observables",
        "member_ids",
        "shard_index",
        "structure_start",
        "structure_stop",
        "atom_start",
        "atom_stop",
        "batch_size",
    }
)
_SIGNATURE_SCHEMA = "fge.derived-prediction-signature.v1"


def _fail(message: str) -> None:
    raise HardFailure(message)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        try:
            value = value.numpy()
        except (TypeError, RuntimeError, ValueError):
            pass
    return np.asarray(value)


def supported_atomic_numbers(
    checkpoint_atomic_numbers: Sequence[int] | Any,
) -> tuple[int, ...]:
    """Validate and preserve the explicit checkpoint element-column order."""

    try:
        values = _as_numpy(checkpoint_atomic_numbers)
    except Exception as exc:
        raise HardFailure("checkpoint atomic_numbers cannot be read") from exc
    if values.ndim != 1 or values.size == 0:
        _fail("checkpoint atomic_numbers must be a nonempty one-dimensional table")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype, np.integer
    ):
        _fail("checkpoint atomic_numbers must use an integer dtype")
    result = tuple(int(value) for value in values.tolist())
    if any(value <= 0 for value in result):
        _fail("checkpoint atomic_numbers must contain only positive integers")
    if len(result) != len(set(result)):
        _fail("checkpoint atomic_numbers must be unique")
    return result


def _atomic_numbers(atoms: Any, *, context: str) -> np.ndarray:
    try:
        count = len(atoms)
        raw = _as_numpy(atoms.numbers)
    except Exception as exc:
        raise HardFailure(f"{context} atomic numbers cannot be read") from exc
    if count < 1:
        _fail(f"{context} cannot be an empty structure")
    if (
        raw.ndim != 1
        or raw.shape != (count,)
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        _fail(f"{context} atomic numbers are invalid")
    numbers = np.ascontiguousarray(raw, dtype=np.dtype("<i8"))
    if (numbers <= 0).any():
        _fail(f"{context} atomic numbers must be positive")
    return numbers


def iter_filtered_structures(
    structures: Iterable[Any], atomic_numbers: Sequence[int] | Any
) -> Iterator[tuple[int, Any]]:
    """Yield supported structures whole, retaining their original source indices."""

    supported = frozenset(supported_atomic_numbers(atomic_numbers))
    try:
        iterator = iter(structures)
    except TypeError as exc:
        raise HardFailure("structures must be iterable") from exc
    for source_index, atoms in enumerate(iterator):
        numbers = _atomic_numbers(atoms, context=f"structure {source_index}")
        if all(int(number) in supported for number in numbers):
            yield source_index, atoms


def build_composition_matrix(
    structures: Iterable[Any], atomic_numbers: Sequence[int] | Any
) -> np.ndarray:
    """Count elements in explicit checkpoint order as a finite float64 matrix."""

    columns = supported_atomic_numbers(atomic_numbers)
    column_by_number = {number: index for index, number in enumerate(columns)}
    try:
        ordered_structures = tuple(structures)
    except TypeError as exc:
        raise HardFailure("structures must be iterable") from exc
    composition = np.zeros(
        (len(ordered_structures), len(columns)), dtype=np.float64
    )
    for row, atoms in enumerate(ordered_structures):
        numbers = _atomic_numbers(atoms, context=f"structure {row}")
        for number_value in numbers:
            number = int(number_value)
            column = column_by_number.get(number)
            if column is None:
                _fail(
                    f"structure {row} contains unsupported atomic number {number}"
                )
            composition[row, column] += 1.0
    if (
        not np.isfinite(composition).all()
        or (composition < 0).any()
        or not np.equal(composition, np.floor(composition)).all()
    ):
        _fail("composition matrix is not a finite nonnegative integer matrix")
    return composition


def _finite_float64(
    value: Any,
    *,
    context: str,
    expected_shape: tuple[int, ...] | None = None,
    scalar: bool = False,
) -> np.ndarray:
    if value is None:
        _fail(f"{context} is missing")
    try:
        raw = _as_numpy(value)
    except Exception as exc:
        raise HardFailure(f"{context} cannot be read") from exc
    if (
        np.issubdtype(raw.dtype, np.bool_)
        or np.issubdtype(raw.dtype, np.complexfloating)
        or not np.issubdtype(raw.dtype, np.number)
    ):
        _fail(f"{context} must be numeric")
    if scalar and raw.shape != ():
        _fail(f"{context} must be scalar")
    if expected_shape is not None and raw.shape != expected_shape:
        _fail(f"{context} has invalid shape")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HardFailure(f"{context} cannot be converted to float64") from exc
    if not np.isfinite(result).all():
        _fail(f"{context} contains NaN or Inf")
    return np.ascontiguousarray(result, dtype=np.float64)


def _geometry_float64(
    value: Any, *, context: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    return _finite_float64(
        value, context=context, expected_shape=expected_shape
    )


def _pbc_uint8(atoms: Any) -> np.ndarray:
    try:
        raw = _as_numpy(atoms.pbc)
    except Exception as exc:
        raise HardFailure("structure pbc cannot be read") from exc
    if raw.shape != (3,):
        _fail("structure pbc has invalid shape")
    if not (
        np.issubdtype(raw.dtype, np.bool_)
        or np.issubdtype(raw.dtype, np.integer)
    ):
        _fail("structure pbc must contain booleans")
    integers = np.asarray(raw, dtype=np.int64)
    if not np.isin(integers, (0, 1)).all():
        _fail("structure pbc must contain booleans")
    return np.ascontiguousarray(integers, dtype=np.uint8)


def _hash_array(
    digest: Any,
    *,
    tag: bytes,
    dtype_tag: bytes,
    values: np.ndarray,
) -> None:
    digest.update(struct.pack("<I", len(tag)))
    digest.update(tag)
    digest.update(struct.pack("<I", len(dtype_tag)))
    digest.update(dtype_tag)
    digest.update(struct.pack("<I", values.ndim))
    for dimension in values.shape:
        digest.update(struct.pack("<Q", int(dimension)))
    raw = values.tobytes(order="C")
    digest.update(struct.pack("<Q", len(raw)))
    digest.update(raw)


def structure_identity(atoms: Any) -> str:
    """Return configuration_id, or an exact geometry-only SHA-256 identity."""

    numbers = _atomic_numbers(atoms, context="structure")
    info = getattr(atoms, "info", None)
    if not isinstance(info, Mapping):
        _fail("structure info must be a mapping")
    if "configuration_id" in info:
        value = info["configuration_id"]
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
        ):
            _fail("configuration_id must be a nonempty stable string")
        return value

    count = len(numbers)
    positions = _geometry_float64(
        getattr(atoms, "positions", None),
        context="structure positions",
        expected_shape=(count, 3),
    )
    cell_object = getattr(atoms, "cell", None)
    cell_value = getattr(cell_object, "array", cell_object)
    cell = _geometry_float64(
        cell_value,
        context="structure cell",
        expected_shape=(3, 3),
    )
    pbc = _pbc_uint8(atoms)

    digest = hashlib.sha256()
    _hash_array(
        digest,
        tag=b"atomic_numbers",
        dtype_tag=b"<i8",
        values=np.ascontiguousarray(numbers, dtype=np.dtype("<i8")),
    )
    _hash_array(
        digest,
        tag=b"positions",
        dtype_tag=b"<f8",
        values=np.ascontiguousarray(positions, dtype=np.dtype("<f8")),
    )
    _hash_array(
        digest,
        tag=b"cell",
        dtype_tag=b"<f8",
        values=np.ascontiguousarray(cell, dtype=np.dtype("<f8")),
    )
    _hash_array(
        digest,
        tag=b"pbc",
        dtype_tag=b"u1",
        values=np.ascontiguousarray(pbc, dtype=np.uint8),
    )
    return digest.hexdigest()


def _record(
    value: Any,
    *,
    split: str,
    previous_source_index: int | None,
) -> tuple[int, Any]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        _fail(f"{split} structure record must be a (source_index, atoms) pair")
    source_index, atoms = value
    if (
        isinstance(source_index, bool)
        or not isinstance(source_index, Integral)
        or int(source_index) < 0
    ):
        _fail(f"{split} source index must be a nonnegative integer")
    normalized_index = int(source_index)
    if (
        previous_source_index is not None
        and normalized_index <= previous_source_index
    ):
        _fail(f"{split} source indices must be strictly increasing")
    _atomic_numbers(atoms, context=f"{split} structure {normalized_index}")
    return normalized_index, atoms


def _materialize_records(
    records: Iterable[Any], *, split: str
) -> tuple[tuple[int, Any], ...]:
    try:
        iterator = iter(records)
    except TypeError as exc:
        raise HardFailure(f"{split} records must be iterable") from exc
    result: list[tuple[int, Any]] = []
    previous: int | None = None
    for raw_record in iterator:
        record = _record(
            raw_record,
            split=split,
            previous_source_index=previous,
        )
        result.append(record)
        previous = record[0]
    return tuple(result)


def _unique_structure_ids(
    records: Sequence[tuple[int, Any]], *, split: str
) -> tuple[str, ...]:
    identities: list[str] = []
    seen: set[str] = set()
    for _, atoms in records:
        identity = structure_identity(atoms)
        if identity in seen:
            _fail(f"{split} split contains duplicate structure identity")
        seen.add(identity)
        identities.append(identity)
    return tuple(identities)


def decontaminate_validation_split(
    test_records: Iterable[Any],
    validation_records: Iterable[Any],
    expected_removed_count: int | None = None,
) -> tuple[tuple[int, Any], ...]:
    """Exclude every validation structure whose identity occurs in test."""

    test = _materialize_records(test_records, split="test")
    validation = _materialize_records(validation_records, split="validation")
    test_ids = _unique_structure_ids(test, split="test")
    validation_ids = _unique_structure_ids(validation, split="validation")
    if expected_removed_count is not None and (
        isinstance(expected_removed_count, bool)
        or not isinstance(expected_removed_count, Integral)
        or int(expected_removed_count) < 0
    ):
        _fail("expected_removed_count must be a nonnegative integer")

    test_identity_set = set(test_ids)
    kept = tuple(
        record
        for record, identity in zip(validation, validation_ids)
        if identity not in test_identity_set
    )
    removed_count = len(validation) - len(kept)
    if (
        expected_removed_count is not None
        and removed_count != int(expected_removed_count)
    ):
        _fail("validation overlap removal count does not match expectation")
    assert_disjoint_splits(test, kept)
    return kept


def assert_disjoint_splits(
    test_records: Iterable[Any], validation_records: Iterable[Any]
) -> None:
    """Hard-fail on within-split duplicates or residual cross-split overlap."""

    test = _materialize_records(test_records, split="test")
    validation = _materialize_records(validation_records, split="validation")
    test_ids = _unique_structure_ids(test, split="test")
    validation_ids = _unique_structure_ids(validation, split="validation")
    if set(test_ids).intersection(validation_ids):
        _fail("test and validation splits contain overlapping structure identity")


def _label_keys(
    keys: Mapping[str, str], *, require_atomization: bool
) -> dict[str, str]:
    if not isinstance(keys, Mapping):
        _fail("label keys must be a mapping")
    names = ("energy", "forces")
    if require_atomization:
        names += ("atomization_energy",)
    normalized: dict[str, str] = {}
    for name in names:
        value = keys.get(name)
        if not isinstance(value, str) or not value:
            _fail(f"label key {name} must be a nonempty string")
        normalized[name] = value
    return normalized


def _call_reader(
    reader: Callable[..., Any],
    atoms: Any,
    *args: Any,
    context: str,
    **kwargs: Any,
) -> Any:
    try:
        return reader(atoms, *args, **kwargs)
    except HardFailure:
        raise
    except Exception as exc:
        raise HardFailure(f"{context} cannot be read") from exc


def _read_energy_value(
    atoms: Any, keys: Mapping[str, str], *, context: str
) -> float:
    value = _call_reader(
        read_energy,
        atoms,
        keys,
        context=f"{context} energy",
    )
    array = _finite_float64(value, context=f"{context} energy", scalar=True)
    return float(array.reshape(-1)[0])


def _read_forces_value(
    atoms: Any, keys: Mapping[str, str], *, context: str
) -> np.ndarray:
    count = len(_atomic_numbers(atoms, context=context))
    value = _call_reader(
        read_forces,
        atoms,
        keys,
        context=f"{context} forces",
    )
    return _finite_float64(
        value,
        context=f"{context} forces",
        expected_shape=(count, 3),
    )


def _read_atomization_value(
    atoms: Any, key: str, *, context: str
) -> float:
    value = _call_reader(
        read_atomization_energy,
        atoms,
        key=key,
        context=f"{context} atomization_energy",
    )
    array = _finite_float64(
        value,
        context=f"{context} atomization_energy",
        scalar=True,
    )
    return float(array.reshape(-1)[0])


def _n_atoms_and_pointer(
    structures: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([len(atoms) for atoms in structures], dtype=np.int64)
    pointer = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            counts.cumsum(dtype=np.int64),
        )
    )
    return counts, pointer


def load_validation_labels(
    records: Iterable[Any],
    *,
    atomic_numbers: Sequence[int] | Any,
    keys: Mapping[str, str],
) -> dict[str, Any]:
    """Load val E/F labels without consulting atomization_energy."""

    columns = supported_atomic_numbers(atomic_numbers)
    label_keys = _label_keys(keys, require_atomization=False)
    ordered = _materialize_records(records, split="validation")
    if not ordered:
        _fail("validation split cannot be empty")
    structure_ids = _unique_structure_ids(ordered, split="validation")
    structures = tuple(atoms for _, atoms in ordered)
    source_indices = tuple(source_index for source_index, _ in ordered)

    energies: list[float] = []
    force_rows: list[np.ndarray] = []
    for index, atoms in enumerate(structures):
        context = f"validation structure {source_indices[index]}"
        energy = _read_energy_value(atoms, label_keys, context=context)
        forces = _read_forces_value(atoms, label_keys, context=context)
        energies.append(energy)
        force_rows.append(forces)

    energy_reference = np.asarray(energies, dtype=np.float64)
    forces_reference = np.concatenate(force_rows, axis=0).astype(
        np.float64, copy=False
    )
    composition = build_composition_matrix(structures, columns)
    n_atoms, structure_ptr = _n_atoms_and_pointer(structures)
    return {
        "energy_reference": energy_reference,
        "forces_reference": forces_reference,
        "composition": composition,
        "n_atoms": n_atoms,
        "structure_ptr": structure_ptr,
        "structure_ids": structure_ids,
        "source_indices": source_indices,
        "structures": structures,
    }


def _validated_signature(signature: Any) -> dict[str, object]:
    if not isinstance(signature, Mapping) or set(signature) != _SIGNATURE_KEYS:
        _fail("prediction shard signature has invalid fields")
    if signature.get("schema_version") != _SIGNATURE_SCHEMA:
        _fail("prediction shard signature has invalid schema")
    try:
        canonical = build_prediction_shard_signature(
            dataset=signature["dataset"],
            observables=signature["observables"],
            member_ids=signature["member_ids"],
            shard_index=signature["shard_index"],
            structure_start=signature["structure_start"],
            structure_stop=signature["structure_stop"],
            atom_start=signature["atom_start"],
            atom_stop=signature["atom_stop"],
            batch_size=signature["batch_size"],
        )
        if dict(signature) != canonical:
            _fail("prediction shard signature is not canonical")
        return canonical
    except HardFailure:
        raise
    except Exception as exc:
        raise HardFailure("prediction shard signature is invalid") from exc


def _validated_payload(payload: Any) -> Any:
    try:
        return validate_prediction_payload(payload)
    except HardFailure:
        raise
    except Exception as exc:
        raise HardFailure("canonical prediction payload is invalid") from exc


def _payload_array(payload: Mapping[str, Any], name: str) -> np.ndarray:
    try:
        return np.asarray(payload[name].detach().cpu().numpy())
    except Exception as exc:
        raise HardFailure(f"canonical field {name} cannot be inspected") from exc


def iter_aligned_test_label_shards(
    shards: Iterable[Any],
    records: Iterable[Any],
    *,
    atomic_numbers: Sequence[int] | Any,
    keys: Mapping[str, str],
) -> Iterator[dict[str, Any]]:
    """Stream corrected test labels aligned to canonical prediction shards."""

    columns = supported_atomic_numbers(atomic_numbers)
    label_keys = _label_keys(keys, require_atomization=True)
    try:
        shard_iterator = iter(shards)
        record_iterator = iter(records)
    except TypeError as exc:
        raise HardFailure("prediction shards and records must be iterable") from exc

    expected_shard_index = 0
    expected_structure_start = 0
    expected_atom_start = 0
    previous_source_index: int | None = None
    baseline_metadata: tuple[Any, ...] | None = None
    seen_structure_ids: set[str] = set()
    saw_shard = False

    for raw_shard in shard_iterator:
        saw_shard = True
        if not isinstance(raw_shard, (tuple, list)) or len(raw_shard) != 2:
            _fail("prediction shard must be a (payload, signature) pair")
        payload, raw_signature = raw_shard
        shape = _validated_payload(payload)
        signature = _validated_signature(raw_signature)
        if signature["observables"] != ["energy", "forces"]:
            _fail("E0 prediction shard observables must be energy and forces")
        if payload["observables"] != signature["observables"]:
            _fail("prediction payload and signature observables are not aligned")
        if payload["member_ids"] != signature["member_ids"]:
            _fail("prediction payload and signature members are not aligned")

        metadata = (
            signature["dataset"],
            tuple(signature["observables"]),
            tuple(signature["member_ids"]),
            signature["batch_size"],
        )
        if baseline_metadata is None:
            baseline_metadata = metadata
        elif metadata != baseline_metadata:
            _fail("prediction shard metadata changed across shards")

        if (
            signature["shard_index"] != expected_shard_index
            or signature["structure_start"] != expected_structure_start
            or signature["atom_start"] != expected_atom_start
        ):
            _fail("prediction shard indices or global ranges are not contiguous")
        if (
            signature["structure_stop"] - signature["structure_start"]
            != shape.structures
            or signature["atom_stop"] - signature["atom_start"]
            != shape.atoms
        ):
            _fail("prediction shard ranges do not match canonical payload shape")

        shard_records: list[tuple[int, Any]] = []
        shard_ids: list[str] = []
        for _ in range(shape.structures):
            try:
                raw_record = next(record_iterator)
            except StopIteration as exc:
                raise HardFailure(
                    "filtered test stream ended before prediction shards"
                ) from exc
            record = _record(
                raw_record,
                split="test",
                previous_source_index=previous_source_index,
            )
            previous_source_index = record[0]
            identity = structure_identity(record[1])
            if identity in seen_structure_ids:
                _fail("test split contains duplicate structure identity")
            seen_structure_ids.add(identity)
            shard_records.append(record)
            shard_ids.append(identity)

        structures = tuple(atoms for _, atoms in shard_records)
        extxyz_n_atoms, expected_pointer = _n_atoms_and_pointer(structures)
        payload_n_atoms = _payload_array(payload, "n_atoms")
        payload_pointer = _payload_array(payload, "structure_ptr")
        if (
            not np.array_equal(payload_n_atoms, extxyz_n_atoms)
            or not np.array_equal(payload_pointer, expected_pointer)
        ):
            _fail("canonical atom counts do not align with filtered test stream")
        if int(extxyz_n_atoms.sum()) != shape.atoms:
            _fail("filtered test atom total does not match canonical payload")

        energies: list[float] = []
        atomization_energies: list[float] = []
        force_rows: list[np.ndarray] = []
        for source_index, atoms in shard_records:
            context = f"test structure {source_index}"
            energy = _read_energy_value(atoms, label_keys, context=context)
            forces = _read_forces_value(atoms, label_keys, context=context)
            atomization = _read_atomization_value(
                atoms,
                label_keys["atomization_energy"],
                context=context,
            )
            energies.append(energy)
            atomization_energies.append(atomization)
            force_rows.append(forces)

        energy_reference = np.asarray(energies, dtype=np.float64)
        atomization_energy = np.asarray(
            atomization_energies, dtype=np.float64
        )
        forces_reference = np.concatenate(force_rows, axis=0).astype(
            np.float64, copy=False
        )
        composition = build_composition_matrix(structures, columns)

        yield {
            "energy_reference": energy_reference,
            "atomization_energy": atomization_energy,
            "forces_reference": forces_reference,
            "composition": composition,
            "n_atoms": extxyz_n_atoms,
            "structure_ptr": expected_pointer,
            "structure_ids": tuple(shard_ids),
        }

        expected_shard_index += 1
        expected_structure_start = int(signature["structure_stop"])
        expected_atom_start = int(signature["atom_stop"])

    if not saw_shard:
        _fail("prediction shard stream cannot be empty")
    try:
        next(record_iterator)
    except StopIteration:
        pass
    else:
        _fail("filtered test stream contains trailing structures")
