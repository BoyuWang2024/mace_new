from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from Uncertainty_Quantification.FGE.fge import extxyz_standard


KEYS = {"energy": "energy", "forces": "forces", "stress": "stress"}


def _atoms(
    *,
    info: dict[str, Any] | None = None,
    arrays: dict[str, Any] | None = None,
    results: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        info={} if info is None else info,
        arrays={} if arrays is None else arrays,
        calc=SimpleNamespace(results=results),
    )


@pytest.mark.parametrize(
    ("reader", "key", "array", "stale_value", "calculator_value"),
    [
        (extxyz_standard.read_energy, "energy", False, -100.0, -1.5),
        (
            extxyz_standard.read_forces,
            "forces",
            True,
            np.zeros((2, 3)),
            np.full((2, 3), 2.0),
        ),
        (
            extxyz_standard.read_stress,
            "stress",
            False,
            np.zeros(6),
            np.arange(6, dtype=float),
        ),
    ],
)
def test_standard_reader_prefers_calculator_result(
    reader: Any,
    key: str,
    array: bool,
    stale_value: Any,
    calculator_value: Any,
) -> None:
    atoms = _atoms(
        info={} if array else {key: stale_value},
        arrays={key: stale_value} if array else {},
        results={key: calculator_value},
    )
    np.testing.assert_equal(reader(atoms, KEYS), calculator_value)


@pytest.mark.parametrize(
    ("reader", "key", "array", "fallback_value"),
    [
        (extxyz_standard.read_energy, "energy", False, -2.5),
        (extxyz_standard.read_forces, "forces", True, np.ones((2, 3))),
        (extxyz_standard.read_stress, "stress", False, np.ones(6)),
    ],
)
@pytest.mark.parametrize(
    "calculator_value",
    [pytest.param(None, id="none"), pytest.param(..., id="missing")],
)
def test_standard_reader_falls_back_when_calculator_result_is_unavailable(
    reader: Any,
    key: str,
    array: bool,
    fallback_value: Any,
    calculator_value: Any,
) -> None:
    results = {} if calculator_value is ... else {key: calculator_value}
    atoms = _atoms(
        info={} if array else {key: fallback_value},
        arrays={key: fallback_value} if array else {},
        results=results,
    )
    np.testing.assert_equal(reader(atoms, KEYS), fallback_value)


def test_standard_readers_fall_back_for_non_mapping_calculator_results() -> None:
    atoms = _atoms(
        info={"energy": -2.5, "stress": np.ones(6)},
        arrays={"forces": np.ones((2, 3))},
        results=object(),
    )
    assert extxyz_standard.read_energy(atoms, KEYS) == -2.5
    np.testing.assert_equal(
        extxyz_standard.read_forces(atoms, KEYS), np.ones((2, 3))
    )
    np.testing.assert_equal(extxyz_standard.read_stress(atoms, KEYS), np.ones(6))


def test_standard_readers_fall_back_to_configured_custom_keys() -> None:
    keys = {"energy": "REF_energy", "forces": "REF_forces", "stress": "REF_stress"}
    atoms = _atoms(
        info={"REF_energy": -3.5, "REF_stress": np.full(6, 3.0)},
        arrays={"REF_forces": np.full((2, 3), 2.0)},
        results={"energy": -300.0, "forces": np.zeros((2, 3)), "stress": np.zeros(6)},
    )
    assert extxyz_standard.read_energy(atoms, keys) == -3.5
    np.testing.assert_equal(
        extxyz_standard.read_forces(atoms, keys), np.full((2, 3), 2.0)
    )
    np.testing.assert_equal(
        extxyz_standard.read_stress(atoms, keys), np.full(6, 3.0)
    )


@pytest.mark.parametrize(
    ("reader", "key", "array", "calculator_value"),
    [
        (extxyz_standard.read_energy, "energy", False, float("nan")),
        (
            extxyz_standard.read_forces,
            "forces",
            True,
            np.full((2, 3), float("inf")),
        ),
        (
            extxyz_standard.read_stress,
            "stress",
            False,
            np.full(6, float("-inf")),
        ),
    ],
)
def test_standard_reader_preserves_non_finite_calculator_result(
    reader: Any,
    key: str,
    array: bool,
    calculator_value: Any,
) -> None:
    fallback_value = np.zeros_like(calculator_value)
    atoms = _atoms(
        info={} if array else {key: fallback_value},
        arrays={key: fallback_value} if array else {},
        results={key: calculator_value},
    )
    np.testing.assert_equal(reader(atoms, KEYS), calculator_value)


def test_read_atomization_energy_reads_its_scalar_field() -> None:
    atoms = _atoms(
        info={"atomization_energy": -0.75, "energy": -101.0},
        arrays={"atomization_energy": np.array([999.0])},
        results={"energy": -102.0},
    )
    assert extxyz_standard.read_atomization_energy(atoms) == -0.75


def test_read_atomization_energy_supports_a_custom_scalar_key() -> None:
    atoms = _atoms(info={"formation_energy": -0.25}, results={})
    assert extxyz_standard.read_atomization_energy(atoms, key="formation_energy") == -0.25
