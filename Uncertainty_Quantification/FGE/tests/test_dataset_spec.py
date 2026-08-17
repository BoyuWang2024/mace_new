from __future__ import annotations

from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.dataset_spec import DatasetSpec
from Uncertainty_Quantification.FGE.fge.errors import HardFailure


def _data_file(tmp_path: Path, name: str = "sample.extxyz") -> Path:
    path = tmp_path / name
    path.write_text("", encoding="utf-8")
    return path


def test_dataset_spec_builds_source_neutral_manifest(tmp_path: Path) -> None:
    path = _data_file(tmp_path)
    spec = DatasetSpec(
        label="matpes_test",
        path=path,
        energy_key="REF_energy",
        forces_key="REF_forces",
        stress_key="REF_stress",
        head_name="Default",
        compute_stress=True,
        batch_size=32,
        shard_size=128,
    )

    assert spec.observables == ("energy", "forces", "stress")
    assert spec.required == frozenset({"energy", "forces", "stress"})
    assert spec.keys == {
        "energy": "REF_energy",
        "forces": "REF_forces",
        "stress": "REF_stress",
        "head": "Default",
    }
    assert spec.neutral_manifest() == {
        "schema_version": "fge.dataset.v1",
        "dataset": "matpes_test",
        "observables": ["energy", "forces", "stress"],
        "keys": spec.keys,
        "batch_size": 32,
        "shard_size": 128,
    }


def test_dataset_spec_accepts_xyz_as_extxyz_container(tmp_path: Path) -> None:
    spec = DatasetSpec(label="mad_test", path=_data_file(tmp_path, "mad-test.xyz"))

    assert spec.observables == ("energy", "forces")
    assert spec.required == frozenset({"energy", "forces"})


@pytest.mark.parametrize("label", ["", "Matpes", "bad label", "../dataset", "_hidden"])
def test_dataset_spec_rejects_invalid_logical_label(tmp_path: Path, label: str) -> None:
    with pytest.raises(HardFailure, match="dataset label"):
        DatasetSpec(label=label, path=_data_file(tmp_path))


@pytest.mark.parametrize("name", ["data.pt", "data.csv", "data.json", "data.xyz.gz"])
def test_dataset_spec_rejects_non_extxyz_container(tmp_path: Path, name: str) -> None:
    with pytest.raises(HardFailure, match=".xyz or .extxyz"):
        DatasetSpec(label="case", path=_data_file(tmp_path, name))


def test_dataset_spec_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(HardFailure, match="does not exist"):
        DatasetSpec(label="case", path=tmp_path / "missing.extxyz")


@pytest.mark.parametrize(
    ("field", "value"),
    [("batch_size", 0), ("batch_size", True), ("shard_size", -1), ("shard_size", 1.5)],
)
def test_dataset_spec_rejects_invalid_positive_integer(
    tmp_path: Path, field: str, value: object
) -> None:
    arguments = {field: value}
    with pytest.raises(HardFailure, match=field):
        DatasetSpec(label="case", path=_data_file(tmp_path), **arguments)
