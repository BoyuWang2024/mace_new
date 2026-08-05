from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.data import load_extxyz
from Uncertainty_Quantification.FGE.fge.errors import HardFailure


KEYS = {
    "energy": "REF_energy",
    "forces": "REF_forces",
    "stress": "REF_stress",
    "head": "head",
}


def _write_energy_forces(path: Path) -> Path:
    path.write_text(
        "2\n"
        'Properties=species:S:1:pos:R:3:REF_forces:R:3 REF_energy=1.0 pbc="F F F"\n'
        "H 0 0 0 0.1 0 0\nH 0 0 1 -0.1 0 0\n"
        "1\n"
        'Properties=species:S:1:pos:R:3:REF_forces:R:3 REF_energy=2.0 pbc="F F F"\n'
        "H 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    return path


def test_predict_contract_accepts_missing_stress(tmp_path: Path) -> None:
    configurations = load_extxyz(
        _write_energy_forces(tmp_path / "data.xyz"),
        keys=KEYS,
        required={"energy", "forces"},
    )
    assert len(configurations) == 2


def test_train_contract_rejects_missing_stress(tmp_path: Path) -> None:
    with pytest.raises(HardFailure, match="stress"):
        load_extxyz(
            _write_energy_forces(tmp_path / "data.xyz"),
            keys=KEYS,
            required={"energy", "forces", "stress"},
        )


def test_data_contract_rejects_wrong_force_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.xyz"
    path.write_text(
        "1\n"
        'Properties=species:S:1:pos:R:3:REF_forces:R:1 REF_energy=1.0 pbc="F F F"\n'
        "H 0 0 0 0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(HardFailure, match="forces"):
        load_extxyz(path, keys=KEYS, required={"energy", "forces"})
