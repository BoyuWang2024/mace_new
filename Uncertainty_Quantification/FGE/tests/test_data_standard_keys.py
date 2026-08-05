from pathlib import Path

from Uncertainty_Quantification.FGE.fge.data import load_extxyz


def test_standard_extxyz_keys_promoted_to_ase_calculator_are_accepted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "standard.extxyz"
    path.write_text(
        "1\n"
        'Properties=species:S:1:pos:R:3:forces:R:3 energy=-1.0 '
        'stress="1 2 3 4 5 6 7 8 9" pbc="F F F"\n'
        "H 0 0 0 0.1 0.2 0.3\n",
        encoding="utf-8",
    )
    configurations = load_extxyz(
        path,
        keys={"energy": "energy", "forces": "forces", "stress": "stress", "head": "head"},
        required={"energy", "forces", "stress"},
    )
    assert len(configurations) == 1
