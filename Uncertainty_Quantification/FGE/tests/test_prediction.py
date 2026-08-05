from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge import prediction
from Uncertainty_Quantification.FGE.fge.errors import HardFailure


def canonical_prediction_fixture() -> dict[str, object]:
    return {
        "schema_version": "fge.prediction.v1",
        "split": "test",
        "member_source": "raw",
        "member_ids": ["member_01", "member_02"],
        "observables": ["energy", "forces"],
        "energy_members": torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float64),
        "forces_members": torch.zeros((2, 3, 3), dtype=torch.float64),
        "energy_reference": torch.tensor([1.5, 2.5], dtype=torch.float64),
        "forces_reference": torch.zeros((3, 3), dtype=torch.float64),
        "n_atoms": torch.tensor([2, 1], dtype=torch.int64),
        "atom_to_structure": torch.tensor([0, 0, 1], dtype=torch.int64),
        "structure_ptr": torch.tensor([0, 2, 3], dtype=torch.int64),
    }


def test_prediction_payload_accepts_exact_canonical_schema() -> None:
    shape = prediction.validate_prediction_payload(canonical_prediction_fixture())
    assert (shape.members, shape.structures, shape.atoms) == (2, 2, 3)


def test_prediction_payload_rejects_noncanonical_atom_mapping() -> None:
    payload = canonical_prediction_fixture()
    payload["atom_to_structure"] = torch.tensor([0, 1, 0])
    with pytest.raises(HardFailure, match="canonical atom mapping"):
        prediction.validate_prediction_payload(payload)


def test_prediction_payload_rejects_extra_key_or_float32() -> None:
    payload = canonical_prediction_fixture()
    payload["source_path"] = "/old/data.xyz"
    with pytest.raises(HardFailure, match="keys"):
        prediction.validate_prediction_payload(payload)
    payload = canonical_prediction_fixture()
    payload["energy_members"] = payload["energy_members"].float()
    with pytest.raises(HardFailure, match="float64"):
        prediction.validate_prediction_payload(payload)


class FakeConfig:
    def __init__(self, root: Path) -> None:
        self.output_dir = root
        self._sections = {"prediction": {"compute_stress": False}}

    def section(self, name: str):
        return self._sections[name]


def test_prediction_writes_raw_available_ema_not_generated(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "manifest.json").write_text(
        json.dumps({"members": [{"member_id": "member_01"}, {"member_id": "member_02"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(prediction, "run_preflight", lambda *_args: {"status": "PASS"})
    monkeypatch.setattr(
        prediction,
        "_generate_prediction_payload",
        lambda *_args: canonical_prediction_fixture(),
    )
    manifest_path = prediction.predict_members(FakeConfig(tmp_path))
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["branches"] == {"raw": "available", "ema": "not_generated"}
    assert torch.load(tmp_path / "prediction" / "test_raw.pt", weights_only=True)[
        "member_source"
    ] == "raw"
