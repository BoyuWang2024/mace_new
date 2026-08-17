from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.derived_artifacts import (
    DerivedLayout,
    build_prediction_shard_signature,
    verify_prediction_shard_if_present,
    write_prediction_shard,
)
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.tests.test_prediction import (
    canonical_prediction_fixture,
)


def _signature() -> dict[str, object]:
    return build_prediction_shard_signature(
        dataset="matpes_test",
        observables=("energy", "forces"),
        member_ids=("member_01", "member_02"),
        shard_index=0,
        structure_start=0,
        structure_stop=2,
        atom_start=0,
        atom_stop=3,
        batch_size=32,
    )


def test_derived_layout_is_experiment_and_dataset_scoped(tmp_path: Path) -> None:
    layout = DerivedLayout(tmp_path, "mace_fge", "matpes_train")

    assert layout.root == tmp_path / "inference" / "mace_fge" / "matpes_train"
    assert layout.prediction_manifest == layout.root / "prediction" / "manifest.json"
    assert layout.prediction_shard(3).name == "shard_000003.pt"
    assert layout.prediction_shard_manifest(3).name == "shard_000003.json"
    assert layout.evaluation_dir == layout.root / "evaluation"


def test_missing_prediction_shard_is_not_reusable(tmp_path: Path) -> None:
    layout = DerivedLayout(tmp_path, "experiment", "dataset")

    assert verify_prediction_shard_if_present(layout, 0, _signature()) is False


def test_prediction_shard_round_trip_is_source_neutral(tmp_path: Path) -> None:
    layout = DerivedLayout(tmp_path, "experiment", "dataset")
    payload = canonical_prediction_fixture()

    tensor_path = write_prediction_shard(layout, 0, payload, _signature())

    assert tensor_path == layout.prediction_shard(0)
    assert verify_prediction_shard_if_present(layout, 0, _signature()) is True
    manifest = json.loads(layout.prediction_shard_manifest(0).read_text(encoding="utf-8"))
    serialized = json.dumps(manifest, sort_keys=True)
    assert manifest["schema_version"] == "fge.derived-prediction-shard.v1"
    assert manifest["artifact"]["path"] == "shard_000000.pt"
    assert len(manifest["artifact"]["sha256"]) == 64
    assert "/old/" not in serialized
    assert "input_path" not in serialized
    assert "legacy" not in serialized


def test_prediction_shard_rejects_partial_artifact(tmp_path: Path) -> None:
    layout = DerivedLayout(tmp_path, "experiment", "dataset")
    layout.prediction_shard(0).parent.mkdir(parents=True)
    torch.save(canonical_prediction_fixture(), layout.prediction_shard(0))

    with pytest.raises(HardFailure, match="partially present"):
        verify_prediction_shard_if_present(layout, 0, _signature())


def test_prediction_shard_rejects_signature_mismatch(tmp_path: Path) -> None:
    layout = DerivedLayout(tmp_path, "experiment", "dataset")
    write_prediction_shard(layout, 0, canonical_prediction_fixture(), _signature())
    changed = {**_signature(), "batch_size": 64}

    with pytest.raises(HardFailure, match="signature"):
        verify_prediction_shard_if_present(layout, 0, changed)


def test_prediction_shard_rejects_hash_mismatch(tmp_path: Path) -> None:
    layout = DerivedLayout(tmp_path, "experiment", "dataset")
    write_prediction_shard(layout, 0, canonical_prediction_fixture(), _signature())
    layout.prediction_shard(0).write_bytes(b"corrupt")

    with pytest.raises(HardFailure, match="SHA-256"):
        verify_prediction_shard_if_present(layout, 0, _signature())


def test_prediction_shard_rejects_corrupt_manifest(tmp_path: Path) -> None:
    layout = DerivedLayout(tmp_path, "experiment", "dataset")
    write_prediction_shard(layout, 0, canonical_prediction_fixture(), _signature())
    layout.prediction_shard_manifest(0).write_text("{", encoding="utf-8")

    with pytest.raises(HardFailure, match="manifest"):
        verify_prediction_shard_if_present(layout, 0, _signature())


@pytest.mark.parametrize("mutation", ["nan", "float32", "mapping"])
def test_prediction_shard_rejects_invalid_payload(
    tmp_path: Path, mutation: str
) -> None:
    layout = DerivedLayout(tmp_path, "experiment", "dataset")
    payload = canonical_prediction_fixture()
    if mutation == "nan":
        payload["energy_members"][0, 0] = float("nan")
    elif mutation == "float32":
        payload["energy_members"] = payload["energy_members"].float()
    else:
        payload["atom_to_structure"] = torch.tensor([0, 1, 0])

    with pytest.raises(HardFailure):
        write_prediction_shard(layout, 0, payload, _signature())
