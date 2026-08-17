from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Uncertainty_Quantification.FGE.fge.artifacts import sha256_file
from Uncertainty_Quantification.FGE.fge.data import ConfigurationShard
from Uncertainty_Quantification.FGE.fge.dataset_spec import DatasetSpec
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge import dataset_prediction
from Uncertainty_Quantification.FGE.tests.test_prediction import (
    canonical_prediction_fixture,
)


class FakeConfig:
    def __init__(self, formal_root: Path) -> None:
        self.output_dir = formal_root
        self.project_name = "mace_fge"
        self._sections = {
            "data": {
                "energy_key": "energy",
                "forces_key": "forces",
                "stress_key": "stress",
                "head_name": "Default",
            },
            "prediction": {"batch_size": 64, "compute_stress": False},
            "training": {"device": "cpu"},
        }

    def section(self, name: str):
        return self._sections[name]


def _spec(tmp_path: Path, *, compute_stress: bool = False) -> DatasetSpec:
    path = tmp_path / "input.extxyz"
    path.write_text("", encoding="utf-8")
    return DatasetSpec(
        label="matpes_test",
        path=path,
        compute_stress=compute_stress,
        batch_size=7,
        shard_size=2,
    )


def _training_manifest() -> dict[str, object]:
    return {
        "members": [
            {"member_id": "member_01", "raw": {"path": "models/one.model"}},
            {"member_id": "member_02", "raw": {"path": "models/two.model"}},
        ]
    }


def _configuration_shard() -> ConfigurationShard:
    return ConfigurationShard(
        index=0,
        structure_start=0,
        structure_stop=2,
        configurations=(
            SimpleNamespace(atomic_numbers=[1, 1]),
            SimpleNamespace(atomic_numbers=[1]),
        ),
    )


def test_load_training_manifest_verifies_formal_markers_and_raw_hashes(
    tmp_path: Path,
) -> None:
    formal = tmp_path / "formal"
    (formal / "training" / "models").mkdir(parents=True)
    members = []
    for index, name in enumerate(("one.model", "two.model"), start=1):
        path = formal / "training" / "models" / name
        path.write_bytes(f"model-{index}".encode())
        members.append(
            {
                "member_id": f"member_{index:02d}",
                "raw": {
                    "path": f"training/models/{name}",
                    "sha256": sha256_file(path),
                },
            }
        )
    (formal / "training" / "manifest.json").write_text(
        json.dumps({"members": members}), encoding="utf-8"
    )
    (formal / "validation.json").write_text('{"status":"PASS"}', encoding="utf-8")
    (formal / "result_manifest.json").write_text(
        '{"status":"PASS"}', encoding="utf-8"
    )

    loaded = dataset_prediction.load_training_manifest(FakeConfig(formal))

    assert [member["member_id"] for member in loaded["members"]] == [
        "member_01",
        "member_02",
    ]


def test_load_training_manifest_rejects_raw_hash_mismatch(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    (formal / "training").mkdir(parents=True)
    model = formal / "member.model"
    model.write_bytes(b"model")
    (formal / "training" / "manifest.json").write_text(
        json.dumps(
            {
                "members": [
                    {
                        "member_id": "member_01",
                        "raw": {"path": "member.model", "sha256": "0" * 64},
                    },
                    {
                        "member_id": "member_02",
                        "raw": {"path": "member.model", "sha256": "0" * 64},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (formal / "validation.json").write_text('{"status":"PASS"}', encoding="utf-8")
    (formal / "result_manifest.json").write_text(
        '{"status":"PASS"}', encoding="utf-8"
    )

    with pytest.raises(HardFailure, match="hash mismatched"):
        dataset_prediction.load_training_manifest(FakeConfig(formal))


def test_predict_dataset_resumes_only_verified_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = FakeConfig(tmp_path / "formal")
    spec = _spec(tmp_path)
    outputs = tmp_path / "outputs"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        dataset_prediction, "load_training_manifest", lambda _config: _training_manifest()
    )
    monkeypatch.setattr(
        dataset_prediction,
        "iter_extxyz_shards",
        lambda *_args, **_kwargs: iter((_configuration_shard(),)),
    )

    def generate(*_args, **kwargs):
        calls.append(kwargs)
        return canonical_prediction_fixture()

    monkeypatch.setattr(dataset_prediction, "generate_prediction_payload", generate)

    manifest_path = dataset_prediction.predict_dataset(config, spec, outputs)

    assert manifest_path == outputs / "inference" / "mace_fge" / "matpes_test" / "prediction" / "manifest.json"
    assert calls == [{"compute_stress": False, "batch_size": 7}]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["dataset"] == spec.neutral_manifest()
    assert manifest["shape_symbols"] == {"K": 2, "S": 2, "A": 3}
    assert manifest["shard_count"] == 1
    assert "input_path" not in json.dumps(manifest, sort_keys=True)

    monkeypatch.setattr(
        dataset_prediction,
        "generate_prediction_payload",
        lambda *_args, **_kwargs: pytest.fail("verified shard was recomputed"),
    )
    assert dataset_prediction.predict_dataset(config, spec, outputs) == manifest_path


def test_predict_dataset_propagates_missing_stress_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = FakeConfig(tmp_path / "formal")
    spec = _spec(tmp_path, compute_stress=True)
    monkeypatch.setattr(
        dataset_prediction, "load_training_manifest", lambda _config: _training_manifest()
    )

    def fail(*_args, **_kwargs):
        raise HardFailure("structure 0 has invalid stress")

    monkeypatch.setattr(dataset_prediction, "iter_extxyz_shards", fail)

    with pytest.raises(HardFailure, match="stress"):
        dataset_prediction.predict_dataset(config, spec, tmp_path / "outputs")


def test_predict_dataset_does_not_write_formal_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal = tmp_path / "formal"
    formal.mkdir()
    sentinel = formal / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    config = FakeConfig(formal)
    spec = _spec(tmp_path)
    monkeypatch.setattr(
        dataset_prediction, "load_training_manifest", lambda _config: _training_manifest()
    )
    monkeypatch.setattr(
        dataset_prediction,
        "iter_extxyz_shards",
        lambda *_args, **_kwargs: iter((_configuration_shard(),)),
    )
    monkeypatch.setattr(
        dataset_prediction,
        "generate_prediction_payload",
        lambda *_args, **_kwargs: canonical_prediction_fixture(),
    )

    dataset_prediction.predict_dataset(config, spec, tmp_path / "outputs")

    assert tuple(formal.iterdir()) == (sentinel,)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
