from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Uncertainty_Quantification.FGE.fge import prediction
from Uncertainty_Quantification.FGE.fge.data import iter_extxyz_shards
from Uncertainty_Quantification.FGE.fge.errors import HardFailure


KEYS = {
    "energy": "REF_energy",
    "forces": "REF_forces",
    "stress": "REF_stress",
    "head": "head",
}


def _write_three_structures(path: Path) -> Path:
    path.write_text(
        "2\n"
        'Properties=species:S:1:pos:R:3:REF_forces:R:3 REF_energy=1.0 pbc="F F F"\n'
        "H 0 0 0 0.1 0 0\nH 0 0 1 -0.1 0 0\n"
        "1\n"
        'Properties=species:S:1:pos:R:3:REF_forces:R:3 REF_energy=2.0 pbc="F F F"\n'
        "H 0 0 0 0 0 0\n"
        "3\n"
        'Properties=species:S:1:pos:R:3:REF_forces:R:3 REF_energy=3.0 pbc="F F F"\n'
        "H 0 0 0 0 0 0\nH 0 0 1 0 0 0\nH 0 0 2 0 0 0\n",
        encoding="utf-8",
    )
    return path


def test_iter_extxyz_shards_preserves_structure_order(tmp_path: Path) -> None:
    shards = tuple(
        iter_extxyz_shards(
            _write_three_structures(tmp_path / "data.extxyz"),
            keys=KEYS,
            required={"energy", "forces"},
            head_name="Default",
            shard_size=2,
        )
    )

    assert [shard.index for shard in shards] == [0, 1]
    assert [(shard.structure_start, shard.structure_stop) for shard in shards] == [
        (0, 2),
        (2, 3),
    ]
    assert [[config.properties["energy"] for config in shard.configurations] for shard in shards] == [
        [1.0, 2.0],
        [3.0],
    ]
    assert [[len(config.atomic_numbers) for config in shard.configurations] for shard in shards] == [
        [2, 1],
        [3],
    ]


def test_iter_extxyz_shards_rejects_missing_requested_stress(tmp_path: Path) -> None:
    with pytest.raises(HardFailure, match="stress"):
        tuple(
            iter_extxyz_shards(
                _write_three_structures(tmp_path / "data.xyz"),
                keys=KEYS,
                required={"energy", "forces", "stress"},
                shard_size=2,
            )
        )


class FakeConfig:
    def __init__(self, root: Path, *, compute_stress: bool = True) -> None:
        self.output_dir = root
        self._sections = {
            "data": {
                "energy_key": "REF_energy",
                "forces_key": "REF_forces",
                "stress_key": "REF_stress",
                "head_name": "Default",
            },
            "prediction": {"batch_size": 2, "compute_stress": compute_stress},
            "training": {"device": "cpu"},
            "paths": {"test_data": str(root / "formal-test.extxyz")},
        }

    def section(self, name: str):
        return self._sections[name]


def _member_result(offset: float) -> dict[str, torch.Tensor]:
    return {
        "energy": torch.tensor([1.0 + offset, 2.0 + offset], dtype=torch.float64),
        "forces": torch.full((3, 3), offset, dtype=torch.float64),
        "stress": torch.full((2, 3, 3), offset, dtype=torch.float64),
        "energy_reference": torch.tensor([1.5, 2.5], dtype=torch.float64),
        "forces_reference": torch.zeros((3, 3), dtype=torch.float64),
        "stress_reference": torch.zeros((2, 3, 3), dtype=torch.float64),
        "n_atoms": torch.tensor([2, 1], dtype=torch.int64),
    }


def test_generate_prediction_payload_accepts_explicit_configurations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = FakeConfig(tmp_path)
    manifest = {
        "members": [
            {"member_id": "member_01", "raw": {"path": "models/member_01.model"}},
            {"member_id": "member_02", "raw": {"path": "models/member_02.model"}},
        ]
    }
    models = [
        SimpleNamespace(
            atomic_numbers=torch.tensor([1]),
            r_max=5.0,
            heads=["Default"],
            parameters=lambda: iter(
                (torch.nn.Parameter(torch.ones(1, dtype=torch.float64)),)
            ),
        ),
        SimpleNamespace(
            atomic_numbers=torch.tensor([1]),
            r_max=5.0,
            heads=["Default"],
            parameters=lambda: iter(
                (torch.nn.Parameter(torch.ones(1, dtype=torch.float64)),)
            ),
        ),
    ]
    results = [_member_result(0.0), _member_result(1.0)]
    monkeypatch.setattr(prediction, "_load_model", lambda *_args: models.pop(0))
    monkeypatch.setattr(prediction, "build_mace_loaders", lambda configurations, **_kwargs: configurations)
    monkeypatch.setattr(prediction, "_infer_one", lambda *_args: results.pop(0))

    payload = prediction.generate_prediction_payload(
        config,
        manifest,
        configurations=(object(), object()),
        compute_stress=True,
    )

    shape = prediction.validate_prediction_payload(payload)
    assert (shape.members, shape.structures, shape.atoms, shape.has_stress) == (2, 2, 3, True)
    assert payload["member_ids"] == ["member_01", "member_02"]
    assert payload["observables"] == ["energy", "forces", "stress"]
    assert payload["energy_members"].tolist() == [[1.0, 2.0], [2.0, 3.0]]


def test_generate_prediction_payload_checks_stress_reference_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = FakeConfig(tmp_path)
    manifest = {
        "members": [
            {"member_id": "member_01", "raw": {"path": "one.model"}},
            {"member_id": "member_02", "raw": {"path": "two.model"}},
        ]
    }
    results = [_member_result(0.0), _member_result(1.0)]
    results[1]["stress_reference"] = torch.ones((2, 3, 3), dtype=torch.float64)
    model = SimpleNamespace(
            atomic_numbers=torch.tensor([1]),
            r_max=5.0,
            heads=["Default"],
            parameters=lambda: iter(
                (torch.nn.Parameter(torch.ones(1, dtype=torch.float64)),)
            ),
        )
    monkeypatch.setattr(prediction, "_load_model", lambda *_args: model)
    monkeypatch.setattr(prediction, "build_mace_loaders", lambda configurations, **_kwargs: configurations)
    monkeypatch.setattr(prediction, "_infer_one", lambda *_args: results.pop(0))

    with pytest.raises(HardFailure, match="references are not aligned"):
        prediction.generate_prediction_payload(
            config,
            manifest,
            configurations=(object(),),
            compute_stress=True,
        )


def test_formal_prediction_wrapper_keeps_configured_test_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = FakeConfig(tmp_path, compute_stress=False)
    sentinel = [object()]
    calls: dict[str, object] = {}
    monkeypatch.setattr(prediction, "load_extxyz", lambda path, **kwargs: calls.update(path=path, kwargs=kwargs) or sentinel)
    monkeypatch.setattr(
        prediction,
        "generate_prediction_payload",
        lambda actual_config, manifest, configurations, *, compute_stress: calls.update(
            config=actual_config,
            manifest=manifest,
            configurations=configurations,
            compute_stress=compute_stress,
        )
        or {"payload": True},
    )

    result = prediction._generate_prediction_payload(config, {"members": []})

    assert result == {"payload": True}
    assert calls["path"] == tmp_path / "formal-test.extxyz"
    assert calls["configurations"] is sentinel
    assert calls["compute_stress"] is False
