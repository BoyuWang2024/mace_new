from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from Uncertainty_Quantification.FGE.fge import prediction


class FakeConfig:
    output_dir = "."

    def section(self, name: str):
        return {
            "data": {"head_name": "default"},
            "prediction": {"batch_size": 2},
            "training": {"device": "cpu"},
        }[name]


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.atomic_numbers = torch.tensor([1], dtype=torch.int64)
        self.r_max = torch.tensor(5.0, dtype=torch.float64)
        self.heads = ["default"]


def _result() -> dict[str, torch.Tensor]:
    return {
        "energy": torch.zeros(1, dtype=torch.float64),
        "forces": torch.zeros((1, 3), dtype=torch.float64),
        "energy_reference": torch.zeros(1, dtype=torch.float64),
        "forces_reference": torch.zeros((1, 3), dtype=torch.float64),
        "n_atoms": torch.ones(1, dtype=torch.int64),
    }


def test_prediction_builds_graphs_in_each_model_dtype_and_restores_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = [FakeModel(), FakeModel()]
    observed: list[torch.dtype] = []
    initial = torch.get_default_dtype()
    monkeypatch.setattr(prediction, "_load_model", lambda *_args: models.pop(0))
    monkeypatch.setattr(
        prediction,
        "build_mace_loaders",
        lambda *_args, **_kwargs: observed.append(torch.get_default_dtype()) or (),
    )
    monkeypatch.setattr(prediction, "_infer_one", lambda *_args: _result())
    manifest = {
        "members": [
            {"member_id": "member_01", "raw": {"path": "one.model"}},
            {"member_id": "member_02", "raw": {"path": "two.model"}},
        ]
    }

    prediction.generate_prediction_payload(
        FakeConfig(),
        manifest,
        [SimpleNamespace()],
        compute_stress=False,
    )

    assert observed == [torch.float64, torch.float64]
    assert torch.get_default_dtype() == initial
