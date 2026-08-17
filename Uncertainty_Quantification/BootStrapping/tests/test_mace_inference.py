from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import sha256_file
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.mace_inference import (
    load_member_model,
    predict_batch,
)


class FakeBatch:
    def __init__(self, ptr: list[int], *, num_graphs: int | None = None, num_nodes: int | None = None):
        self.ptr = torch.tensor(ptr, dtype=torch.int64)
        self.num_graphs = len(ptr) - 1 if num_graphs is None else num_graphs
        self.num_nodes = ptr[-1] if num_nodes is None else num_nodes
        self.positions = torch.zeros((ptr[-1], 3), dtype=torch.float32)

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {"ptr": self.ptr, "positions": self.positions}


class FakeModel(torch.nn.Module):
    def __init__(self, output: dict[str, Any]):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.output = output
        self.calls: list[dict[str, object]] = []

    def forward(self, data: dict[str, torch.Tensor], **kwargs: object) -> dict[str, Any]:
        self.calls.append({"data": data, **kwargs})
        return self.output


def _output(*, stress: object | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "energy": torch.tensor([-1.0, -2.0]),
        "forces": torch.arange(15, dtype=torch.float32).reshape(5, 3),
    }
    if stress is not None:
        result["stress"] = stress
    return result


def test_energy_force_prediction_disables_stress_and_returns_numpy() -> None:
    model = FakeModel(_output())

    result = predict_batch(model, FakeBatch([0, 2, 5]), domains=("energy", "forces"))

    assert set(result) == {"energy", "forces"}
    assert result["energy"].shape == (2,)
    assert result["forces"].shape == (5, 3)
    assert all(isinstance(value, np.ndarray) for value in result.values())
    assert len(model.calls) == 1
    call = model.calls[0]
    data = call["data"]
    assert isinstance(data, dict)
    assert torch.equal(data["ptr"], torch.tensor([0, 2, 5]))
    assert call["training"] is False
    assert call["compute_force"] is True
    assert call["compute_virials"] is False
    assert call["compute_stress"] is False


def test_stress_prediction_requests_stress_and_normalizes_singleton_axis() -> None:
    model = FakeModel(_output(stress=torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3)))

    result = predict_batch(
        model,
        FakeBatch([0, 2, 5]),
        domains=("energy", "forces", "stress"),
    )

    assert result["stress"].shape == (2, 3, 3)
    assert model.calls[0]["compute_stress"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("energy", torch.tensor([-1.0, float("nan")])),
        ("forces", torch.full((5, 3), float("inf"))),
        ("stress", torch.full((2, 3, 3), float("nan"))),
    ],
)
def test_prediction_rejects_non_finite_outputs(field: str, value: torch.Tensor) -> None:
    output = _output(stress=torch.zeros((2, 3, 3)))
    output[field] = value

    with pytest.raises(HardFailure, match=field):
        predict_batch(
            FakeModel(output),
            FakeBatch([0, 2, 5]),
            domains=("energy", "forces", "stress"),
        )


@pytest.mark.parametrize(
    ("batch", "output", "message"),
    [
        (FakeBatch([1, 3, 6]), _output(), "ptr"),
        (FakeBatch([0, 2, 5], num_graphs=3), _output(), "structure"),
        (FakeBatch([0, 2, 5], num_nodes=4), _output(), "atom"),
        (FakeBatch([0, 2, 5]), {**_output(), "energy": torch.zeros(3)}, "energy"),
        (FakeBatch([0, 2, 5]), {**_output(), "forces": torch.zeros(4, 3)}, "forces"),
    ],
)
def test_prediction_rejects_batch_or_output_shape_mismatch(
    batch: FakeBatch, output: dict[str, object], message: str
) -> None:
    with pytest.raises(HardFailure, match=message):
        predict_batch(FakeModel(output), batch, domains=("energy", "forces"))


def test_load_member_model_verifies_hash_and_configures_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "raw_best.model"
    path.write_bytes(b"trusted model placeholder")
    expected_sha256 = sha256_file(path)
    model = FakeModel(_output())
    calls: list[tuple[Path, object, object]] = []

    def fake_load(source: Path, *, map_location: object, weights_only: object) -> torch.nn.Module:
        calls.append((source, map_location, weights_only))
        return model

    monkeypatch.setattr(torch, "load", fake_load)

    loaded = load_member_model(
        path,
        expected_sha256=expected_sha256,
        device="cpu",
        dtype=torch.float64,
    )

    assert loaded is model
    assert calls == [(path.resolve(), "cpu", False)]
    assert next(model.parameters()).device.type == "cpu"
    assert next(model.parameters()).dtype == torch.float64
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_load_member_model_remaps_embedded_jit_modules_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "raw_best.model"
    path.write_bytes(b"trusted model placeholder")
    model = FakeModel(_output())
    jit_map_locations: list[object] = []

    def fake_jit_load(source: object, map_location: object = None) -> object:
        del source
        jit_map_locations.append(map_location)
        return object()

    def fake_load(source: Path, *, map_location: object, weights_only: object) -> torch.nn.Module:
        del source, map_location, weights_only
        torch.jit.load(object())
        return model

    monkeypatch.setattr(torch.jit, "load", fake_jit_load)
    monkeypatch.setattr(torch, "load", fake_load)

    loaded = load_member_model(
        path,
        expected_sha256=sha256_file(path),
        device="cpu",
        dtype=torch.float64,
    )

    assert loaded is model
    assert jit_map_locations == [torch.device("cpu")]
    assert torch.jit.load is fake_jit_load


def test_load_member_model_rejects_hash_mismatch_and_non_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "raw_best.model"
    path.write_bytes(b"trusted model placeholder")
    with pytest.raises(HardFailure, match="SHA-256"):
        load_member_model(
            path,
            expected_sha256="0" * 64,
            device="cpu",
            dtype=torch.float32,
        )

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model": "not a module"})
    with pytest.raises(HardFailure, match="torch.nn.Module"):
        load_member_model(
            path,
            expected_sha256=sha256_file(path),
            device="cpu",
            dtype=torch.float32,
        )
