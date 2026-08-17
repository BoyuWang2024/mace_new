from __future__ import annotations

from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.preflight import _load_model


def test_load_model_remaps_embedded_jit_modules_to_requested_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "member.model"
    path.write_bytes(b"placeholder")
    model = torch.nn.Linear(1, 1)
    jit_locations: list[object] = []

    def fake_jit_load(source: object, map_location: object = None) -> object:
        del source
        jit_locations.append(map_location)
        return object()

    def fake_load(
        source: Path, *, map_location: object, weights_only: object
    ) -> torch.nn.Module:
        assert source == path
        assert map_location == "cpu"
        assert weights_only is False
        torch.jit.load(object())
        return model

    monkeypatch.setattr(torch.jit, "load", fake_jit_load)
    monkeypatch.setattr(torch, "load", fake_load)

    loaded = _load_model(path, "cpu")

    assert loaded is model
    assert jit_locations == [torch.device("cpu")]
    assert torch.jit.load is fake_jit_load
