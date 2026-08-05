from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.artifacts import (
    ExperimentLayout,
    StagingExperiment,
    atomic_torch_save,
    atomic_write_json,
    normalize_artifact_path,
    sha256_file,
)
from Uncertainty_Quantification.FGE.fge.errors import HardFailure


def test_atomic_json_keeps_existing_target_when_replace_fails(
    tmp_path: Path, monkeypatch
):
    target = tmp_path / "value.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(*_args):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(target, {"new": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_json_rejects_nan(tmp_path: Path):
    with pytest.raises(HardFailure, match="non-finite JSON"):
        atomic_write_json(tmp_path / "bad.json", {"value": float("nan")})


def test_atomic_torch_save_round_trips_and_hashes(tmp_path: Path):
    target = tmp_path / "tensor.pt"
    payload = {"value": torch.tensor([1.0, 2.0], dtype=torch.float64)}

    atomic_torch_save(target, payload)

    loaded = torch.load(target, map_location="cpu", weights_only=True)
    assert torch.equal(loaded["value"], payload["value"])
    assert len(sha256_file(target)) == 64


def test_normalize_artifact_path_rejects_escape(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()

    with pytest.raises(HardFailure, match="relative path"):
        normalize_artifact_path(root, tmp_path / "escape.pt")


def test_experiment_layout_uses_fixed_directories(tmp_path: Path):
    layout = ExperimentLayout(tmp_path / "run")

    assert layout.training_manifest == tmp_path / "run/training/manifest.json"
    assert layout.prediction_tensor == tmp_path / "run/prediction/test_raw.pt"
    assert layout.result_manifest == tmp_path / "run/result_manifest.json"


def test_staging_publish_is_atomic_and_refuses_completed_target(tmp_path: Path):
    final = tmp_path / "case"
    with StagingExperiment(final) as transaction:
        atomic_write_json(transaction.layout.root / "validation.json", {"status": "PASS"})
        published = transaction.publish()

    assert published == final
    assert json.loads((final / "validation.json").read_text()) == {"status": "PASS"}

    atomic_write_json(final / "result_manifest.json", {"status": "PASS"})
    with pytest.raises(HardFailure, match="completed result"):
        StagingExperiment(final)
