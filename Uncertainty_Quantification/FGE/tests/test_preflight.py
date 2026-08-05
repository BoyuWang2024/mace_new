from __future__ import annotations

import json
from pathlib import Path

import torch

from Uncertainty_Quantification.FGE.fge.preflight import run_preflight


class FakeConfig:
    def __init__(self, root: Path) -> None:
        self.output_dir = root
        self.project_name = "case"
        self._sections = {
            "paths": {
                "base_checkpoint": str(root / "base.model"),
                "train_data": str(root / "train.xyz"),
                "val_data": str(root / "val.xyz"),
                "test_data": str(root / "test.xyz"),
                "output_root": str(root),
            },
            "data": {"energy_key": "E", "forces_key": "F", "stress_key": "S", "head_name": "Default"},
            "training": {"expected_readout_parameter_count": 4},
        }

    def section(self, name: str):
        return self._sections[name]


def test_evaluate_preflight_never_calls_model_forward(tmp_path: Path, monkeypatch) -> None:
    config = FakeConfig(tmp_path)
    (tmp_path / "training").mkdir()
    (tmp_path / "prediction").mkdir()
    (tmp_path / "training" / "manifest.json").write_text(
        json.dumps({"members": [], "base_model_metrics": {}}), encoding="utf-8"
    )
    torch.save({"schema_version": "fge.prediction.v1"}, tmp_path / "prediction" / "test_raw.pt")
    monkeypatch.setattr(
        torch.nn.Module,
        "__call__",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("forward")),
    )
    report = run_preflight(config, "evaluate")
    assert report["stage"] == "evaluate"
    assert "data_sha256" not in json.dumps(report)
    assert (tmp_path / "preflight" / "evaluate.json").is_file()
