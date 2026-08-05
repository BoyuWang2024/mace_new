from __future__ import annotations

import json
from pathlib import Path

import torch

from Uncertainty_Quantification.FGE.fge.evaluation import evaluate_prediction


def _canonical_prediction() -> dict[str, object]:
    return {
        "schema_version": "fge.prediction.v1",
        "split": "test",
        "member_source": "raw",
        "member_ids": ["member_01", "member_02"],
        "observables": ["energy", "forces"],
        "energy_members": torch.tensor(
            [[2.0, 8.0], [4.0, 12.0]], dtype=torch.float64
        ),
        "forces_members": torch.tensor(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ],
            dtype=torch.float64,
        ),
        "energy_reference": torch.tensor([3.0, 9.0], dtype=torch.float64),
        "forces_reference": torch.tensor(
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
        ),
        "n_atoms": torch.tensor([1, 1], dtype=torch.int64),
        "atom_to_structure": torch.tensor([0, 1], dtype=torch.int64),
        "structure_ptr": torch.tensor([0, 1, 2], dtype=torch.int64),
    }


def test_evaluate_prediction_writes_both_branches_without_models_or_data(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "prediction").mkdir(parents=True)
    (run_dir / "training").mkdir()
    torch.save(_canonical_prediction(), run_dir / "prediction" / "test_raw.pt")
    training = {
        "base_model_metrics": {"energy_rmse": 2.0, "forces_rmse": 3.0},
        "members": [
            {"member_id": "member_01", "raw_metrics": {"energy_rmse": 1.0, "forces_rmse": 2.0}},
            {"member_id": "member_02", "raw_metrics": {"energy_rmse": 3.0, "forces_rmse": 4.0}},
        ]
    }
    (run_dir / "training" / "manifest.json").write_text(
        json.dumps(training), encoding="utf-8"
    )
    config = {
        "ensemble": {"eps_energy_ratio": 0.1, "eps_force_ratio": 0.1},
        "evaluation": {"risk_coverages": (1.0, 0.5), "force_structure_quantile": 0.95},
    }

    summary = evaluate_prediction(config, run_dir)

    assert set(summary) == {"equal_weight", "validation_weighted", "warnings"}
    for branch in ("equal_weight", "validation_weighted"):
        branch_dir = run_dir / "evaluation" / branch
        assert {
            "ensemble.pt",
            "uncertainty.pt",
            "metrics.json",
            "correlations.csv",
            "risk_coverage.csv",
        } == {path.name for path in branch_dir.iterdir()}
        ensemble = torch.load(branch_dir / "ensemble.pt", weights_only=True)
        assert ensemble["energy"].dtype == torch.float64
        assert ensemble["forces"].device.type == "cpu"
        metrics = json.loads((branch_dir / "metrics.json").read_text("utf-8"))
        assert metrics["energy_total"]["rmse"] >= 0.0
    assert (run_dir / "evaluation" / "report.md").read_text("utf-8").startswith(
        "# FGE 评估报告"
    )
    assert not (run_dir / "evaluation" / "model.pt").exists()
