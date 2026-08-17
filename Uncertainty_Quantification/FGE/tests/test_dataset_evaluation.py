from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from Uncertainty_Quantification.FGE.fge.dataset_evaluation import (
    _global_risk_rows,
    evaluate_dataset,
)
from Uncertainty_Quantification.FGE.fge.derived_artifacts import (
    DerivedLayout,
    build_prediction_shard_signature,
    write_prediction_shard,
)
from Uncertainty_Quantification.FGE.fge.metrics import compute_risk_coverage


class FakeConfig:
    def __init__(self, formal_root: Path) -> None:
        self.output_dir = formal_root
        self.project_name = "mace_fge"
        self._sections = {
            "ensemble": {"eps_energy_ratio": 0.1, "eps_force_ratio": 0.1},
            "evaluation": {
                "risk_coverages": (1.0, 0.5),
                "force_structure_quantile": 0.95,
            },
            "quality": {
                "weak_rmse_multiplier": 2.0,
                "collapsed_rmse_multiplier": 10.0,
            },
        }

    def section(self, name: str):
        return self._sections[name]


def _prediction(offset: float) -> dict[str, object]:
    return {
        "schema_version": "fge.prediction.v1",
        "split": "test",
        "member_source": "raw",
        "member_ids": ["member_01", "member_02"],
        "observables": ["energy", "forces", "stress"],
        "energy_members": torch.tensor(
            [[1.0 + offset], [3.0 + offset]], dtype=torch.float64
        ),
        "forces_members": torch.tensor(
            [
                [[1.0 + offset, 0.0, 0.0]],
                [[3.0 + offset, 0.0, 0.0]],
            ],
            dtype=torch.float64,
        ),
        "stress_members": torch.tensor(
            [
                [[[1.0 + offset, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                [[[3.0 + offset, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            ],
            dtype=torch.float64,
        ),
        "energy_reference": torch.tensor([offset], dtype=torch.float64),
        "forces_reference": torch.zeros((1, 3), dtype=torch.float64),
        "stress_reference": torch.zeros((1, 3, 3), dtype=torch.float64),
        "n_atoms": torch.tensor([1], dtype=torch.int64),
        "atom_to_structure": torch.tensor([0], dtype=torch.int64),
        "structure_ptr": torch.tensor([0, 1], dtype=torch.int64),
    }


def _prepare(tmp_path: Path) -> tuple[FakeConfig, DerivedLayout]:
    formal = tmp_path / "formal"
    (formal / "training").mkdir(parents=True)
    training = {
        "base_model_metrics": {"energy_rmse": 2.0, "forces_rmse": 2.0},
        "members": [
            {
                "member_id": "member_01",
                "raw_metrics": {"energy_rmse": 1.0, "forces_rmse": 1.0},
            },
            {
                "member_id": "member_02",
                "raw_metrics": {"energy_rmse": 3.0, "forces_rmse": 3.0},
            },
        ],
    }
    (formal / "training" / "manifest.json").write_text(
        json.dumps(training), encoding="utf-8"
    )
    layout = DerivedLayout(tmp_path / "outputs", "mace_fge", "matpes_test")
    shards = []
    for index, offset in enumerate((0.0, 10.0)):
        signature = build_prediction_shard_signature(
            dataset="matpes_test",
            observables=("energy", "forces", "stress"),
            member_ids=("member_01", "member_02"),
            shard_index=index,
            structure_start=index,
            structure_stop=index + 1,
            atom_start=index,
            atom_stop=index + 1,
            batch_size=4,
        )
        write_prediction_shard(layout, index, _prediction(offset), signature)
        shards.append(
            {
                "index": index,
                "manifest": {
                    "path": f"shards/shard_{index:06d}.json",
                },
            }
        )
    layout.prediction_manifest.write_text(
        json.dumps(
            {
                "schema_version": "fge.derived-prediction.v1",
                "status": "PASS",
                "experiment": "mace_fge",
                "dataset": {
                    "schema_version": "fge.dataset.v1",
                    "dataset": "matpes_test",
                    "observables": ["energy", "forces", "stress"],
                    "keys": {
                        "energy": "energy",
                        "forces": "forces",
                        "stress": "stress",
                        "head": "default",
                    },
                    "batch_size": 4,
                    "shard_size": 1,
                },
                "member_source": "raw",
                "member_ids": ["member_01", "member_02"],
                "observables": ["energy", "forces", "stress"],
                "shape_symbols": {"K": 2, "S": 2, "A": 2},
                "shard_count": 2,
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )
    return FakeConfig(formal), layout


def test_global_risk_rows_sort_all_shards_together() -> None:
    u0 = torch.tensor([0.1, 0.9], dtype=torch.float64)
    e0 = torch.tensor([1.0, 9.0], dtype=torch.float64)
    u1 = torch.tensor([0.2, 0.8], dtype=torch.float64)
    e1 = torch.tensor([2.0, 8.0], dtype=torch.float64)
    coverages = (1.0, 0.5)

    expected = compute_risk_coverage(
        torch.cat((u0, u1)), torch.cat((e0, e1)), coverages
    )

    assert _global_risk_rows(((u0, e0), (u1, e1)), coverages) == expected


def test_evaluate_dataset_writes_sharded_branches_and_global_metrics(
    tmp_path: Path,
) -> None:
    config, layout = _prepare(tmp_path)

    audit_path = evaluate_dataset(config, layout)

    assert audit_path == layout.evaluation_dir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["observables"] == ["energy", "forces", "stress"]
    for branch in ("equal_weight", "validation_weighted"):
        branch_root = layout.evaluation_dir / branch
        assert len(tuple((branch_root / "shards").glob("*.pt"))) == 2
        ensemble = torch.load(branch_root / "ensemble.pt", weights_only=True)
        assert "energy" not in ensemble
        assert ensemble["shape_symbols"] == {"S": 2, "A": 2}
        metrics = json.loads((branch_root / "metrics.json").read_text("utf-8"))
        assert "stress_component" in metrics
        with (branch_root / "risk_coverage.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert any(row["metric"] == "stress_component_std" for row in rows)
    assert (layout.evaluation_dir / "report.md").is_file()
