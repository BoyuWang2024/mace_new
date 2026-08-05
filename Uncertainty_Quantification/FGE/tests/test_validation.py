from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.artifacts import atomic_torch_save, atomic_write_json
from Uncertainty_Quantification.FGE.fge.evaluation import evaluate_prediction
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.manifests import build_prediction_manifest
from Uncertainty_Quantification.FGE.fge.validation import schema_signature, validate_result


class FakeConfig:
    project_name = "case"

    def __init__(self, root: Path) -> None:
        self.output_dir = root
        self._sections = {
            "ensemble": {"eps_energy_ratio": 0.1, "eps_force_ratio": 0.1},
            "evaluation": {"risk_coverages": (1.0, 0.5), "force_structure_quantile": 0.95},
        }

    def section(self, name: str):
        return self._sections[name]


def _make_result(root: Path, *, warning: bool = False) -> FakeConfig:
    (root / "training" / "members" / "raw").mkdir(parents=True)
    (root / "training" / "members" / "ema").mkdir(parents=True)
    members = []
    for index in (1, 2):
        member_id = f"member_{index:02d}"
        raw = root / "training" / "members" / "raw" / f"{member_id}.model"
        ema = root / "training" / "members" / "ema" / f"{member_id}.model"
        raw.write_bytes(f"raw-{index}".encode())
        ema.write_bytes(f"ema-{index}".encode())
        from Uncertainty_Quantification.FGE.fge.artifacts import sha256_file

        members.append(
            {
                "member_id": member_id,
                "raw": {"path": raw.relative_to(root).as_posix(), "sha256": sha256_file(raw)},
                "ema": {"path": ema.relative_to(root).as_posix(), "sha256": sha256_file(ema)},
                "raw_metrics": {"energy_rmse": float(index), "forces_rmse": float(index + 1)},
            }
        )
    training = {
        "schema_version": "fge.training.v1",
        "base_model_metrics": {"energy_rmse": 2.0, "forces_rmse": 3.0},
        "members": members,
        "warnings": ([{"code": "weak_rmse", "member_id": "member_01"}] if warning else []),
    }
    atomic_write_json(root / "training" / "manifest.json", training)
    payload = {
        "schema_version": "fge.prediction.v1",
        "split": "test",
        "member_source": "raw",
        "member_ids": ["member_01", "member_02"],
        "observables": ["energy", "forces"],
        "energy_members": torch.tensor([[1.0, 3.0], [2.0, 5.0]], dtype=torch.float64),
        "forces_members": torch.tensor(
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]],
            dtype=torch.float64,
        ),
        "energy_reference": torch.tensor([1.5, 4.0], dtype=torch.float64),
        "forces_reference": torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64),
        "n_atoms": torch.tensor([1, 1], dtype=torch.int64),
        "atom_to_structure": torch.tensor([0, 1], dtype=torch.int64),
        "structure_ptr": torch.tensor([0, 1, 2], dtype=torch.int64),
    }
    prediction_path = root / "prediction" / "test_raw.pt"
    atomic_torch_save(prediction_path, payload)
    atomic_write_json(
        root / "prediction" / "manifest.json",
        build_prediction_manifest(
            root=root,
            prediction_path=prediction_path,
            member_count=2,
            structure_count=2,
            atom_count=2,
            observables=("energy", "forces"),
        ),
    )
    config = FakeConfig(root)
    evaluate_prediction(config, root)
    return config


def test_validator_detects_tampered_prediction(tmp_path: Path) -> None:
    config = _make_result(tmp_path)
    with (tmp_path / "prediction" / "test_raw.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(HardFailure, match="hash mismatch"):
        validate_result(config, tmp_path)


def test_quality_warning_does_not_fail_validation(tmp_path: Path) -> None:
    config = _make_result(tmp_path, warning=True)
    report = validate_result(config, tmp_path)
    assert report["status"] == "PASS"
    assert report["warnings"][0]["code"] == "weak_rmse"
    assert (tmp_path / "result_manifest.json").is_file()


def test_schema_signature_has_no_numeric_k_s_a(tmp_path: Path) -> None:
    config = _make_result(tmp_path)
    validate_result(config, tmp_path)
    signature = schema_signature(tmp_path)
    encoded = json.dumps(signature, sort_keys=True)
    assert "fge.prediction.v1" in encoded
    assert '"shape": ["K", "S"]' in encoded
