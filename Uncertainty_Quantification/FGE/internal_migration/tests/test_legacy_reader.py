from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from internal_migration.migration.legacy_reader import read_legacy_run
from fge.errors import HardFailure


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_legacy_run(root: Path, *, corrupt_hash: bool = False) -> Path:
    (root / "members" / "raw").mkdir(parents=True)
    (root / "members" / "ema").mkdir(parents=True)
    members = []
    for index in range(1, 3):
        raw = root / "members" / "raw" / f"member_{index:02d}.model"
        ema = root / "members" / "ema" / f"member_{index:02d}_ema.model"
        raw.write_bytes(f"raw-{index}".encode())
        ema.write_bytes(f"ema-{index}".encode())
        members.append(
            {
                "status": "committed",
                "member_id": index,
                "cycle_index": index,
                "raw_model_path": raw.relative_to(root).as_posix(),
                "raw_sha256": "0" * 64 if corrupt_hash and index == 1 else _sha256(raw),
                "ema_model_path": ema.relative_to(root).as_posix(),
                "ema_sha256": _sha256(ema),
                "validation_raw": {"rmse_e_per_atom": 0.1 * index, "rmse_f": 0.2 * index},
                "validation_ema": {"rmse_e_per_atom": 0.11 * index, "rmse_f": 0.22 * index},
                "quality": {"status": "valid"},
                "quality_ema": {"status": "valid"},
            }
        )
    manifest = {
        "K_requested": 2,
        "K_valid": 2,
        "K_usable": 2,
        "crash": None,
        "base_validation_metrics": {"rmse_e_per_atom": 0.3, "rmse_f": 0.4},
        "members": members,
    }
    (root / "fge_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "predictions").mkdir()
    torch.save(
        {
            "E_members": torch.tensor([[1.0, 2.0], [1.2, 1.8]]),
            "F_members": torch.arange(24, dtype=torch.float32).reshape(2, 4, 3),
            "E_ref": torch.tensor([1.1, 1.9]),
            "F_ref": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "n_atoms": torch.tensor([1, 3]),
            "atom_to_structure": torch.tensor([0, 1, 1, 1]),
            "ptr": torch.tensor([0, 1, 4]),
            "member_ids": [1, 2],
            "member_source": "raw",
            "split": "test",
            "data_path": "/old/private/test.extxyz",
            "manifest_path": "/old/private/fge_manifest.json",
        },
        root / "predictions" / "test_raw_member_predictions.pt",
    )
    return root


def test_reader_validates_hashes_and_normalizes_prediction(tmp_path: Path) -> None:
    legacy = read_legacy_run(make_legacy_run(tmp_path / "legacy"))
    assert [member.member_id for member in legacy.members] == ["member_01", "member_02"]
    assert legacy.prediction["member_ids"] == ["member_01", "member_02"]
    assert legacy.prediction["energy_members"].dtype == torch.float64
    assert "data_path" not in legacy.prediction
    assert legacy.base_metrics == {"energy_rmse": 0.3, "forces_rmse": 0.4}


def test_reader_rejects_member_hash_mismatch(tmp_path: Path) -> None:
    with pytest.raises(HardFailure, match="hash mismatch"):
        read_legacy_run(make_legacy_run(tmp_path / "legacy", corrupt_hash=True))

