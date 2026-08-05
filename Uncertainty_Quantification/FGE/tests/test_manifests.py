from __future__ import annotations

from pathlib import Path

from Uncertainty_Quantification.FGE.fge.artifacts import atomic_write_json, sha256_file
from Uncertainty_Quantification.FGE.fge.manifests import (
    build_prediction_manifest,
    build_result_manifest,
    build_training_manifest,
)


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_training_manifest_uses_contiguous_members_and_current_hashes(tmp_path: Path):
    root = tmp_path / "run"
    base = _write(tmp_path / "base.model", b"base")
    raw_1 = _write(root / "training/members/raw/member_01.model", b"raw-1")
    ema_1 = _write(root / "training/members/ema/member_01.model", b"ema-1")
    raw_2 = _write(root / "training/members/raw/member_02.model", b"raw-2")
    ema_2 = _write(root / "training/members/ema/member_02.model", b"ema-2")

    manifest = build_training_manifest(
        root=root,
        project_name="case",
        k_requested=2,
        base_model_path=base,
        members=[
            {
                "member_id": "member_01",
                "cycle": 1,
                "raw_path": raw_1,
                "ema_path": ema_1,
                "raw_metrics": {"rmse_e_per_atom": 1.0, "rmse_f": 2.0},
                "ema_metrics": {"rmse_e_per_atom": 1.1, "rmse_f": 2.1},
                "frozen_backbone_verified": True,
            },
            {
                "member_id": "member_02",
                "cycle": 2,
                "raw_path": raw_2,
                "ema_path": ema_2,
                "raw_metrics": {"rmse_e_per_atom": 0.9, "rmse_f": 1.9},
                "ema_metrics": {"rmse_e_per_atom": 1.0, "rmse_f": 2.0},
                "frozen_backbone_verified": True,
            },
        ],
        warnings=[{"code": "weak_rmse", "member_id": "member_01"}],
    )

    assert manifest["k_requested"] == manifest["k_committed"] == 2
    assert manifest["base_model_sha256"] == sha256_file(base)
    assert manifest["members"][0]["raw"]["path"] == (
        "training/members/raw/member_01.model"
    )
    assert manifest["members"][1]["ema"]["sha256"] == sha256_file(ema_2)
    assert manifest["warnings"][0]["code"] == "weak_rmse"


def test_prediction_manifest_marks_only_raw_available(tmp_path: Path):
    root = tmp_path / "run"
    prediction = _write(root / "prediction/test_raw.pt", b"prediction")

    manifest = build_prediction_manifest(
        root=root,
        prediction_path=prediction,
        member_count=8,
        structure_count=19,
        atom_count=41,
        observables=("energy", "forces"),
    )

    assert manifest["branches"] == {"raw": "available", "ema": "not_generated"}
    assert manifest["artifact"]["sha256"] == sha256_file(prediction)
    assert manifest["shape_symbols"] == {"K": 8, "S": 19, "A": 41}


def test_result_manifest_excludes_work_and_itself(tmp_path: Path):
    root = tmp_path / "run"
    atomic_write_json(root / "validation.json", {"status": "PASS"})
    atomic_write_json(root / "training/manifest.json", {"schema_version": "fge.training.v1"})
    _write(root / "_work/resume/state.pt", b"transient")

    manifest = build_result_manifest(root=root, project_name="case")
    paths = {item["path"] for item in manifest["artifacts"]}

    assert paths == {"training/manifest.json", "validation.json"}
    assert "result_manifest.json" not in paths
    assert all(not path.startswith("_work/") for path in paths)
