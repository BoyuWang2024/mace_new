from __future__ import annotations

import json
from pathlib import Path

import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.completed_run import (
    audit_completed_run,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.manifests import build_run_manifest
from Uncertainty_Quantification.BootStrapping.bootstrap.schema import RUN_SCHEMA


def _completed_run(root: Path, member_count: int = 8) -> Path:
    artifacts: list[Path] = []
    for index in range(member_count):
        member = root / "members" / f"member_{index:03d}"
        for mode in ("raw", "ema"):
            for stage in ("best", "final"):
                path = member / "models" / f"{mode}_{stage}.model"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"model-{index}-{mode}-{stage}".encode("ascii"))
                artifacts.append(path)
        resume = member / "resume" / "latest.pt"
        resume.parent.mkdir(parents=True)
        resume.write_bytes(f"resume-{index}".encode("ascii"))
        artifacts.append(resume)
    origin = root / "origin_manifest.json"
    origin.write_text(json.dumps({"schema": "mace.bootstrap.origin/v1"}), encoding="utf-8")
    artifacts.append(origin)
    build_run_manifest(
        root,
        schema=RUN_SCHEMA,
        artifacts=artifacts,
        metadata={
            "member_count": member_count,
            "parameter_modes": ["raw", "ema"],
            "splits": ["val", "test"],
            "status": "complete",
        },
    )
    return root


def test_audit_completed_run_binds_ordered_raw_best_models(tmp_path: Path) -> None:
    root = _completed_run(tmp_path / "run")
    source = audit_completed_run(root, expected_members=8)

    assert source.root == root.resolve()
    assert source.member_count == 8
    assert [member.index for member in source.members] == list(range(8))
    assert [member.path.relative_to(root).as_posix() for member in source.members] == [
        f"members/member_{index:03d}/models/raw_best.model" for index in range(8)
    ]
    assert all(len(member.sha256) == 64 for member in source.members)
    assert len(source.run_manifest_sha256) == 64


def test_audit_completed_run_rejects_member_count_mismatch(tmp_path: Path) -> None:
    with pytest.raises(HardFailure, match="member count"):
        audit_completed_run(_completed_run(tmp_path / "run", member_count=2), expected_members=8)


def test_audit_completed_run_rejects_model_hash_drift(tmp_path: Path) -> None:
    root = _completed_run(tmp_path / "run")
    path = root / "members" / "member_003" / "models" / "raw_best.model"
    path.write_bytes(path.read_bytes() + b"drift")

    with pytest.raises(HardFailure, match="SHA-256|size drift"):
        audit_completed_run(root, expected_members=8)
