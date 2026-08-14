from __future__ import annotations

import json
from pathlib import Path

import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.manifests import build_run_manifest
from Uncertainty_Quantification.BootStrapping.bootstrap.schema import RUN_SCHEMA
from Uncertainty_Quantification.BootStrapping.bootstrap.validation import validate_run


def _minimal_run(root: Path) -> Path:
    files = []
    for index in range(2):
        member = root / "members" / f"member_{index:03d}"
        for mode in ("raw", "ema"):
            for stage in ("best", "final"):
                path = member / "models" / f"{mode}_{stage}.model"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"model")
                files.append(path)
        resume = member / "resume" / "latest.pt"
        resume.parent.mkdir()
        resume.write_bytes(b"resume")
        files.append(resume)
    origin = root / "origin_manifest.json"
    origin.write_text(json.dumps({"schema": "mace.bootstrap.origin/v1"}), encoding="utf-8")
    files.append(origin)
    build_run_manifest(root, schema=RUN_SCHEMA, artifacts=files, metadata={"member_count": 2, "parameter_modes": ["raw", "ema"], "splits": ["val", "test"], "status": "complete"})
    return root


def test_validator_accepts_complete_hash_authenticated_run(tmp_path: Path) -> None:
    report = validate_run(_minimal_run(tmp_path / "run"), require_predictions=False)
    assert report.member_count == 2
    assert report.model_count == 8
    assert report.resume_count == 2


def test_validator_rejects_unmanifested_file_and_absolute_origin(tmp_path: Path) -> None:
    root = _minimal_run(tmp_path / "run")
    (root / "extra.bin").write_bytes(b"hidden")
    with pytest.raises(HardFailure, match="unmanifested"):
        validate_run(root, require_predictions=False)
