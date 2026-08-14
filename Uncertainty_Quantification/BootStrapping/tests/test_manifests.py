from __future__ import annotations

import json
from pathlib import Path

import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure


def test_manifest_uses_relative_paths_and_audits_files(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.manifests import build_run_manifest, validate_manifest

    root = tmp_path / "run"
    artifact = root / "members" / "member_000" / "best.model"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model")
    manifest_path = build_run_manifest(root, schema="mace.bootstrap.run/v1", artifacts=[artifact], metadata={"run_id": "tiny"})
    manifest = validate_manifest(manifest_path, expected_schema="mace.bootstrap.run/v1")
    assert manifest["artifacts"][0]["path"] == "members/member_000/best.model"
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")


def test_manifest_rejects_hash_drift_and_escape(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.manifests import build_run_manifest, validate_manifest

    root = tmp_path / "run"
    root.mkdir()
    artifact = root / "value.bin"
    artifact.write_bytes(b"first")
    manifest_path = build_run_manifest(root, schema="mace.bootstrap.run/v1", artifacts=[artifact])
    artifact.write_bytes(b"other")
    with pytest.raises(HardFailure, match="SHA-256|size drift"):
        validate_manifest(manifest_path, expected_schema="mace.bootstrap.run/v1")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["artifacts"][0]["path"] = "../outside"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(HardFailure, match="escapes"):
        validate_manifest(manifest_path, expected_schema="mace.bootstrap.run/v1")
