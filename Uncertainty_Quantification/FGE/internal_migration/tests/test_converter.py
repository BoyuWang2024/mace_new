from __future__ import annotations

import json
from pathlib import Path

import torch

from fge.config import load_config
from fge.validation import schema_signature
from internal_migration.migration.converter import convert_legacy_run
from internal_migration.tests.test_legacy_reader import make_legacy_run


def test_converter_builds_provenance_free_valid_result(
    tmp_path: Path, minimal_config_dict: dict
) -> None:
    import yaml

    legacy_root = make_legacy_run(tmp_path / "legacy")
    base = tmp_path / "base.model"
    base.write_bytes(b"base")
    minimal_config_dict["paths"]["base_checkpoint"] = str(base)
    minimal_config_dict["paths"]["output_root"] = str(tmp_path / "configured-output")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(minimal_config_dict, sort_keys=False), encoding="utf-8")
    config = load_config(config_path)
    destination = tmp_path / "formal-result"
    audit_path = tmp_path / "internal-audit.json"

    result = convert_legacy_run(
        config=config,
        legacy_root=legacy_root,
        destination=destination,
        audit_path=audit_path,
    )

    assert result == destination.resolve()
    assert (destination / "result_manifest.json").is_file()
    assert not (destination / "checkpoints").exists()
    prediction = torch.load(destination / "prediction" / "test_raw.pt", weights_only=True)
    assert set(prediction) == {
        "schema_version", "split", "member_source", "member_ids", "observables",
        "energy_members", "forces_members", "energy_reference", "forces_reference",
        "n_atoms", "atom_to_structure", "structure_ptr",
    }
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in destination.rglob("*.json")
    )
    assert str(legacy_root) not in serialized
    assert "/old/private" not in serialized
    assert json.loads(audit_path.read_text(encoding="utf-8"))["status"] == "PASS"
    assert schema_signature(destination)["schema_version"] == "fge.schema-signature.v1"

