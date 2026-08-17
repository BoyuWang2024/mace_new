"""Resumable dataset-scoped prediction using formal FGE raw members read-only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import ExperimentLayout, atomic_write_json, sha256_file
from .data import iter_extxyz_shards
from .dataset_spec import DatasetSpec
from .derived_artifacts import (
    DerivedLayout,
    build_prediction_shard_signature,
    verify_prediction_shard_if_present,
    write_prediction_shard,
)
from .errors import HardFailure
from .prediction import generate_prediction_payload


def _pass_marker(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HardFailure(f"formal result marker cannot be loaded: {path.name}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        raise HardFailure(f"formal result marker must have PASS status: {path.name}")


def load_training_manifest(config: Any) -> dict[str, Any]:
    """Load and verify formal raw members without writing the formal result root."""
    layout = ExperimentLayout(Path(config.output_dir))
    _pass_marker(layout.validation)
    _pass_marker(layout.result_manifest)
    try:
        manifest = json.loads(layout.training_manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HardFailure("training manifest cannot be loaded") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("members"), list):
        raise HardFailure("training manifest has no committed members")
    members = manifest["members"]
    expected_ids = [f"member_{index:02d}" for index in range(1, len(members) + 1)]
    if len(members) < 2 or [member.get("member_id") for member in members] != expected_ids:
        raise HardFailure("training manifest members are not contiguous")
    for member in members:
        artifact = member.get("raw")
        if not isinstance(artifact, dict):
            raise HardFailure("training manifest raw member is invalid")
        relative = artifact.get("path")
        expected_hash = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise HardFailure("training manifest raw member is invalid")
        model_path = layout.root / relative
        if not model_path.is_file() or sha256_file(model_path) != expected_hash:
            raise HardFailure("raw member is missing or hash mismatched")
    return manifest


def _publish_manifest(path: Path, payload: dict[str, Any]) -> Path:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HardFailure("derived prediction manifest cannot be loaded") from exc
        if existing != payload:
            raise HardFailure("derived prediction manifest does not match this run")
        return path
    atomic_write_json(path, payload)
    return path


def predict_dataset(config: Any, spec: DatasetSpec, outputs_root: Path) -> Path:
    """Predict one explicit extxyz dataset while preserving formal results read-only."""
    training = load_training_manifest(config)
    member_ids = [member["member_id"] for member in training["members"]]
    layout = DerivedLayout(Path(outputs_root), str(config.project_name), spec.label)
    formal_root = Path(config.output_dir).resolve()
    derived_root = layout.root.resolve()
    if derived_root == formal_root or formal_root in derived_root.parents:
        raise HardFailure("derived prediction output cannot be inside a formal result")

    shard_rows: list[dict[str, Any]] = []
    structure_count = 0
    atom_count = 0
    for shard in iter_extxyz_shards(
        spec.path,
        keys=spec.keys,
        required=set(spec.required),
        head_name=spec.head_name,
        shard_size=spec.shard_size,
    ):
        shard_atoms = sum(
            len(configuration.atomic_numbers) for configuration in shard.configurations
        )
        atom_stop = atom_count + shard_atoms
        signature = build_prediction_shard_signature(
            dataset=spec.label,
            observables=spec.observables,
            member_ids=member_ids,
            shard_index=shard.index,
            structure_start=shard.structure_start,
            structure_stop=shard.structure_stop,
            atom_start=atom_count,
            atom_stop=atom_stop,
            batch_size=spec.batch_size,
        )
        if not verify_prediction_shard_if_present(layout, shard.index, signature):
            payload = generate_prediction_payload(
                config,
                training,
                shard.configurations,
                compute_stress=spec.compute_stress,
                batch_size=spec.batch_size,
            )
            write_prediction_shard(layout, shard.index, payload, signature)
        verify_prediction_shard_if_present(layout, shard.index, signature)
        shard_manifest = layout.prediction_shard_manifest(shard.index)
        shard_rows.append(
            {
                "index": shard.index,
                "manifest": {
                    "path": shard_manifest.relative_to(layout.prediction_dir).as_posix(),
                    "sha256": sha256_file(shard_manifest),
                },
            }
        )
        structure_count = shard.structure_stop
        atom_count = atom_stop
    if not shard_rows:
        raise HardFailure("derived prediction produced no shards")

    return _publish_manifest(
        layout.prediction_manifest,
        {
            "schema_version": "fge.derived-prediction.v1",
            "status": "PASS",
            "experiment": config.project_name,
            "dataset": spec.neutral_manifest(),
            "member_source": "raw",
            "member_ids": member_ids,
            "observables": list(spec.observables),
            "shape_symbols": {
                "K": len(member_ids),
                "S": structure_count,
                "A": atom_count,
            },
            "shard_count": len(shard_rows),
            "shards": shard_rows,
        },
    )
