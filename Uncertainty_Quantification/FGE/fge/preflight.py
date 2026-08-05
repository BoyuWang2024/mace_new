"""Stage-specific, forward-free gates for the FGE workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .artifacts import ExperimentLayout, atomic_write_json, sha256_file
from .data import load_extxyz
from .errors import HardFailure
from .members import freeze_readouts


def _path(config: Any, name: str) -> Path:
    return Path(config.section("paths")[name])


def _load_model(path: Path, device: str = "cpu") -> torch.nn.Module:
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise HardFailure(f"model cannot be loaded: {path}") from exc
    if not isinstance(model, torch.nn.Module):
        raise HardFailure("checkpoint does not contain a complete torch model")
    return model


def run_preflight(config: Any, stage: str) -> dict[str, Any]:
    """Validate only the inputs needed by one stage, without any model forward pass."""
    if stage not in {"train", "predict", "evaluate"}:
        raise HardFailure("preflight stage must be train, predict, or evaluate")
    layout = ExperimentLayout(config.output_dir)
    report: dict[str, Any] = {
        "schema_version": "fge.preflight.v1",
        "stage": stage,
        "status": "PASS",
    }
    if stage == "train":
        base_path = _path(config, "base_checkpoint")
        if not base_path.is_file():
            raise HardFailure("base checkpoint is missing")
        data_config = config.section("data")
        keys = {
            "energy": data_config["energy_key"],
            "forces": data_config["forces_key"],
            "stress": data_config["stress_key"],
            "head": data_config["head_name"],
        }
        train = load_extxyz(
            _path(config, "train_data"), keys=keys, required={"energy", "forces", "stress"}, head_name=data_config["head_name"]
        )
        validation = load_extxyz(
            _path(config, "val_data"), keys=keys, required={"energy", "forces", "stress"}, head_name=data_config["head_name"]
        )
        model = _load_model(base_path)
        guard = freeze_readouts(
            model, config.section("training")["expected_readout_parameter_count"]
        )
        report.update(
            {
                "base_checkpoint_sha256": sha256_file(base_path),
                "train_structure_count": len(train),
                "validation_structure_count": len(validation),
                "trainable_parameter_count": sum(
                    parameter.numel() for _, parameter in guard.trainable_parameters
                ),
            }
        )
    elif stage == "predict":
        manifest_path = layout.training_manifest
        if not manifest_path.is_file():
            raise HardFailure("training manifest is missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HardFailure("training manifest cannot be parsed") from exc
        members = manifest.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise HardFailure("training manifest has no committed ensemble")
        for member in members:
            artifact = member.get("raw", {})
            member_path = layout.root / artifact.get("path", "")
            if not member_path.is_file() or sha256_file(member_path) != artifact.get("sha256"):
                raise HardFailure("raw member is missing or hash mismatched")
        data_config = config.section("data")
        configurations = load_extxyz(
            _path(config, "test_data"),
            keys={
                "energy": data_config["energy_key"],
                "forces": data_config["forces_key"],
                "stress": data_config["stress_key"],
                "head": data_config["head_name"],
            },
            required={"energy", "forces"},
            head_name=data_config["head_name"],
        )
        report.update({"member_count": len(members), "test_structure_count": len(configurations)})
    else:
        if not layout.training_manifest.is_file():
            raise HardFailure("training manifest is missing")
        if not layout.prediction_tensor.is_file():
            raise HardFailure("canonical prediction is missing")
        try:
            training = json.loads(layout.training_manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HardFailure("training manifest cannot be parsed") from exc
        if "members" not in training or "base_model_metrics" not in training:
            raise HardFailure("training metrics required for evaluation are missing")
        report["prediction_available"] = True
    atomic_write_json(layout.preflight_dir / f"{stage}.json", report)
    return report
