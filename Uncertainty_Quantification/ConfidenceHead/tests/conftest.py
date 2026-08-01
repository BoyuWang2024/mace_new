"""Test fixtures for ConfidenceHead configuration parsing."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def write_valid_config(tmp_path: Path, *, profile: str = "smoke_test") -> Path:
    """Write a complete valid configuration and every input file it names."""
    for filename in ("model.pt", "train.extxyz", "validation.extxyz", "test.extxyz"):
        (tmp_path / filename).write_text("fixture\n", encoding="utf-8")

    document = {
        "profile": profile,
        "checkpoint": {
            "path": "model.pt",
            "expected_sha256": "1" * 64,
            "feature_modules": [
                {"name": "products.0", "expected_dim": 512},
                {"name": "products.1", "expected_dim": 128},
            ],
        },
        "data": {
            "train": {"path": "train.extxyz", "expected_sha256": "2" * 64},
            "validation": {
                "path": "validation.extxyz",
                "expected_sha256": "3" * 64,
            },
            "test": {"path": "test.extxyz", "expected_sha256": "4" * 64},
        },
        "cache": {
            "build_batch_size": 32,
            "shard_max_atoms": 100000,
            "num_workers": 0,
            "pin_memory": True,
            "resume": True,
        },
        "binning": {
            "algorithm": "fixed_linear_v1",
            "force": {"num_bins": 50, "max_error": 0.3},
            "energy": {"num_bins": 50, "max_error": 0.5},
        },
        "model": {
            "force": {
                "target_mode": "atom_mean",
                "hidden_dims": [256, 256, 256],
                "dropout": 0.05,
            },
            "energy": {
                "cumulant_order": 3,
                "projection_dim": 512,
                "adapter_dropout": 0.1,
                "hidden_dims": [256, 256],
                "dropout": 0.2,
                "signed_root": True,
            },
        },
        "loss": {"force_coefficient": 1.0, "energy_coefficient": 0.3},
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "trainer": {
            "batch_size": 32,
            "max_epochs": 100,
            "early_stopping_patience": 20,
            "resume": True,
        },
        "runtime": {"seed": 1234, "device": "cpu", "deterministic": True},
        "logging": {
            "jsonl": True,
            "wandb": False,
            "wandb_project": "mace-confidence-head",
            "wandb_mode": "disabled",
        },
        "run": {"name_prefix": "unit", "output_root": "outputs"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def update_yaml(path: Path, updates: dict[str, Any]) -> None:
    """Apply dotted-key updates to a YAML document without bypassing parsing."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    for dotted_key, value in updates.items():
        keys = dotted_key.split(".")
        target = document
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
