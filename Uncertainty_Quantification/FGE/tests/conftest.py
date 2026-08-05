from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def minimal_config_dict() -> dict:
    return {
        "schema_version": "fge.v1",
        "project": {
            "name": "fge_test",
            "method": "FGE",
            "backend": "mace",
        },
        "paths": {
            "base_checkpoint": "data/base.model",
            "train_data": "data/train.extxyz",
            "val_data": "data/val.extxyz",
            "test_data": "data/test.extxyz",
            "output_root": "outputs/fge_test",
        },
        "data": {
            "format": "extxyz",
            "energy_key": "energy",
            "forces_key": "forces",
            "stress_key": "stress",
            "head_name": "default",
        },
        "training": {
            "mode": "readout_only_official_mace",
            "trainable_scope": "readouts",
            "expected_readout_parameter_count": 2192,
            "seed": 2026,
            "member_count": 2,
            "epochs_per_cycle": 1,
            "batch_size": 2,
            "validation_batch_size": 2,
            "weight_decay": 0.0,
            "energy_weight": 1.0,
            "forces_weight": 1.0,
            "stress_weight": 10.0,
            "huber_delta": 0.01,
            "beta": 0.9,
            "amsgrad": False,
            "max_grad_norm": 10.0,
            "device": "cpu",
        },
        "ema": {
            "enabled": True,
            "mode": "global",
            "decay": 0.99,
            "main_member_source": "raw",
        },
        "fge": {
            "schedule": "asymmetric_triangular",
            "rise_fraction": 0.2,
            "lr_min": 1.0e-8,
            "lr_max": 1.0e-7,
        },
        "prediction": {
            "split": "test",
            "member_source": "raw",
            "batch_size": 2,
            "compute_stress": False,
        },
        "ensemble": {
            "equal_weight": True,
            "validation_error_weighted": True,
            "eps_energy_ratio": 1.0e-3,
            "eps_force_ratio": 1.0e-3,
        },
        "evaluation": {
            "risk_coverages": [1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1],
            "force_structure_quantile": 0.95,
        },
        "quality": {
            "weak_rmse_multiplier": 1.5,
            "collapsed_rmse_multiplier": 3.0,
        },
        "wandb": {
            "enabled": True,
            "project": "FGE_MACE_UQ",
            "entity": None,
            "mode": "online",
        },
        "smoke": {
            "enabled": True,
            "allow_identical_splits": True,
            "max_train_structures": 20,
            "max_validation_structures": 20,
            "max_test_structures": 20,
            "max_train_batches": 2,
            "max_validation_batches": 2,
        },
    }


@pytest.fixture
def write_config(tmp_path: Path, minimal_config_dict: dict):
    def _write(updates: dict | None = None) -> Path:
        payload = deepcopy(minimal_config_dict)
        for section, values in (updates or {}).items():
            if isinstance(values, dict) and isinstance(payload.get(section), dict):
                payload[section].update(values)
            else:
                payload[section] = values
        path = tmp_path / "case.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    return _write
