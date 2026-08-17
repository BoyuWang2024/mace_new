from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.inference_config import (
    load_inference_config,
)


def _document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": {
            "run": "../outputs/mace_readout_B8_full",
            "member_count": 8,
            "parameter_mode": "raw",
            "stage": "best",
        },
        "datasets": {
            "matpes_test": {
                "kind": "existing_result",
                "source": "../../../data/dataset/matpes_test.extxyz",
                "domains": ["energy", "forces", "stress"],
                "existing_split": "test",
            },
            "mad_test": {
                "kind": "predict",
                "source": "../../../data/dataset/mad-test.xyz",
                "domains": ["energy", "forces"],
            },
            "matpes_train": {
                "kind": "predict",
                "source": "../../../data/dataset/matpes_train.extxyz",
                "domains": ["energy", "forces", "stress"],
            },
        },
        "prediction": {
            "batch_size": 16,
            "max_structures_per_chunk": 1024,
            "max_atoms_per_chunk": 50000,
            "device": "cuda",
            "precision": "float64",
            "num_workers": 0,
        },
        "uncertainty": {
            "ddof": 1,
            "compute_std": True,
            "compute_gmd": True,
            "gmd_pairs": "distinct_unordered",
        },
        "plot": {
            "output_root": "../../Plots/BootStrapping",
            "dpi": 300,
            "figure_size": [7.0, 7.0],
            "scatter_max_points": 20000,
            "scatter_seed": 20260817,
            "grid_size": 160,
            "gaussian_sigma": 1.2,
            "contour_masses": [0.5, 0.7, 0.85, 0.95, 0.99],
        },
    }


def _write_config(tmp_path: Path, document: dict[str, object] | None = None) -> Path:
    path = tmp_path / "configs" / "request.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(document or _document(), sort_keys=False), encoding="utf-8")
    return path


def test_completed_inference_config_declares_exact_contract(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    loaded = load_inference_config(config_path)

    assert loaded.source.member_count == 8
    assert loaded.source.parameter_mode == "raw"
    assert loaded.source.stage == "best"
    assert loaded.source.run == (config_path.parent / "../outputs/mace_readout_B8_full").resolve()
    assert loaded.datasets["matpes_test"].kind == "existing_result"
    assert loaded.datasets["matpes_test"].domains == ("energy", "forces", "stress")
    assert loaded.datasets["matpes_test"].existing_split == "test"
    assert loaded.datasets["mad_test"].domains == ("energy", "forces")
    assert loaded.datasets["mad_test"].existing_split is None
    assert loaded.datasets["matpes_train"].domains == ("energy", "forces", "stress")
    assert loaded.prediction.device == "cuda"
    assert loaded.uncertainty.ddof == 1
    assert loaded.plot.scatter_max_points == 20000


def test_config_rejects_mad_stress(tmp_path: Path) -> None:
    document = _document()
    document["datasets"]["mad_test"]["domains"] = ["energy", "forces", "stress"]  # type: ignore[index]
    with pytest.raises(HardFailure, match="mad_test.*stress"):
        load_inference_config(_write_config(tmp_path, document))


def test_config_rejects_existing_split_on_predict_dataset(tmp_path: Path) -> None:
    document = _document()
    document["datasets"]["mad_test"]["existing_split"] = "test"  # type: ignore[index]
    with pytest.raises(HardFailure, match="existing_split"):
        load_inference_config(_write_config(tmp_path, document))


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    document = _document()
    document["prediction"]["adaptive_batching"] = True  # type: ignore[index]
    with pytest.raises(HardFailure, match="unknown key prediction.adaptive_batching"):
        load_inference_config(_write_config(tmp_path, document))
