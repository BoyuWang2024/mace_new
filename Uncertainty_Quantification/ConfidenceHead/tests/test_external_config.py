from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from confidence_head.config import load_config
from confidence_head.external_config import ExternalConfigError, load_external_config

from conftest import update_yaml, write_valid_config


def _matrix(tmp_path: Path):
    force_path = write_valid_config(tmp_path)
    update_yaml(
        force_path,
        {
            "profile": "production",
            "loss.force_coefficient": 1.0,
            "loss.energy_coefficient": 0.0,
        },
    )
    force = load_config(force_path)
    energy = {}
    for order in range(1, 9):
        path = tmp_path / f"energy-{order}.yaml"
        path.write_bytes(force_path.read_bytes())
        update_yaml(
            path,
            {
                "loss.force_coefficient": 0.0,
                "loss.energy_coefficient": 1.0,
                "model.energy.cumulant_order": order,
            },
        )
        energy[order] = load_config(path)
    return force, energy


def _write_external(tmp_path: Path) -> Path:
    (tmp_path / "mad.xyz").write_text("fixture\n", encoding="utf-8")
    document = {
        "schema_version": 1,
        "dataset": {
            "name": "mad_test",
            "path": "mad.xyz",
            "expected_sha256": "a" * 64,
            "expected_structures": 9486,
            "expected_atoms": 258586,
            "source_index_path": "mad-source-index.csv",
        },
        "production_config_dir": ".",
        "output_root": "external",
        "plot_root": "plots",
        "cache": {
            "split": "inference",
            "build_batch_size": 32,
            "shard_max_atoms": 100000,
            "resume": True,
        },
        "runtime": {"device": "cuda", "head_batch_size": 32},
    }
    path = tmp_path / "external.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_load_external_config_resolves_paths_and_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = _matrix(tmp_path)
    monkeypatch.setattr(
        "confidence_head.external_config.discover_production_matrix",
        lambda path: matrix,
    )

    config = load_external_config(_write_external(tmp_path))

    assert config.dataset.name == "mad_test"
    assert config.dataset.path == (tmp_path / "mad.xyz").resolve()
    assert config.dataset.source_index_path == (
        tmp_path / "mad-source-index.csv"
    ).resolve()
    assert config.cache.split == "inference"
    assert config.runtime_device == "cuda"
    assert config.force_config == matrix[0]
    assert list(config.energy_configs) == list(range(1, 9))


def test_load_external_config_rejects_unknown_dataset_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "confidence_head.external_config.discover_production_matrix",
        lambda path: _matrix(tmp_path),
    )
    path = _write_external(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["dataset"]["unexpected"] = True
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ExternalConfigError, match="dataset keys"):
        load_external_config(path)


def test_external_config_rejects_non_inference_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "confidence_head.external_config.discover_production_matrix",
        lambda path: _matrix(tmp_path),
    )
    path = _write_external(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["cache"]["split"] = "test"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ExternalConfigError, match="inference"):
        load_external_config(path)
