"""Cache-only workflow contracts for immutable fitted binning artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from confidence_head.artifacts import (
    atomic_json_dump,
    atomic_torch_save,
    load_torch_artifact,
)
from confidence_head.cache import CacheIncompleteError, CacheWriter, ContinuousBatch
from confidence_head.config import load_config
from confidence_head.run_naming import make_run_tag
from confidence_head.workflows import fit_bins as workflow
from conftest import update_yaml, write_valid_config


def _configure_log_binning(path: Path) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["binning"]["algorithm"] = "train_quantile_log_v1"
    document["binning"]["force"].pop("max_error")
    document["binning"]["energy"].pop("max_error")
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def _batch(
    *,
    split: str,
    force_values: list[float],
    energy_values: list[float],
) -> ContinuousBatch:
    structures = len(energy_values)
    assert len(force_values) == structures
    return ContinuousBatch(
        structure_index=torch.arange(structures, dtype=torch.long),
        structure_id=tuple(f"{split}-{index}" for index in range(structures)),
        atomic_numbers=torch.ones(structures, dtype=torch.long),
        atom_offsets=torch.arange(structures + 1, dtype=torch.long),
        scalar_features=torch.zeros(structures, 640, dtype=torch.float32),
        force_prediction=torch.tensor(force_values, dtype=torch.float32)
        .reshape(-1, 1)
        .repeat(1, 3),
        force_reference=torch.zeros(structures, 3, dtype=torch.float64),
        energy_prediction=torch.tensor(energy_values, dtype=torch.float32),
        energy_reference=torch.zeros(structures, dtype=torch.float64),
    )


def _write_complete_cache(config) -> Path:
    identity = workflow._cache_identity(config)
    root = config.run.output_root / config.run.name_prefix / "cache" / identity
    writer = CacheWriter(root, cache_id=identity, shard_max_atoms=10)
    writer.append(
        _batch(
            split="train",
            force_values=[0.1, 0.2],
            energy_values=[0.3, 0.4],
        ),
        split="train",
    )
    writer.finalize_split("train")
    writer.append(
        _batch(
            split="validation",
            force_values=[100.0],
            energy_values=[100.0],
        ),
        split="validation",
    )
    writer.finalize_split("validation")
    writer.append(
        _batch(
            split="test",
            force_values=[200.0],
            energy_values=[200.0],
        ),
        split="test",
    )
    writer.finalize_split("test")
    writer.finalize()
    return root


def _load_branch(root: Path, name: str) -> dict:
    return load_torch_artifact(root / "binning.pt")["branches"][name]


def test_fit_bins_reads_only_train_from_complete_cache(tmp_path: Path) -> None:
    config = load_config(write_valid_config(tmp_path))
    _write_complete_cache(config)

    root = workflow.run_fit_bins(config)

    force = _load_branch(root, "force")
    energy = _load_branch(root, "energy")
    assert sum(force["counts"].tolist()) == 2
    assert sum(energy["counts"].tolist()) == 2
    assert force["overflow_count"] == 0
    assert energy["overflow_count"] == 0
    assert force["thresholds"].device.type == "cpu"
    assert force["thresholds"].dtype is torch.float64

    artifact = load_torch_artifact(root / "binning.pt")
    assert set(artifact) == {
        "schema_version",
        "formula_version",
        "cache_id",
        "experiment_id",
        "binning_id",
        "run_id",
        "algorithm",
        "branches",
    }
    expected_directory = f"{make_run_tag(config)}-{artifact['experiment_id'][:12]}"
    assert root == (
        config.run.output_root
        / config.run.name_prefix
        / "runs"
        / expected_directory
        / "binning"
    )
    manifest = json.loads((root / "binning_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version",
        "formula_version",
        "artifact_filename",
        "artifact_sha256",
        "cache_id",
        "experiment_id",
        "binning_id",
        "run_id",
        "algorithm",
        "enabled_branches",
        "train_structure_count",
        "train_atom_count",
        "branches",
    }
    assert manifest["artifact_filename"] == "binning.pt"
    assert manifest["enabled_branches"] == ["energy", "force"]
    assert manifest["train_structure_count"] == 2
    assert manifest["train_atom_count"] == 2
    common = {
        "num_bins",
        "thresholds",
        "reps",
        "counts",
        "sources",
        "overflow_count",
    }
    fixed = common | {"max_error", "bin_width"}
    assert set(artifact["branches"]["force"]) == fixed
    assert set(manifest["branches"]["force"]) == fixed


def test_fit_bins_omits_disabled_branch(monkeypatch, tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"loss.force_coefficient": 0.0})
    config = load_config(path)
    _write_complete_cache(config)
    monkeypatch.setattr(workflow, "force_errors", pytest.fail)

    root = workflow.run_fit_bins(config)

    artifact = load_torch_artifact(root / "binning.pt")
    assert set(artifact["branches"]) == {"energy"}


def test_fit_bins_energy_disabled_never_computes_energy_errors(
    monkeypatch, tmp_path: Path
) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"loss.energy_coefficient": 0.0})
    config = load_config(path)
    _write_complete_cache(config)
    monkeypatch.setattr(workflow, "energy_errors", pytest.fail)

    root = workflow.run_fit_bins(config)
    assert set(load_torch_artifact(root / "binning.pt")["branches"]) == {"force"}


def test_fit_bins_reuses_identical_artifacts_without_rewriting(
    tmp_path: Path,
) -> None:
    config = load_config(write_valid_config(tmp_path))
    _write_complete_cache(config)
    root = workflow.run_fit_bins(config)
    artifact_path = root / "binning.pt"
    manifest_path = root / "binning_manifest.json"
    before = (artifact_path.read_bytes(), manifest_path.read_bytes())

    assert workflow.run_fit_bins(config) == root

    assert (artifact_path.read_bytes(), manifest_path.read_bytes()) == before


def test_binning_id_ignores_training_hyperparameters(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    first_config = load_config(path)
    _write_complete_cache(first_config)
    first_root = workflow.run_fit_bins(first_config)
    first = load_torch_artifact(first_root / "binning.pt")

    update_yaml(path, {"optimizer.learning_rate": 0.002})
    second_config = load_config(path)
    assert workflow._cache_identity(second_config) == first["cache_id"]
    second_root = workflow.run_fit_bins(second_config)
    second = load_torch_artifact(second_root / "binning.pt")

    assert first["experiment_id"] != second["experiment_id"]
    assert first["binning_id"] == second["binning_id"]
    assert first["run_id"] != second["run_id"]


def test_fit_bins_rejects_existing_different_artifact_without_overwrite(
    tmp_path: Path,
) -> None:
    config = load_config(write_valid_config(tmp_path))
    _write_complete_cache(config)
    root = workflow.run_fit_bins(config)
    artifact_path = root / "binning.pt"
    artifact = load_torch_artifact(artifact_path)
    artifact["branches"]["force"]["max_error"] = 0.9
    atomic_torch_save(artifact_path, artifact)
    before = (
        artifact_path.read_bytes(),
        (root / "binning_manifest.json").read_bytes(),
    )

    with pytest.raises(workflow.BinningIdentityError, match="differs"):
        workflow.run_fit_bins(config)

    assert (
        artifact_path.read_bytes(),
        (root / "binning_manifest.json").read_bytes(),
    ) == before


def test_fit_bins_rejects_manifest_identity_mismatch(tmp_path: Path) -> None:
    config = load_config(write_valid_config(tmp_path))
    _write_complete_cache(config)
    root = workflow.run_fit_bins(config)
    manifest_path = root / "binning_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["binning_id"] = "0" * 64
    atomic_json_dump(manifest_path, manifest)

    with pytest.raises(workflow.BinningIdentityError, match="identity"):
        workflow.run_fit_bins(config)


def test_fit_bins_requires_committed_complete_cache(tmp_path: Path) -> None:
    config = load_config(write_valid_config(tmp_path))
    identity = workflow._cache_identity(config)
    root = config.run.output_root / config.run.name_prefix / "cache" / identity
    writer = CacheWriter(root, cache_id=identity, shard_max_atoms=10)
    writer.append(
        _batch(split="train", force_values=[0.1], energy_values=[0.2]),
        split="train",
    )
    writer.finalize_split("train")

    with pytest.raises(CacheIncompleteError):
        workflow.run_fit_bins(config)


def test_fit_bins_rejects_incomplete_artifact_pair_without_overwrite(
    tmp_path: Path,
) -> None:
    config = load_config(write_valid_config(tmp_path))
    _write_complete_cache(config)
    root = workflow.run_fit_bins(config)
    artifact_path = root / "binning.pt"
    before = artifact_path.read_bytes()
    (root / "binning_manifest.json").unlink()

    with pytest.raises(workflow.BinningIdentityError, match="both"):
        workflow.run_fit_bins(config)

    assert artifact_path.read_bytes() == before


def test_train_quantile_algorithm_is_loaded_from_config(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    _configure_log_binning(path)

    config = load_config(path)
    assert config.binning.algorithm == "train_quantile_log_v1"
    assert not hasattr(config.binning.force, "max_error")
    assert not hasattr(config.binning.energy, "max_error")


def test_log_artifact_and_manifest_use_exact_branch_schema(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    _configure_log_binning(path)
    config = load_config(path)
    _write_complete_cache(config)

    root = workflow.run_fit_bins(config)
    artifact = load_torch_artifact(root / "binning.pt")
    manifest = json.loads((root / "binning_manifest.json").read_text(encoding="utf-8"))
    common = {
        "num_bins",
        "thresholds",
        "reps",
        "counts",
        "sources",
        "overflow_count",
    }
    expected = common | {"lower", "upper"}
    assert set(artifact["branches"]["force"]) == expected
    assert set(manifest["branches"]["force"]) == expected


def test_fixed_binning_requires_exact_fixed_branch_fields(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["binning"]["force"].pop("max_error")
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="missing.*max_error"):
        load_config(path)

    document["binning"]["force"]["max_error"] = 0.3
    document["binning"]["force"]["lower"] = 0.01
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown.*lower"):
        load_config(path)


def test_log_binning_rejects_fixed_or_extra_branch_fields(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"binning.algorithm": "train_quantile_log_v1"})

    with pytest.raises(ValueError, match="unknown.*max_error"):
        load_config(path)

    _configure_log_binning(path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["binning"]["energy"]["upper"] = 1.0
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown.*upper"):
        load_config(path)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({}, "unit_linear_atommean-f50_e50-order3"),
        (
            {"loss.force_coefficient": 0.0},
            "unit_linear_foff_e50-order3",
        ),
        (
            {
                "binning.algorithm": "train_quantile_log_v1",
                "model.force.target_mode": "component",
                "loss.energy_coefficient": 0.0,
            },
            "unit_log_component-f50_eoff",
        ),
    ],
)
def test_run_tag_is_human_readable_and_deterministic(
    tmp_path: Path, updates: dict[str, object], expected: str
) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, updates)
    if updates.get("binning.algorithm") == "train_quantile_log_v1":
        _configure_log_binning(path)

    assert make_run_tag(load_config(path)) == expected


def test_script_only_accepts_config(monkeypatch, tmp_path: Path) -> None:
    from Uncertainty_Quantification.ConfidenceHead.scripts import fit_bins as script

    config_path = write_valid_config(tmp_path)
    called: list[Path] = []
    monkeypatch.setattr(
        script, "run_fit_bins", lambda config: called.append(config.source_path)
    )

    assert script.main(["--config", str(config_path)]) == 0
    assert called == [config_path.resolve()]
    with pytest.raises(SystemExit) as error:
        script.main(["--config", str(config_path), "--device", "cpu"])
    assert error.value.code == 2
