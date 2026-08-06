"""Integration tests for CPU-only test evaluation from committed cache artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from confidence_head.artifacts import load_torch_artifact
from confidence_head.cache import CacheWriter, ContinuousBatch
from confidence_head.config import load_config
from confidence_head.workflows.check_training import run_check_training
from confidence_head.workflows.evaluate import run_evaluate
from confidence_head.workflows.fit_bins import run_fit_bins
from confidence_head.workflows.train import run_train

from conftest import update_yaml, write_valid_config


def _batch(
    split: str,
    *,
    atom_counts: tuple[int, ...],
    energy_prediction: tuple[float, ...],
    force_prediction: torch.Tensor | None = None,
) -> ContinuousBatch:
    structures = len(atom_counts)
    atoms = sum(atom_counts)
    offsets = [0]
    for count in atom_counts:
        offsets.append(offsets[-1] + count)
    if force_prediction is None:
        force_prediction = (
            torch.arange(1, atoms + 1, dtype=torch.float32)
            .reshape(-1, 1)
            .repeat(1, 3)
            / 10.0
        )
    return ContinuousBatch(
        structure_index=torch.arange(structures, dtype=torch.int64),
        structure_id=tuple(f"{split}-{index}" for index in range(structures)),
        atomic_numbers=torch.ones(atoms, dtype=torch.int64),
        atom_offsets=torch.tensor(offsets, dtype=torch.int64),
        scalar_features=torch.arange(atoms * 640, dtype=torch.float32).reshape(
            atoms, 640
        )
        / 1000.0,
        force_prediction=force_prediction,
        force_reference=torch.zeros(atoms, 3, dtype=torch.float64),
        energy_prediction=torch.tensor(energy_prediction, dtype=torch.float32),
        energy_reference=torch.zeros(structures, dtype=torch.float64),
    )


def _completed_run(tmp_path: Path, *, branch: str):
    config_path = write_valid_config(tmp_path)
    updates: dict[str, object] = {
        "binning.force.num_bins": 3,
        "binning.force.max_error": 10.0,
        "binning.energy.num_bins": 3,
        "binning.energy.max_error": 10.0,
        "model.force.hidden_dims": [8],
        "model.energy.cumulant_order": 1,
        "model.energy.projection_dim": 512,
        "model.energy.hidden_dims": [8],
        "model.energy.adapter_dropout": 0.0,
        "model.energy.dropout": 0.0,
        "loss.force_coefficient": 1.0 if branch == "force" else 0.0,
        "loss.energy_coefficient": 1.0 if branch == "energy" else 0.0,
        "trainer.batch_size": 2,
        "trainer.max_epochs": 1,
        "trainer.early_stopping_patience": 1,
        "runtime.device": "cpu",
        "logging.wandb": False,
        "logging.wandb_mode": "disabled",
    }
    update_yaml(config_path, updates)
    config = load_config(config_path)

    from confidence_head.workflows import fit_bins as fit_workflow

    cache_id = fit_workflow._cache_identity(config)
    cache_root = config.run.output_root / config.run.name_prefix / "cache" / cache_id
    writer = CacheWriter(cache_root, cache_id=cache_id, shard_max_atoms=2)
    writer.append(
        _batch(
            "train",
            atom_counts=(1, 2),
            energy_prediction=(0.5, 1.5),
        ),
        split="train",
    )
    writer.finalize_split("train")
    writer.append(
        _batch(
            "validation",
            atom_counts=(2,),
            energy_prediction=(1.0,),
        ),
        split="validation",
    )
    writer.finalize_split("validation")
    writer.append(
        _batch(
            "test",
            atom_counts=(1, 2),
            energy_prediction=(2.0, 9.0),
            force_prediction=torch.tensor(
                [[1.0, 2.0, 3.0], [3.0, 0.0, 3.0], [0.0, 1.0, 2.0]],
                dtype=torch.float32,
            ),
        ),
        split="test",
    )
    writer.finalize_split("test")
    writer.finalize()

    binning_root = run_fit_bins(config)
    run_train(config)
    run_check_training(config)
    return config, binning_root.parent


def test_force_evaluation_streams_atom_mean_errors_and_reuses_result(
    tmp_path: Path,
) -> None:
    config, run_root = _completed_run(tmp_path, branch="force")

    manifest = run_evaluate(config)
    predictions = load_torch_artifact(manifest.parent / "test_predictions.pt")
    metrics = json.loads(
        (manifest.parent / "test_metrics.json").read_text(encoding="utf-8")
    )
    before = {path: path.stat().st_mtime_ns for path in (
        manifest,
        manifest.parent / "test_predictions.pt",
        manifest.parent / "test_metrics.json",
    )}

    assert predictions["enabled_branches"] == ("force",)
    assert predictions["structure_ids"] == ("test-0", "test-1")
    assert predictions["structure_offsets"].tolist() == [0, 1, 3]
    assert predictions["force"]["errors"].tolist() == pytest.approx([2.0, 2.0, 1.0])
    assert predictions["force"]["logits"].shape == (3, 3)
    assert predictions["force"]["logits"].device.type == "cpu"
    assert "energy" not in predictions
    assert metrics["branches"]["force"]["sample_count"] == 3

    assert run_evaluate(config) == manifest
    assert before == {path: path.stat().st_mtime_ns for path in before}


def test_energy_evaluation_divides_structure_error_by_atom_count(
    tmp_path: Path,
) -> None:
    config, _ = _completed_run(tmp_path, branch="energy")

    manifest = run_evaluate(config)
    predictions = load_torch_artifact(manifest.parent / "test_predictions.pt")

    assert predictions["enabled_branches"] == ("energy",)
    assert predictions["energy"]["errors"].tolist() == pytest.approx([2.0, 4.5])
    assert predictions["energy"]["logits"].shape == (2, 3)
    assert predictions["force_target_mode"] is None
    assert "force" not in predictions


def test_evaluation_uses_committed_training_identity_not_current_code_identity(
    tmp_path: Path, monkeypatch
) -> None:
    config, _ = _completed_run(tmp_path, branch="force")

    def forbid_recomputed_identity(*args, **kwargs):
        raise AssertionError("evaluation recomputed current source identity")

    monkeypatch.setattr(
        "confidence_head.workflows.train.code_identity", forbid_recomputed_identity
    )

    assert run_evaluate(config).name == "evaluation_manifest.json"


def test_evaluation_rejects_invalid_training_validation_without_outputs(
    tmp_path: Path,
) -> None:
    config, run_root = _completed_run(tmp_path, branch="force")
    validation_path = run_root / "run" / "training_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["valid"] = False
    validation_path.write_text(json.dumps(validation), encoding="utf-8")

    with pytest.raises(RuntimeError, match="training validation|sha256"):
        run_evaluate(config)
    assert not (run_root / "run" / "test_predictions.pt").exists()
    assert not (run_root / "run" / "evaluation_manifest.json").exists()


def test_evaluation_rejects_ambiguous_matching_run_directories(tmp_path: Path) -> None:
    config, run_root = _completed_run(tmp_path, branch="force")
    duplicate = run_root.parent / f"{run_root.name[:-12]}{'f' * 12}"
    duplicate.mkdir()

    with pytest.raises(RuntimeError, match="exactly one"):
        run_evaluate(config)
