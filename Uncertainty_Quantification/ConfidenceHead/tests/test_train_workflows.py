"""Integration tests for training orchestration and validation workflows."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch
import pytest

from confidence_head.artifacts import atomic_torch_save, load_torch_artifact
from confidence_head.cache import CacheWriter, ContinuousBatch
from confidence_head.config import load_config
from confidence_head.workflows.check_training import run_check_training
from confidence_head.workflows.fit_bins import run_fit_bins
from confidence_head.workflows.train import (
    ControlledEpochStop,
    RunConflictError,
    run_train,
)


from conftest import update_yaml, write_valid_config


def test_training_workflow_entry_points_are_callable() -> None:
    """Removing either public workflow entry point must break the smoke contract."""
    assert callable(run_train)
    assert callable(run_check_training)


def _batch(split: str, values: tuple[float, ...]) -> ContinuousBatch:
    structures = len(values)
    return ContinuousBatch(
        structure_index=torch.arange(structures, dtype=torch.long),
        structure_id=tuple(f"{split}-{index}" for index in range(structures)),
        atomic_numbers=torch.ones(structures, dtype=torch.long),
        atom_offsets=torch.arange(structures + 1, dtype=torch.long),
        scalar_features=torch.arange(structures * 640, dtype=torch.float32).reshape(
            structures, 640
        )
        / 1000.0,
        force_prediction=torch.tensor(values, dtype=torch.float32)
        .reshape(-1, 1)
        .repeat(1, 3),
        force_reference=torch.zeros(structures, 3, dtype=torch.float64),
        energy_prediction=torch.tensor(values, dtype=torch.float32),
        energy_reference=torch.zeros(structures, dtype=torch.float64),
    )


def _ready_force_run(tmp_path: Path, extra_updates: dict[str, object] | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = write_valid_config(tmp_path)
    updates: dict[str, object] = {
        "binning.force.num_bins": 3,
        "model.force.hidden_dims": [8],
        "loss.energy_coefficient": 0.0,
        "trainer.batch_size": 1,
        "trainer.max_epochs": 2,
        "trainer.early_stopping_patience": 2,
        "logging.wandb": False,
        "logging.wandb_mode": "disabled",
    }
    if extra_updates:
        updates.update(extra_updates)
    update_yaml(path, updates)
    config = load_config(path)
    from confidence_head.workflows import fit_bins as fit_workflow

    cache_identity = fit_workflow._cache_identity(config)
    cache_root = (
        config.run.output_root / config.run.name_prefix / "cache" / cache_identity
    )
    writer = CacheWriter(cache_root, cache_id=cache_identity, shard_max_atoms=8)
    for split, values in {
        "train": (0.02, 0.05, 0.08),
        "validation": (0.03, 0.06),
        "test": (0.04,),
    }.items():
        writer.append(_batch(split, values), split=split)
        writer.finalize_split(split)
    writer.finalize()
    binning_root = run_fit_bins(config)
    return config, binning_root


def test_train_and_check_commit_a_cache_only_force_run(tmp_path: Path) -> None:
    """Skipping orchestration or finalization must leave this release chain incomplete."""
    config, binning_root = _ready_force_run(tmp_path)

    best_path = run_train(config)
    validation_path = run_check_training(config)

    run_root = binning_root.parent
    run_dir = run_root / "run"
    assert best_path == run_dir / "best.pt"
    assert validation_path == run_dir / "training_validation.json"
    assert {path.name for path in run_dir.iterdir()} == {
        "best.pt",
        "events.jsonl",
        "last.pt",
        "training_manifest.json",
        "training_summary.json",
        "training_validation.json",
    }
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["epoch"] for event in events] == [0, 1]
    assert all(event["overflow"] == {"force": 0} for event in events)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["valid"] is True
    assert validation["best_epoch"] == min(
        range(len(events)), key=lambda index: events[index]["validation"]["total_loss"]
    )


def test_training_scripts_accept_only_config(monkeypatch, tmp_path: Path) -> None:
    from Uncertainty_Quantification.ConfidenceHead.scripts import (
        check_training as check_script,
    )
    from Uncertainty_Quantification.ConfidenceHead.scripts import train as train_script

    config_path = write_valid_config(tmp_path)
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        train_script,
        "run_train",
        lambda config: calls.append(("train", config.source_path)),
    )
    monkeypatch.setattr(
        check_script,
        "run_check_training",
        lambda config: calls.append(("check", config.source_path)),
    )
    assert train_script.main(["--config", str(config_path)]) == 0
    assert check_script.main(["--config", str(config_path)]) == 0
    assert calls == [
        ("train", config_path.resolve()),
        ("check", config_path.resolve()),
    ]
    for script in (train_script, check_script):
        with pytest.raises(SystemExit) as error:
            script.main(["--config", str(config_path), "--device", "cpu"])
        assert error.value.code == 2


def _state_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_state_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def test_controlled_epoch_stop_resumes_to_uninterrupted_state(tmp_path: Path) -> None:
    uninterrupted_config, _ = _ready_force_run(tmp_path / "full")
    uninterrupted_best = run_train(uninterrupted_config)
    uninterrupted_last = load_torch_artifact(uninterrupted_best.parent / "last.pt")

    resumed_config, resumed_bins = _ready_force_run(tmp_path / "resume")
    with pytest.raises(ControlledEpochStop):
        run_train(resumed_config, _stop_after_completed_epochs=1)
    resumed_run = resumed_bins.parent / "run"
    assert (resumed_run / "last.pt").is_file()
    assert not (resumed_run / "training_summary.json").exists()

    resumed_best = run_train(resumed_config)
    resumed_last = load_torch_artifact(resumed_best.parent / "last.pt")
    assert _state_equal(uninterrupted_last["model_state"], resumed_last["model_state"])
    assert _state_equal(
        uninterrupted_last["optimizer_state"], resumed_last["optimizer_state"]
    )
    assert (resumed_run / "events.jsonl").read_text(encoding="utf-8").count("\n") == 2


def test_completed_run_refuses_retraining_and_check_reuses_exact_pair(
    tmp_path: Path,
) -> None:
    config, _ = _ready_force_run(tmp_path)
    run_train(config)
    with pytest.raises(RunConflictError, match="complete"):
        run_train(config)

    validation = run_check_training(config)
    before = (
        validation.read_bytes(),
        (validation.parent / "training_manifest.json").read_bytes(),
    )
    assert run_check_training(config) == validation
    assert before == (
        validation.read_bytes(),
        (validation.parent / "training_manifest.json").read_bytes(),
    )


@pytest.mark.parametrize("corruption", ["event", "nonfinite_best", "unknown_root"])
def test_check_or_train_fails_closed_preserving_corrupt_evidence(
    tmp_path: Path, corruption: str
) -> None:
    config, bins = _ready_force_run(tmp_path)
    if corruption == "unknown_root":
        marker = bins.parent / "mystery.txt"
        marker.write_text("evidence", encoding="utf-8")
        with pytest.raises(RunConflictError, match="unknown"):
            run_train(config)
        assert marker.read_text(encoding="utf-8") == "evidence"
        return

    run_train(config)
    run_dir = bins.parent / "run"
    if corruption == "event":
        events_path = run_dir / "events.jsonl"
        lines = events_path.read_text(encoding="utf-8").splitlines()
        events_path.write_text("\n".join((lines[0], lines[0])) + "\n", encoding="utf-8")
        evidence = events_path.read_bytes()
    else:
        best_path = run_dir / "best.pt"
        payload = load_torch_artifact(best_path)
        next(iter(payload["model_state"].values())).fill_(float("nan"))
        atomic_torch_save(best_path, payload)
        evidence = best_path.read_bytes()

    with pytest.raises(RunConflictError):
        run_check_training(config)
    assert not (run_dir / "training_validation.json").exists()
    assert not (run_dir / "training_manifest.json").exists()
    if corruption == "event":
        assert (run_dir / "events.jsonl").read_bytes() == evidence
    else:
        assert (run_dir / "best.pt").read_bytes() == evidence


def test_energy_only_run_updates_projection_and_omits_force_state(
    tmp_path: Path,
) -> None:
    config, _ = _ready_force_run(
        tmp_path,
        {
            "loss.force_coefficient": 0.0,
            "loss.energy_coefficient": 0.3,
            "binning.energy.num_bins": 3,
            "model.energy.cumulant_order": 1,
            "model.energy.hidden_dims": [8],
            "trainer.max_epochs": 1,
        },
    )
    best = run_train(config)
    report = json.loads(run_check_training(config).read_text(encoding="utf-8"))
    state = load_torch_artifact(best)["model_state"]
    assert report["enabled_branches"] == ["energy"]
    assert report["energy_projection_updated"] is True
    assert any(name.startswith("energy_adapter.projection.") for name in state)
    assert all(not name.startswith("force_head.") for name in state)


def test_training_uses_epoch_seed_shuffle_and_complete_validation(
    tmp_path: Path, monkeypatch
) -> None:
    config, _ = _ready_force_run(tmp_path)
    from confidence_head.workflows import train as train_workflow

    shuffle_seeds: list[int] = []
    validation_counts: list[int] = []
    original_shuffled = train_workflow.iter_shuffled_cache_batches
    original_sequential = train_workflow.iter_cache_batches

    def shuffled(manifest, split, batch_size, *, seed):
        shuffle_seeds.append(seed)
        return original_shuffled(manifest, split, batch_size, seed=seed)

    def sequential(manifest, split, batch_size):
        batches = list(original_sequential(manifest, split, batch_size))
        if split == "validation":
            validation_counts.append(sum(len(batch.structure_id) for batch in batches))
        return iter(batches)

    monkeypatch.setattr(train_workflow, "iter_shuffled_cache_batches", shuffled)
    monkeypatch.setattr(train_workflow, "iter_cache_batches", sequential)
    run_train(config)
    assert shuffle_seeds == [config.runtime.seed, config.runtime.seed + 1]
    assert validation_counts == [2, 2]


def test_train_never_loads_mace_and_check_never_calls_forward(
    tmp_path: Path, monkeypatch
) -> None:
    config, _ = _ready_force_run(tmp_path)
    original_load = torch.load

    def guarded_load(path, *args, **kwargs):
        assert Path(path).name != "model.pt"
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    run_train(config)
    monkeypatch.setattr(
        "confidence_head.model.MultiBranchConfidenceModel.forward",
        lambda *args, **kwargs: pytest.fail("check must not run inference"),
    )
    run_check_training(config)


def test_resume_false_rejects_valid_epoch_boundary(tmp_path: Path) -> None:
    config, _ = _ready_force_run(
        tmp_path, {"trainer.resume": False, "trainer.max_epochs": 2}
    )
    with pytest.raises(ControlledEpochStop):
        run_train(config, _stop_after_completed_epochs=1)
    with pytest.raises(RunConflictError, match="resume is disabled"):
        run_train(config)


class _FakeCommError(Exception):
    pass


class _FakeAuthenticationError(Exception):
    pass


class _FakeWandbRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict, int]] = []
        self.finished = False

    def log(self, event, *, step):
        self.logs.append((dict(event), step))

    def finish(self):
        self.finished = True


class _FakeWandb:
    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class errors:
        CommError = _FakeCommError
        AuthenticationError = _FakeAuthenticationError

    def __init__(self) -> None:
        self.modes: list[str] = []
        self.run = _FakeWandbRun()

    def init(self, **kwargs):
        self.modes.append(kwargs["mode"])
        if kwargs["mode"] == "online":
            raise _FakeCommError("offline")
        (Path(kwargs["dir"]) / "wandb").mkdir(exist_ok=True)
        return self.run


def test_wandb_auto_connection_failure_falls_back_inside_run(
    tmp_path: Path, monkeypatch
) -> None:
    config, bins = _ready_force_run(
        tmp_path,
        {
            "logging.wandb": True,
            "logging.wandb_mode": "auto",
            "trainer.max_epochs": 1,
        },
    )
    fake = _FakeWandb()
    monkeypatch.setattr("confidence_head.logging._import_wandb", lambda: fake)

    run_train(config)

    run_dir = bins.parent / "run"
    summary = json.loads(
        (run_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    assert fake.modes == ["online", "offline"]
    assert fake.run.finished is True
    assert summary["wandb_mode"] == "offline"
    assert (run_dir / "wandb").is_dir()
    events = [event for event, _ in fake.run.logs]
    assert events and events[0]["overflow"] == {"force": 0}


def test_finalization_partial_pair_and_hash_mutation_fail_closed(
    tmp_path: Path,
) -> None:
    config, bins = _ready_force_run(tmp_path)
    run_train(config)
    validation = run_check_training(config)
    manifest = validation.parent / "training_manifest.json"
    validation_before = validation.read_bytes()
    manifest.unlink()

    with pytest.raises(RunConflictError, match="partial"):
        run_check_training(config)
    assert validation.read_bytes() == validation_before
    assert not manifest.exists()


def test_existing_completion_detects_support_file_hash_change(tmp_path: Path) -> None:
    config, bins = _ready_force_run(tmp_path)
    run_train(config)
    validation = run_check_training(config)
    run_dir = bins.parent / "run"
    best_path = run_dir / "best.pt"
    payload = load_torch_artifact(best_path)
    tensor = next(iter(payload["model_state"].values()))
    tensor.add_(1.0)
    atomic_torch_save(best_path, payload)
    completion_before = (
        validation.read_bytes(),
        (run_dir / "training_manifest.json").read_bytes(),
    )

    with pytest.raises(RunConflictError, match="validation differs"):
        run_check_training(config)
    assert completion_before == (
        validation.read_bytes(),
        (run_dir / "training_manifest.json").read_bytes(),
    )


@pytest.mark.parametrize("damage", ["missing_last", "corrupt_last", "event_behind"])
def test_partial_resume_and_event_history_are_rejected_without_cleanup(
    tmp_path: Path, damage: str
) -> None:
    config, bins = _ready_force_run(tmp_path)
    run_dir = bins.parent / "run"
    if damage in {"missing_last", "corrupt_last"}:
        with pytest.raises(ControlledEpochStop):
            run_train(config, _stop_after_completed_epochs=1)
        last_path = run_dir / "last.pt"
        if damage == "missing_last":
            last_path.unlink()
        else:
            last_path.write_bytes(b"not-a-checkpoint")
        evidence = {
            path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()
        }
        with pytest.raises(RunConflictError, match="last|checkpoint"):
            run_train(config)
    else:
        run_train(config)
        events_path = run_dir / "events.jsonl"
        events_path.write_text(
            events_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
            encoding="utf-8",
        )
        evidence = {
            path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()
        }
        with pytest.raises(RunConflictError, match="event"):
            run_check_training(config)

    assert evidence == {
        path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()
    }


def test_partial_immutable_snapshot_refuses_training(tmp_path: Path) -> None:
    config, bins = _ready_force_run(tmp_path)
    config_dir = bins.parent / "config"
    config_dir.mkdir()
    (config_dir / "source.yaml").write_bytes(config.source_path.read_bytes())
    before = (config_dir / "source.yaml").read_bytes()

    with pytest.raises(RunConflictError, match="partial"):
        run_train(config)

    assert (config_dir / "source.yaml").read_bytes() == before
    assert not (bins.parent / "run").exists()


def test_component_force_branch_has_three_component_labels_and_no_energy_state(
    tmp_path: Path,
) -> None:
    config, _ = _ready_force_run(
        tmp_path,
        {
            "model.force.target_mode": "component",
            "trainer.max_epochs": 1,
        },
    )
    best = run_train(config)
    event = json.loads(
        (best.parent / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    state = load_torch_artifact(best)["model_state"]

    assert event["train"]["force_samples"] == 9
    assert event["validation"]["force_samples"] == 6
    assert any(name.startswith("force_head.components.0.") for name in state)
    assert all(not name.startswith("energy_") for name in state)


def test_best_checkpoint_must_match_strict_minimum_event(tmp_path: Path) -> None:
    config, bins = _ready_force_run(tmp_path)
    best_path = run_train(config)
    events = [
        json.loads(line)
        for line in (best_path.parent / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    payload = load_torch_artifact(best_path)
    wrong_epoch = 1 - payload["epoch"]
    payload["epoch"] = wrong_epoch
    payload["validation_metrics"] = events[wrong_epoch]["validation"]
    atomic_torch_save(best_path, payload)

    with pytest.raises(RunConflictError, match="strict minimum"):
        run_check_training(config)
    assert not (bins.parent / "run" / "training_manifest.json").exists()


def test_train_script_failure_is_nonzero_from_external_directory(
    tmp_path: Path,
) -> None:
    config_path = write_valid_config(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "cache manifest is missing" in result.stderr


def test_check_rejects_false_complete_epoch_sample_counts(tmp_path: Path) -> None:
    config, bins = _ready_force_run(tmp_path)
    run_train(config)
    events_path = bins.parent / "run" / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    events[0]["train"]["force_samples"] = 1
    events_path.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )

    with pytest.raises(RunConflictError, match="sample count"):
        run_check_training(config)
