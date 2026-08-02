"""Checkpoint, epoch training, and exact epoch-resume contracts."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from confidence_head.artifacts import load_torch_artifact
from confidence_head.binning import (
    BinningArtifact,
    fit_fixed_linear,
    labels_from_thresholds,
)
from confidence_head.cache import ContinuousBatch
from confidence_head.checkpoint import (
    BEST_CHECKPOINT_KEYS,
    LAST_CHECKPOINT_KEYS,
    EarlyStoppingState,
    TrainingState,
    load_last,
    save_best,
    save_last,
)
from confidence_head.config import load_config
from confidence_head.logging import JsonlLogger, TrainingLogger, WandbMirror
from confidence_head.labels import energy_errors, force_errors
from confidence_head.model import MultiBranchConfidenceModel
from confidence_head.runtime import capture_rng_state, configure_runtime
from confidence_head.trainer import ConfidenceTrainer

from conftest import update_yaml, write_valid_config


IDENTITY = {
    "run_id": "run-test",
    "experiment_id": "experiment-test",
    "cache_id": "cache-test",
    "binning_id": "bins-test",
}


def _config(tmp_path: Path, **updates: Any):
    path = write_valid_config(tmp_path)
    if updates:
        update_yaml(path, updates)
    return load_config(path)


def _model(config, *, feature_dim: int = 4) -> MultiBranchConfidenceModel:
    return MultiBranchConfidenceModel.from_config(config, feature_dim=feature_dim)


def _prime_adamw(model: nn.Module, optimizer: torch.optim.AdamW) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _artifact(config) -> BinningArtifact:
    branches = {}
    if config.force_enabled:
        branches["force"] = fit_fixed_linear(
            torch.tensor([0.02, 0.11], dtype=torch.float64),
            num_bins=config.binning.force.num_bins,
            max_error=config.binning.force.max_error,
        )
    if config.energy_enabled:
        branches["energy"] = fit_fixed_linear(
            torch.tensor([0.01, 0.07], dtype=torch.float64),
            num_bins=config.binning.energy.num_bins,
            max_error=config.binning.energy.max_error,
        )
    artifact = BinningArtifact.create(
        cache_id=IDENTITY["cache_id"],
        experiment_id=IDENTITY["experiment_id"],
        algorithm="fixed_linear_v1",
        branches=branches,
        force_target_mode=(
            config.model.force.target_mode if config.force_enabled else None
        ),
    )
    return replace(
        artifact,
        binning_id=IDENTITY["binning_id"],
        run_id=IDENTITY["run_id"],
    )


def _batch(
    *,
    feature_shift: float = 0.0,
    force_error: float = 0.03,
    energy_error: float = 0.02,
    atom_counts: tuple[int, ...] = (2, 1),
) -> ContinuousBatch:
    offsets = torch.tensor((0,) + tuple(np.cumsum(atom_counts)), dtype=torch.long)
    atoms = int(offsets[-1])
    structures = len(atom_counts)
    features = (
        torch.arange(atoms * 4, dtype=torch.float32).reshape(atoms, 4) / 10
        + feature_shift
    )
    return ContinuousBatch(
        structure_index=torch.arange(structures),
        structure_id=tuple(f"s-{feature_shift}-{i}" for i in range(structures)),
        atomic_numbers=torch.ones(atoms, dtype=torch.long),
        atom_offsets=offsets,
        scalar_features=features,
        force_prediction=torch.full((atoms, 3), force_error),
        force_reference=torch.zeros((atoms, 3)),
        energy_prediction=torch.tensor(
            [energy_error * count for count in atom_counts], dtype=torch.float32
        ),
        energy_reference=torch.zeros(structures),
    )


def _logger(path: Path) -> TrainingLogger:
    return TrainingLogger(JsonlLogger.resume(path), WandbMirror("disabled", None))


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_nested_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def test_best_and_last_checkpoint_exact_model_only_and_resume_keys(tmp_path: Path):
    config = _config(tmp_path)
    model = _model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _prime_adamw(model, optimizer)
    metrics = {"force_loss": 0.4, "energy_loss": 0.2, "total_loss": 0.46}

    best_path = save_best(
        tmp_path / "best.pt",
        model=model,
        identity=IDENTITY,
        epoch=2,
        validation_metrics=metrics,
    )
    state = TrainingState(3, 2, 0.46, 0, False)
    last_path = save_last(
        tmp_path / "last.pt",
        model=model,
        optimizer=optimizer,
        identity=IDENTITY,
        validation_metrics=metrics,
        state=state,
    )

    best = load_torch_artifact(best_path)
    last = load_torch_artifact(last_path)
    assert set(best) == BEST_CHECKPOINT_KEYS
    assert set(last) == LAST_CHECKPOINT_KEYS
    assert set(last) - set(best) == {
        "next_epoch",
        "optimizer_state",
        "early_stopping_state",
        "best_epoch",
        "best_validation_loss",
        "rng_state",
        "completed",
    }
    assert not ({"optimizer_state", "rng_state", "next_epoch"} & set(best))
    for name, saved in best["model_state"].items():
        assert saved.device.type == "cpu"
        assert saved.data_ptr() != model.state_dict()[name].data_ptr()
    torch.load(best_path, map_location="cpu", weights_only=True)
    torch.load(last_path, map_location="cpu", weights_only=True)


def test_checkpoint_identity_is_exact_and_rolling_save_replaces(tmp_path: Path):
    config = _config(tmp_path)
    model = _model(config)
    path = tmp_path / "best.pt"
    save_best(
        path,
        model=model,
        identity=IDENTITY,
        epoch=0,
        validation_metrics={"total_loss": 1.0},
    )
    with torch.no_grad():
        next(model.parameters()).add_(1)
    save_best(
        path,
        model=model,
        identity=IDENTITY,
        epoch=1,
        validation_metrics={"total_loss": 0.5},
    )
    assert load_torch_artifact(path)["epoch"] == 1
    assert list(tmp_path.glob(".best.pt.*.tmp")) == []

    for bad in (
        {**IDENTITY, "extra": "x"},
        {key: value for key, value in IDENTITY.items() if key != "run_id"},
        {**IDENTITY, "run_id": ""},
    ):
        with pytest.raises((TypeError, ValueError), match="identity"):
            save_best(
                path,
                model=model,
                identity=bad,
                epoch=1,
                validation_metrics={"total_loss": 0.5},
            )


def test_save_last_rejects_optimizer_bound_to_another_model(tmp_path: Path):
    config = _config(tmp_path)
    model = _model(config)
    other = _model(config)
    with pytest.raises(ValueError, match="optimizer"):
        save_last(
            tmp_path / "last.pt",
            model=model,
            optimizer=torch.optim.AdamW(other.parameters(), lr=1e-3),
            identity=IDENTITY,
            validation_metrics={"total_loss": 0.5},
            state=TrainingState(1, 0, 0.5, 0, False),
        )


def test_save_load_and_trainer_reject_non_adamw_optimizer(tmp_path: Path):
    config = _config(tmp_path)
    model = _model(config)
    sgd = torch.optim.SGD(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="AdamW"):
        save_last(
            tmp_path / "sgd-last.pt",
            model=model,
            optimizer=sgd,
            identity=IDENTITY,
            validation_metrics={"total_loss": 0.5},
            state=TrainingState(1, 0, 0.5, 0, False),
        )

    adamw = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _prime_adamw(model, adamw)
    path = save_last(
        tmp_path / "last.pt",
        model=model,
        optimizer=adamw,
        identity=IDENTITY,
        validation_metrics={"total_loss": 0.5},
        state=TrainingState(1, 0, 0.5, 0, False),
    )
    target = _model(config)
    target_sgd = torch.optim.SGD(target.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="AdamW"):
        load_last(path, target, target_sgd, IDENTITY)

    logger = _logger(tmp_path / "events-sgd.jsonl")
    try:
        with pytest.raises(ValueError, match="AdamW"):
            ConfidenceTrainer(
                model=target,
                optimizer=target_sgd,
                config=config,
                binning=_artifact(config),
                device=torch.device("cpu"),
                logger=logger,
                identity=IDENTITY,
                best_path=tmp_path / "best.pt",
                last_path=tmp_path / "trainer-last.pt",
            )
    finally:
        logger.close()


def _first_optimizer_entry(payload: dict[str, Any]) -> dict[str, Any]:
    return next(iter(payload["optimizer_state"]["state"].values()))


def _drop_optimizer_parameter_state(payload: dict[str, Any]) -> None:
    state = payload["optimizer_state"]["state"]
    state.pop(next(iter(state)))


def _add_optimizer_parameter_state(payload: dict[str, Any]) -> None:
    state = payload["optimizer_state"]["state"]
    state[max(state) + 1] = copy.deepcopy(next(iter(state.values())))


def _append_optimizer_param_group(payload: dict[str, Any]) -> None:
    groups = payload["optimizer_state"]["param_groups"]
    groups.append(copy.deepcopy(groups[0]))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(schema_version=999), "version"),
        (lambda payload: payload.update(run_id="other"), "identity"),
        (
            lambda payload: payload["model_state"].pop(
                next(iter(payload["model_state"]))
            ),
            "model",
        ),
        (
            lambda payload: payload["parameter_schema"].pop(
                next(iter(payload["parameter_schema"]))
            ),
            "schema",
        ),
        (
            lambda payload: _first_optimizer_entry(payload).update(
                exp_avg=_first_optimizer_entry(payload)["exp_avg"].reshape(-1)
            ),
            "optimizer",
        ),
        (
            lambda payload: _first_optimizer_entry(payload).update(
                exp_avg_sq=_first_optimizer_entry(payload)["exp_avg_sq"].double()
            ),
            "optimizer",
        ),
        (
            lambda payload: _first_optimizer_entry(payload).pop("exp_avg"),
            "optimizer",
        ),
        (
            lambda payload: _first_optimizer_entry(payload).update(
                unexpected=torch.tensor(0.0)
            ),
            "optimizer",
        ),
        (
            lambda payload: _first_optimizer_entry(payload).update(
                step=torch.tensor(-1.5)
            ),
            "optimizer",
        ),
        (_drop_optimizer_parameter_state, "optimizer"),
        (_add_optimizer_parameter_state, "optimizer"),
        (_append_optimizer_param_group, "optimizer"),
        (
            lambda payload: payload["optimizer_state"]["param_groups"][0].update(
                params=list(
                    reversed(payload["optimizer_state"]["param_groups"][0]["params"])
                )
            ),
            "optimizer",
        ),
        (
            lambda payload: payload["optimizer_state"]["param_groups"][0].update(
                lr=0.002
            ),
            "optimizer",
        ),
        (lambda payload: payload.pop("optimizer_state"), "keys"),
        (lambda payload: payload.pop("rng_state"), "keys"),
        (
            lambda payload: payload["early_stopping_state"].update(bad_epochs=-1),
            "early",
        ),
        (lambda payload: payload.update(best_validation_loss=float("nan")), "finite"),
        (
            lambda payload: next(iter(payload["model_state"].values())).fill_(
                float("nan")
            ),
            "finite",
        ),
        (
            lambda payload: payload["optimizer_state"]["param_groups"][0].update(
                lr=float("nan")
            ),
            "optimizer",
        ),
        (
            lambda payload: payload["rng_state"].update(
                torch_cpu=torch.zeros(1, dtype=torch.uint8)
            ),
            "RNG",
        ),
        (lambda payload: payload.update(created_at="not-a-time"), "ISO"),
    ],
)
def test_load_last_rejects_corruption_before_model_or_rng_mutation(
    tmp_path: Path, mutation, match: str
):
    config = _config(tmp_path)
    source = _model(config)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    _prime_adamw(source, source_optimizer)
    path = save_last(
        tmp_path / "last.pt",
        model=source,
        optimizer=source_optimizer,
        identity=IDENTITY,
        validation_metrics={"total_loss": 0.5},
        state=TrainingState(1, 0, 0.5, 0, False),
    )
    payload = load_torch_artifact(path)
    mutation(payload)
    torch.save(payload, path)

    target = _model(config)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    before_model = copy.deepcopy(target.state_dict())
    before_optimizer = copy.deepcopy(target_optimizer.state_dict())
    before_rng = capture_rng_state()
    with pytest.raises((TypeError, ValueError, RuntimeError), match=match):
        load_last(path, target, target_optimizer, IDENTITY)
    assert _nested_equal(before_model, target.state_dict())
    assert _nested_equal(before_optimizer, target_optimizer.state_dict())
    assert _nested_equal(before_rng, capture_rng_state())


def test_load_last_rejects_parameter_shape_and_dtype_mismatch(tmp_path: Path):
    config = _config(tmp_path)
    model = _model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _prime_adamw(model, optimizer)
    path = save_last(
        tmp_path / "last.pt",
        model=model,
        optimizer=optimizer,
        identity=IDENTITY,
        validation_metrics={"total_loss": 0.5},
        state=TrainingState(1, 0, 0.5, 0, False),
    )
    for field in ("shape", "dtype"):
        payload = load_torch_artifact(path)
        name = next(iter(payload["parameter_schema"]))
        payload["parameter_schema"][name][field] = (
            [999] if field == "shape" else "torch.float64"
        )
        torch.save(payload, path)
        with pytest.raises(ValueError, match="schema"):
            load_last(path, model, optimizer, IDENTITY)
        save_last(
            path,
            model=model,
            optimizer=optimizer,
            identity=IDENTITY,
            validation_metrics={"total_loss": 0.5},
            state=TrainingState(1, 0, 0.5, 0, False),
        )


@pytest.mark.parametrize(
    ("state", "current_total", "mutation"),
    [
        (
            TrainingState(2, 0, 0.5, 1, False),
            0.6,
            lambda payload: payload["early_stopping_state"].update(bad_epochs=0),
        ),
        (
            TrainingState(2, 0, 0.5, 1, False),
            0.6,
            lambda payload: payload.update(epoch=0),
        ),
        (
            TrainingState(1, 0, 0.5, 0, False),
            0.5,
            lambda payload: payload["validation_metrics"].update(total_loss=0.6),
        ),
        (
            TrainingState(2, 0, 0.5, 1, False),
            0.6,
            lambda payload: payload["validation_metrics"].update(total_loss=0.4),
        ),
    ],
)
def test_load_last_rejects_inconsistent_resume_semantics_before_mutation(
    tmp_path: Path,
    state: TrainingState,
    current_total: float,
    mutation,
):
    config = _config(tmp_path)
    source = _model(config)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    _prime_adamw(source, source_optimizer)
    path = save_last(
        tmp_path / "semantic-last.pt",
        model=source,
        optimizer=source_optimizer,
        identity=IDENTITY,
        validation_metrics={"total_loss": current_total},
        state=state,
    )
    payload = load_torch_artifact(path)
    mutation(payload)
    torch.save(payload, path)

    target = _model(config)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    before_model = copy.deepcopy(target.state_dict())
    before_optimizer = copy.deepcopy(target_optimizer.state_dict())
    before_rng = capture_rng_state()
    with pytest.raises(ValueError, match="checkpoint|early|best|validation"):
        load_last(path, target, target_optimizer, IDENTITY)
    assert _nested_equal(before_model, target.state_dict())
    assert _nested_equal(before_optimizer, target_optimizer.state_dict())
    assert _nested_equal(before_rng, capture_rng_state())


def test_early_stopping_strict_improvement_equality_and_patience():
    state = EarlyStoppingState.initial()
    state, improved = state.observe(0, 1.0)
    assert improved and state == EarlyStoppingState(0, 1.0, 0)
    equal, improved = state.observe(1, 1.0)
    assert not improved and equal.bad_epochs == 1
    worse, improved = equal.observe(2, 2.0)
    assert not improved and worse.should_stop(2)
    better, improved = worse.observe(3, 0.9)
    assert improved and better.bad_epochs == 0 and better.best_epoch == 3
    with pytest.raises(ValueError, match="finite"):
        better.observe(4, math.nan)
    with pytest.raises(ValueError, match="patience"):
        better.should_stop(True)


@pytest.mark.parametrize(
    ("updates", "expected_prefix", "forbidden"),
    [
        ({"loss.energy_coefficient": 0.0}, "force_head.", "energy"),
        (
            {"loss.force_coefficient": 0.0, "loss.energy_coefficient": 0.3},
            "energy_adapter.",
            "force",
        ),
    ],
)
def test_checkpoint_state_contains_enabled_branches_only(
    tmp_path: Path, updates, expected_prefix: str, forbidden: str
):
    config = _config(tmp_path, **updates)
    model = _model(config)
    path = save_best(
        tmp_path / "best.pt",
        model=model,
        identity=IDENTITY,
        epoch=0,
        validation_metrics={"total_loss": 1.0},
    )
    keys = load_torch_artifact(path)["model_state"]
    assert any(name.startswith(expected_prefix) for name in keys)
    assert all(forbidden not in name for name in keys)


def test_train_and_validation_use_modes_grad_contract_and_all_uneven_samples(
    tmp_path: Path,
):
    config = _config(tmp_path, **{"loss.energy_coefficient": 0.0})
    model = _model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logger = _logger(tmp_path / "events.jsonl")
    trainer = ConfidenceTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        binning=_artifact(config),
        device=torch.device("cpu"),
        logger=logger,
        identity=IDENTITY,
        best_path=tmp_path / "best.pt",
        last_path=tmp_path / "last.pt",
    )
    modes: list[tuple[bool, bool, int]] = []
    original = model.forward

    def sentinel(features, offsets):
        modes.append((model.training, torch.is_grad_enabled(), features.shape[0]))
        return original(features, offsets)

    model.forward = sentinel  # type: ignore[method-assign]
    train = trainer.train_epoch([_batch(atom_counts=(2, 1)), _batch(atom_counts=(1,))])
    assert modes == [(True, True, 3), (True, True, 1)]
    modes.clear()
    before = copy.deepcopy(model.state_dict())
    before_rng = capture_rng_state()
    validation = trainer.validate_epoch(
        [_batch(atom_counts=(2, 1)), _batch(atom_counts=(1,))]
    )
    assert modes == [(False, False, 3), (False, False, 1)]
    assert _nested_equal(before, model.state_dict())
    assert _nested_equal(before_rng, capture_rng_state())
    assert train["force_samples"] == validation["force_samples"] == 4
    assert train["energy_samples"] == validation["energy_samples"] == 0
    logger.close()


def test_component_and_energy_sample_counts_are_true_counts(tmp_path: Path):
    config = _config(tmp_path, **{"model.force.target_mode": "component"})
    model = _model(config)
    trainer = ConfidenceTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        config=config,
        binning=_artifact(config),
        device=torch.device("cpu"),
        logger=_logger(tmp_path / "events.jsonl"),
        identity=IDENTITY,
        best_path=tmp_path / "best.pt",
        last_path=tmp_path / "last.pt",
    )
    result = trainer.validate_epoch(
        [_batch(atom_counts=(2, 1)), _batch(atom_counts=(1,))]
    )
    assert result["force_samples"] == 12
    assert result["energy_samples"] == 3
    assert result["total_loss"] == pytest.approx(
        config.loss.force_coefficient * result["force_loss"]
        + config.loss.energy_coefficient * result["energy_loss"]
    )
    trainer.logger.close()


def test_uneven_batch_metrics_match_independent_hard_ce(tmp_path: Path):
    config = _config(tmp_path)
    model = _model(config)
    artifact = _artifact(config)
    trainer = ConfidenceTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        config=config,
        binning=artifact,
        device=torch.device("cpu"),
        logger=_logger(tmp_path / "events-parity.jsonl"),
        identity=IDENTITY,
        best_path=tmp_path / "best.pt",
        last_path=tmp_path / "last.pt",
    )
    batches = [
        _batch(atom_counts=(1,), feature_shift=0.1),
        _batch(atom_counts=(1, 1, 1), feature_shift=0.4),
    ]
    force_logits: list[torch.Tensor] = []
    energy_logits: list[torch.Tensor] = []
    calls = 0

    def fixed_logits(features: torch.Tensor, offsets: torch.Tensor):
        nonlocal calls
        force = torch.zeros(features.shape[0], config.binning.force.num_bins)
        energy = torch.zeros(offsets.numel() - 1, config.binning.energy.num_bins)
        force[:, 0] = float(calls + 1)
        energy[:, 1] = float(2 * calls + 1)
        calls += 1
        force_logits.append(force)
        energy_logits.append(energy)
        return {"force": force, "energy": energy}

    model.forward = fixed_logits  # type: ignore[method-assign]
    result = trainer.validate_epoch(batches)
    force_labels = torch.cat(
        [
            labels_from_thresholds(
                force_errors(
                    batch.force_prediction, batch.force_reference, "atom_mean"
                ),
                artifact.branches["force"].thresholds,
            )
            for batch in batches
        ]
    )
    energy_labels = torch.cat(
        [
            labels_from_thresholds(
                energy_errors(
                    batch.energy_prediction,
                    batch.energy_reference,
                    batch.atom_offsets[1:] - batch.atom_offsets[:-1],
                ),
                artifact.branches["energy"].thresholds,
            )
            for batch in batches
        ]
    )
    expected_force = F.cross_entropy(torch.cat(force_logits), force_labels)
    expected_energy = F.cross_entropy(torch.cat(energy_logits), energy_labels)
    assert result["force_samples"] == 4
    assert result["energy_samples"] == 4
    assert result["force_loss"] == pytest.approx(expected_force.item())
    assert result["energy_loss"] == pytest.approx(expected_energy.item())
    assert result["total_loss"] == pytest.approx(
        config.loss.force_coefficient * expected_force.item()
        + config.loss.energy_coefficient * expected_energy.item()
    )
    trainer.logger.close()


@pytest.mark.parametrize("method", ["train_epoch", "validate_epoch"])
def test_epochs_reject_empty_batches_and_nonfinite_data(tmp_path: Path, method: str):
    config = _config(tmp_path, **{"loss.energy_coefficient": 0.0})
    model = _model(config)
    trainer = ConfidenceTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        config=config,
        binning=_artifact(config),
        device=torch.device("cpu"),
        logger=_logger(tmp_path / f"{method}.jsonl"),
        identity=IDENTITY,
        best_path=tmp_path / "best.pt",
        last_path=tmp_path / "last.pt",
    )
    with pytest.raises(ValueError, match="empty"):
        getattr(trainer, method)([])
    batch = _batch()
    batch.scalar_features[0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError), match="finite"):
        getattr(trainer, method)([batch])
    trainer.logger.close()


@pytest.mark.parametrize(
    ("updates", "state"),
    [
        (
            {"trainer.max_epochs": 8, "trainer.early_stopping_patience": 2},
            TrainingState(3, 0, 0.5, 2, False),
        ),
        (
            {"trainer.max_epochs": 3, "trainer.early_stopping_patience": 4},
            TrainingState(3, 2, 0.5, 0, False),
        ),
    ],
)
def test_fit_rejects_unmarked_terminal_resume_before_requesting_batches(
    tmp_path: Path, updates: dict[str, Any], state: TrainingState
):
    config = _config(tmp_path, **updates)
    model = _model(config)
    logger = _logger(tmp_path / "terminal-events.jsonl")
    for epoch in range(state.next_epoch):
        logger.append_epoch({"epoch": epoch})
    trainer = ConfidenceTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        config=config,
        binning=_artifact(config),
        device=torch.device("cpu"),
        logger=logger,
        identity=IDENTITY,
        best_path=tmp_path / "best.pt",
        last_path=tmp_path / "last.pt",
    )
    requested: list[int] = []

    def train_factory(epoch: int):
        requested.append(epoch)
        return []

    try:
        with pytest.raises(ValueError, match="completed|terminal"):
            trainer.fit(train_factory, lambda: [], state=state)
        assert requested == []
    finally:
        logger.close()


class ControlledEpochStop(RuntimeError):
    pass


def _read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run_training(root: Path, *, stop_after: int | None, resume: bool):
    root.mkdir(parents=True, exist_ok=True)
    config = _config(
        root, **{"trainer.max_epochs": 4, "trainer.early_stopping_patience": 4}
    )
    configure_runtime(9123, True, "cpu")
    model = _model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    logger = _logger(root / "events.jsonl")
    trainer = ConfidenceTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        binning=_artifact(config),
        device=torch.device("cpu"),
        logger=logger,
        identity=IDENTITY,
        best_path=root / "best.pt",
        last_path=root / "last.pt",
    )
    state = None
    if resume:
        state = load_last(root / "last.pt", model, optimizer, IDENTITY)

    def train_factory(epoch: int):
        generator = torch.Generator().manual_seed(config.runtime.seed + epoch)
        order = torch.randperm(3, generator=generator).tolist()
        return [_batch(feature_shift=float(index) / 10) for index in order]

    def validation_factory():
        return [
            _batch(feature_shift=0.15),
            _batch(feature_shift=0.35, atom_counts=(1,)),
        ]

    try:
        result = trainer.fit(
            train_factory,
            validation_factory,
            state=state,
            _stop_after_completed_epochs=stop_after,
            _stop_exception=ControlledEpochStop,
        )
    finally:
        logger.close()
    return {
        "state": result,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "events": _read_events(root / "events.jsonl"),
    }


def test_interrupted_epoch_boundary_resume_is_exact_with_dropout_and_adamw(
    tmp_path: Path,
):
    uninterrupted = _run_training(tmp_path / "a", stop_after=None, resume=False)
    with pytest.raises(ControlledEpochStop):
        _run_training(tmp_path / "b", stop_after=2, resume=False)
    committed = load_torch_artifact(tmp_path / "b" / "last.pt")
    assert committed["next_epoch"] == 2
    assert len(_read_events(tmp_path / "b" / "events.jsonl")) == 2
    resumed = _run_training(tmp_path / "b", stop_after=None, resume=True)

    assert _nested_equal(uninterrupted["model"], resumed["model"])
    assert _nested_equal(uninterrupted["optimizer"], resumed["optimizer"])
    assert uninterrupted["state"] == resumed["state"]
    assert uninterrupted["events"] == resumed["events"]
    assert resumed["state"].completed is True
    best = load_torch_artifact(tmp_path / "b" / "best.pt")
    totals = [event["validation"]["total_loss"] for event in resumed["events"]]
    assert best["epoch"] == totals.index(min(totals))
    assert best["validation_metrics"]["total_loss"] == min(totals)


def test_fit_stops_on_patience_and_marks_last_completed(tmp_path: Path, monkeypatch):
    config = _config(
        tmp_path, **{"trainer.max_epochs": 8, "trainer.early_stopping_patience": 2}
    )
    model = _model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _prime_adamw(model, optimizer)
    trainer = ConfidenceTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        binning=_artifact(config),
        device=torch.device("cpu"),
        logger=_logger(tmp_path / "events.jsonl"),
        identity=IDENTITY,
        best_path=tmp_path / "best.pt",
        last_path=tmp_path / "last.pt",
    )
    monkeypatch.setattr(
        trainer,
        "train_epoch",
        lambda batches: {
            "force_loss": 1.0,
            "energy_loss": 1.0,
            "total_loss": 1.3,
            "force_samples": 1,
            "energy_samples": 1,
        },
    )
    values = iter([1.0, 1.0, 1.1])
    monkeypatch.setattr(
        trainer,
        "validate_epoch",
        lambda batches: {
            "force_loss": 1.0,
            "energy_loss": 0.0,
            "total_loss": next(values),
            "force_samples": 1,
            "energy_samples": 1,
        },
    )
    result = trainer.fit(lambda epoch: [object()], lambda: [object()])
    trainer.logger.close()
    assert result == TrainingState(3, 0, 1.0, 2, True)
    assert load_torch_artifact(tmp_path / "last.pt")["completed"] is True
    assert [event["epoch"] for event in _read_events(tmp_path / "events.jsonl")] == [
        0,
        1,
        2,
    ]
