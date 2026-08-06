"""Orchestrate cache-only ConfidenceHead training runs."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..artifacts import atomic_json_dump, load_torch_artifact
from ..binning import BinningArtifact
from ..cache import (
    CacheManifest,
    iter_cache_batches,
    iter_shuffled_cache_batches,
    load_complete_cache,
)
from ..checkpoint import TrainingState, load_last
from ..config import ConfidenceHeadConfig
from ..identity import CodeIdentity, code_identity, experiment_id
from ..logging import TrainingLogger
from ..model import MultiBranchConfidenceModel
from ..run_naming import make_run_tag
from ..runtime import configure_runtime, environment_snapshot
from ..trainer import ConfidenceTrainer
from . import fit_bins as fit_bins_workflow


TRAINING_SCHEMA_VERSION = 1
TRAINING_FORMULA_VERSION = "confidence_head_training_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_ROOT_ENTRIES = frozenset({"binning", "config", "identity", "plots", "run"})
_RUN_ENTRIES = frozenset(
    {
        "events.jsonl",
        "best.pt",
        "last.pt",
        "training_summary.json",
        "training_validation.json",
        "training_manifest.json",
        "test_predictions.pt",
        "test_metrics.json",
        "evaluation_manifest.json",
        "wandb",
    }
)


class RunConflictError(RuntimeError):
    """An existing run cannot be safely reused or overwritten."""


class ControlledEpochStop(RuntimeError):
    """Test-only stop raised after a fully committed epoch boundary."""


@dataclass(frozen=True)
class RunInputs:
    config: ConfidenceHeadConfig
    code: CodeIdentity
    cache: CacheManifest
    binning: BinningArtifact
    run_root: Path
    run_dir: Path
    identity: dict[str, str]
    overflow: dict[str, int]


_ALLOWED_EVENT_KEYS = {
    "epoch",
    "train",
    "validation",
    "improved",
    "best_epoch",
    "best_validation_loss",
    "bad_epochs",
    "completed",
    "overflow",
}
_METRIC_KEYS = {
    "force_loss",
    "energy_loss",
    "total_loss",
    "force_samples",
    "energy_samples",
}


def _finite_event_value(value: object, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RunConflictError(f"{where} must be finite")
    return float(value)


def _nonnegative_event_int(value: object, where: str) -> int:
    if type(value) is not int or value < 0:
        raise RunConflictError(f"{where} must be a non-negative integer")
    return value


def _validate_event_overflow(
    value: object,
    *,
    where: str,
    inputs: RunInputs,
) -> dict[str, int]:
    expected_keys = set(inputs.binning.branches)
    if type(value) is not dict or set(value) != expected_keys:
        raise RunConflictError(f"{where} schema differs from enabled branches")
    result = {
        branch: _nonnegative_event_int(value[branch], f"{where}.{branch}")
        for branch in expected_keys
    }
    if result != inputs.overflow:
        raise RunConflictError(f"{where} diagnostics differ")
    return result


def _expected_event_samples(inputs: RunInputs, split: str, branch: str) -> int:
    shards = inputs.cache.splits.get(split, ())
    if branch == "energy":
        return sum(shard.num_structures for shard in shards)
    atoms = sum(shard.num_atoms for shard in shards)
    multiplier = 3 if inputs.config.model.force.target_mode == "component" else 1
    return atoms * multiplier


def _validate_event_metrics(
    value: object,
    *,
    where: str,
    split: str,
    inputs: RunInputs,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _METRIC_KEYS:
        raise RunConflictError(f"{where} metric schema differs")
    result = dict(value)
    for name in ("force_loss", "energy_loss", "total_loss"):
        _finite_event_value(result[name], f"{where}.{name}")
    for branch in ("force", "energy"):
        count = result[f"{branch}_samples"]
        if type(count) is not int or count < 0:
            raise RunConflictError(f"{where}.{branch}_samples differs")
        enabled = branch in inputs.binning.branches
        if enabled and count != _expected_event_samples(inputs, split, branch):
            raise RunConflictError(
                f"{where}.{branch}_samples sample count differs from complete cache"
            )
        if not enabled and (count != 0 or float(result[f"{branch}_loss"]) != 0.0):
            raise RunConflictError(f"{where} contains disabled {branch} metrics")
    expected_total = inputs.config.loss.force_coefficient * float(
        result["force_loss"]
    ) + inputs.config.loss.energy_coefficient * float(result["energy_loss"])
    if not math.isclose(
        float(result["total_loss"]), expected_total, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RunConflictError(f"{where}.total_loss formula differs")
    return result


def _validate_event_history(
    inputs: RunInputs,
    *,
    require_completed: bool,
) -> tuple[list[dict[str, Any]], TrainingState]:
    path = inputs.run_dir / "events.jsonl"
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise RunConflictError("events.jsonl is empty or has an incomplete tail")
    try:
        events = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunConflictError(f"events.jsonl is corrupt: {error}") from error

    best_epoch: int | None = None
    best_loss = math.inf
    bad_epochs = 0
    terminal = False
    for epoch, event in enumerate(events):
        if type(event) is not dict or set(event) != _ALLOWED_EVENT_KEYS:
            raise RunConflictError(f"event {epoch} schema differs")
        event_epoch = _nonnegative_event_int(event["epoch"], f"event {epoch}.epoch")
        if event_epoch != epoch:
            raise RunConflictError("event epochs must be contiguous from zero")
        if terminal:
            raise RunConflictError("events continue after a completed epoch")
        _validate_event_metrics(
            event["train"],
            where=f"event {epoch}.train",
            split="train",
            inputs=inputs,
        )
        validation = _validate_event_metrics(
            event["validation"],
            where=f"event {epoch}.validation",
            split="validation",
            inputs=inputs,
        )
        total = float(validation["total_loss"])
        improved = total < best_loss
        if improved:
            best_epoch = epoch
            best_loss = total
            bad_epochs = 0
        else:
            bad_epochs += 1
        completed = (
            bad_epochs >= inputs.config.trainer.early_stopping_patience
            or epoch + 1 >= inputs.config.trainer.max_epochs
        )
        event_best_epoch = _nonnegative_event_int(
            event["best_epoch"], f"event {epoch}.best_epoch"
        )
        event_best_loss = _finite_event_value(
            event["best_validation_loss"],
            f"event {epoch}.best_validation_loss",
        )
        event_bad_epochs = _nonnegative_event_int(
            event["bad_epochs"], f"event {epoch}.bad_epochs"
        )
        if (
            event["improved"] is not improved
            or event_best_epoch != best_epoch
            or event_best_loss != best_loss
            or event_bad_epochs != bad_epochs
            or event["completed"] is not completed
        ):
            raise RunConflictError(f"event {epoch} best/early-stop semantics differ")
        _validate_event_overflow(
            event["overflow"], where=f"event {epoch}.overflow", inputs=inputs
        )
        terminal = completed

    if not events or best_epoch is None:
        raise RunConflictError("event history contains no committed epoch")
    if terminal is not require_completed:
        expected = "completed" if require_completed else "resumable"
        raise RunConflictError(f"event history is not a {expected} training run")
    state = TrainingState(
        len(events),
        best_epoch,
        best_loss,
        len(events) - best_epoch - 1,
        terminal,
    )
    return events, state


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RunConflictError(f"{where} is unreadable: {error}") from error
    if type(value) is not dict:
        raise RunConflictError(f"{where} must contain a JSON object")
    return value


def _exact_directory(
    path: Path, expected: set[str] | frozenset[str], *, where: str
) -> None:
    if not path.is_dir():
        raise RunConflictError(f"{where} directory is missing")
    actual = {item.name for item in path.iterdir()}
    if actual != set(expected):
        raise RunConflictError(
            f"{where} directory entries differ: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _enabled(config: ConfidenceHeadConfig) -> list[str]:
    return [
        name
        for name, enabled in (
            ("energy", config.energy_enabled),
            ("force", config.force_enabled),
        )
        if enabled
    ]


def _train_counts(cache: CacheManifest) -> tuple[int, int]:
    shards = cache.splits.get("train", ())
    return (
        sum(shard.num_structures for shard in shards),
        sum(shard.num_atoms for shard in shards),
    )


def _load_run_inputs(config: ConfidenceHeadConfig) -> RunInputs:
    if not isinstance(config, ConfidenceHeadConfig):
        raise TypeError("config must be a ConfidenceHeadConfig")
    code = code_identity(REPOSITORY_ROOT)
    cache_identity = fit_bins_workflow._cache_identity(config)
    cache = load_complete_cache(
        fit_bins_workflow._cache_root(config, cache_identity),
        expected_cache_id=cache_identity,
        expected_allow_cross_split_duplicates=config.profile == "smoke_test",
    )
    experiment_identity = experiment_id(config, cache_identity, code=code)
    run_root = fit_bins_workflow._binning_root(config, experiment_identity).parent
    binning_root = run_root / "binning"
    _exact_directory(
        binning_root,
        {"binning.pt", "binning_manifest.json"},
        where="binning",
    )
    target_mode = config.model.force.target_mode if config.force_enabled else None
    try:
        binning = BinningArtifact.from_payload(
            load_torch_artifact(binning_root / "binning.pt"),
            force_target_mode=target_mode,
        )
    except Exception as error:
        raise RunConflictError(f"binning artifact is invalid: {error}") from error
    expected_branches = set(_enabled(config))
    expected_identity = {
        "run_id": binning.run_id,
        "experiment_id": experiment_identity,
        "cache_id": cache_identity,
        "binning_id": binning.binning_id,
    }
    if (
        binning.cache_id != cache_identity
        or binning.experiment_id != experiment_identity
        or binning.algorithm != config.binning.algorithm
        or set(binning.branches) != expected_branches
    ):
        raise RunConflictError("cache/binning/config identity mismatch")
    structures, atoms = _train_counts(cache)
    manifest = _read_json(
        binning_root / "binning_manifest.json", where="binning manifest"
    )
    expected_manifest = fit_bins_workflow._manifest_payload(
        binning,
        artifact_sha256=fit_bins_workflow.sha256_file(binning_root / "binning.pt"),
        train_structure_count=structures,
        train_atom_count=atoms,
    )
    if manifest != expected_manifest:
        raise RunConflictError("binning manifest identity or hash differs")
    if run_root.exists():
        unknown = {item.name for item in run_root.iterdir()} - _ROOT_ENTRIES
        if unknown:
            raise RunConflictError(
                f"run root contains unknown entries: {sorted(unknown)}"
            )
    overflow = {
        name: branch.overflow_count for name, branch in sorted(binning.branches.items())
    }
    return RunInputs(
        config=config,
        code=code,
        cache=cache,
        binning=binning,
        run_root=run_root,
        run_dir=run_root / "run",
        identity=expected_identity,
        overflow=overflow,
    )


def _snapshot_payloads(inputs: RunInputs) -> dict[Path, Any]:
    config = inputs.config
    return {
        inputs.run_root / "config" / "resolved.json": _json_safe(asdict(config)),
        inputs.run_root / "identity" / "run_identity.json": {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "formula_version": TRAINING_FORMULA_VERSION,
            **inputs.identity,
            "run_tag": make_run_tag(config),
            "enabled_branches": _enabled(config),
        },
        inputs.run_root / "identity" / "environment.json": environment_snapshot(
            REPOSITORY_ROOT
        ),
        inputs.run_root / "identity" / "git.json": asdict(inputs.code),
    }


def _commit_or_validate_snapshots(inputs: RunInputs) -> None:
    config_dir = inputs.run_root / "config"
    identity_dir = inputs.run_root / "identity"
    source_path = config_dir / "source.yaml"
    expected_source = inputs.config.source_path.read_bytes()
    payloads = _snapshot_payloads(inputs)
    config_exists = config_dir.exists()
    identity_exists = identity_dir.exists()
    if config_exists != identity_exists:
        raise RunConflictError("immutable snapshot directories are partial")
    if config_exists:
        _exact_directory(
            config_dir, {"source.yaml", "resolved.json"}, where="config snapshot"
        )
        _exact_directory(
            identity_dir,
            {"run_identity.json", "environment.json", "git.json"},
            where="identity snapshot",
        )
        if source_path.read_bytes() != expected_source:
            raise RunConflictError("source config snapshot differs")
        for path, expected in payloads.items():
            if _read_json(path, where=path.name) != expected:
                raise RunConflictError(f"immutable snapshot {path.name} differs")
        return

    config_dir.mkdir(parents=True, exist_ok=False)
    identity_dir.mkdir(parents=True, exist_ok=False)
    _atomic_bytes(source_path, expected_source)
    for path, payload in payloads.items():
        atomic_json_dump(path, payload)
    _commit_or_validate_snapshots(inputs)


def _feature_dtype(cache: CacheManifest, batch_size: int) -> torch.dtype:
    try:
        batch = next(iter(iter_cache_batches(cache, "train", batch_size=batch_size)))
    except StopIteration as error:
        raise RunConflictError("train cache is empty") from error
    dtype = batch.scalar_features.dtype
    if dtype not in {torch.float32, torch.float64}:
        raise RunConflictError("cache feature dtype is unsupported")
    return dtype


class _OverflowLogger(TrainingLogger):
    def __init__(self, base: TrainingLogger, overflow: Mapping[str, int]) -> None:
        super().__init__(base.local, base.mirror)
        self._overflow = dict(overflow)

    def append_epoch(self, event: Mapping[str, Any]) -> None:
        enriched = dict(event)
        enriched["overflow"] = dict(self._overflow)
        super().append_epoch(enriched)


def _new_model_optimizer(
    inputs: RunInputs,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MultiBranchConfidenceModel, torch.optim.AdamW]:
    model = MultiBranchConfidenceModel.from_config(inputs.config).to(
        device=device, dtype=dtype
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=inputs.config.optimizer.learning_rate,
        weight_decay=inputs.config.optimizer.weight_decay,
    )
    return model, optimizer


def _summary(
    inputs: RunInputs,
    state: TrainingState,
    *,
    wandb_mode: str,
    wandb_failed: bool,
) -> dict[str, Any]:
    stop_reason = (
        "max_epochs"
        if state.next_epoch >= inputs.config.trainer.max_epochs
        else "early_stopping"
    )
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "formula_version": TRAINING_FORMULA_VERSION,
        **inputs.identity,
        "enabled_branches": _enabled(inputs.config),
        "stop_reason": stop_reason,
        "epoch_count": state.next_epoch,
        "event_count": state.next_epoch,
        "last_epoch": state.next_epoch - 1,
        "best_epoch": state.best_epoch,
        "best_validation_loss": state.best_validation_loss,
        "overflow": dict(inputs.overflow),
        "wandb_mode": wandb_mode,
        "wandb_failed": wandb_failed,
    }


def run_train(
    config: ConfidenceHeadConfig,
    *,
    _stop_after_completed_epochs: int | None = None,
) -> Path:
    """Train an immutable experiment from committed cache and binning inputs."""
    inputs = _load_run_inputs(config)
    inputs.run_root.mkdir(parents=True, exist_ok=True)
    _commit_or_validate_snapshots(inputs)

    run_preexisting = inputs.run_dir.exists()
    if run_preexisting:
        unknown = {item.name for item in inputs.run_dir.iterdir()} - _RUN_ENTRIES
        if unknown:
            raise RunConflictError(
                f"run directory contains unknown entries: {sorted(unknown)}"
            )
    else:
        inputs.run_dir.mkdir(parents=True, exist_ok=False)

    summary_path = inputs.run_dir / "training_summary.json"
    validation_path = inputs.run_dir / "training_validation.json"
    manifest_path = inputs.run_dir / "training_manifest.json"
    if manifest_path.exists():
        raise RunConflictError("run is already complete")
    if validation_path.exists():
        raise RunConflictError("run contains partial finalization evidence")
    if summary_path.exists():
        raise RunConflictError("training is already complete; run check_training.py")

    device = configure_runtime(
        config.runtime.seed, config.runtime.deterministic, config.runtime.device
    )
    dtype = _feature_dtype(inputs.cache, config.trainer.batch_size)
    model, optimizer = _new_model_optimizer(inputs, device=device, dtype=dtype)

    events_path = inputs.run_dir / "events.jsonl"
    best_path = inputs.run_dir / "best.pt"
    last_path = inputs.run_dir / "last.pt"
    state: TrainingState | None = None
    evidence = {
        "events": events_path.exists(),
        "best": best_path.exists(),
        "last": last_path.exists(),
    }
    if run_preexisting:
        if evidence["last"]:
            if not config.trainer.resume:
                raise RunConflictError("resume is disabled for the existing run")
            if not evidence["events"] or not evidence["best"]:
                raise RunConflictError(
                    "partial run is missing events or best checkpoint"
                )
            try:
                committed_events, replayed_state = _validate_event_history(
                    inputs, require_completed=False
                )
                last_payload = load_torch_artifact(last_path)
                state = load_last(last_path, model, optimizer, inputs.identity)
            except Exception as error:
                raise RunConflictError(
                    f"last checkpoint cannot resume: {error}"
                ) from error
            if len(committed_events) != state.next_epoch:
                raise RunConflictError(
                    "resume event count differs from last checkpoint"
                )
            if replayed_state != state:
                raise RunConflictError(
                    "resume event state differs from last checkpoint"
                )
            if committed_events[-1]["validation"] != last_payload["validation_metrics"]:
                raise RunConflictError(
                    "resume final event differs from last checkpoint"
                )
            if state.completed:
                raise RunConflictError(
                    "training is already complete; run check_training.py"
                )
        elif any(evidence.values()):
            raise RunConflictError("partial run has no valid last checkpoint")
        else:
            raise RunConflictError("existing empty run has no resumable checkpoint")

    base_logger = TrainingLogger.start(
        events_path,
        config.logging,
        {**inputs.identity, "name": make_run_tag(config)},
    )
    logger = _OverflowLogger(base_logger, inputs.overflow)
    trainer = ConfidenceTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        binning=inputs.binning,
        device=device,
        logger=logger,
        identity=inputs.identity,
        best_path=best_path,
        last_path=last_path,
    )
    try:
        final_state = trainer.fit(
            lambda epoch: iter_shuffled_cache_batches(
                inputs.cache,
                "train",
                config.trainer.batch_size,
                seed=config.runtime.seed + epoch,
            ),
            lambda: iter_cache_batches(
                inputs.cache, "validation", config.trainer.batch_size
            ),
            state=state,
            _stop_after_completed_epochs=_stop_after_completed_epochs,
            _stop_exception=ControlledEpochStop,
        )
        logger.update_summary({"overflow": dict(inputs.overflow)})
    finally:
        logger.close()
    effective_wandb_mode = logger.mirror.mode
    wandb_failed = logger.mirror.failed
    if not final_state.completed:
        raise RuntimeError("trainer returned an incomplete terminal state")
    atomic_json_dump(
        summary_path,
        _summary(
            inputs,
            final_state,
            wandb_mode=effective_wandb_mode,
            wandb_failed=wandb_failed,
        ),
    )
    return best_path
