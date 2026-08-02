"""Validate completed ConfidenceHead training artifacts without inference."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

from ..artifacts import atomic_json_dump, load_torch_artifact
from ..checkpoint import (
    BEST_CHECKPOINT_KEYS,
    CHECKPOINT_FORMULA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    LAST_CHECKPOINT_KEYS,
    TrainingState,
    load_last,
    parameter_schema,
)
from ..config import ConfidenceHeadConfig
from ..identity import sha256_file
from ..runtime import capture_rng_state, restore_rng_state
from .train import (
    REPOSITORY_ROOT,
    TRAINING_FORMULA_VERSION,
    TRAINING_SCHEMA_VERSION,
    RunConflictError,
    RunInputs,
    _commit_or_validate_snapshots,
    _enabled,
    _feature_dtype,
    _load_run_inputs,
    _new_model_optimizer,
    _summary,
)


_VALIDATION_FILENAME = "training_validation.json"
_MANIFEST_FILENAME = "training_manifest.json"
_REQUIRED_RUN_FILES = {
    "events.jsonl",
    "best.pt",
    "last.pt",
    "training_summary.json",
}
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


def _finite(value: object, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RunConflictError(f"{where} must be finite")
    return float(value)


def _expected_samples(inputs: RunInputs, split: str, branch: str) -> int:
    shards = inputs.cache.splits.get(split, ())
    if branch == "energy":
        return sum(shard.num_structures for shard in shards)
    atoms = sum(shard.num_atoms for shard in shards)
    multiplier = 3 if inputs.config.model.force.target_mode == "component" else 1
    return atoms * multiplier


def _metrics(
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
        _finite(result[name], f"{where}.{name}")
    for branch in ("force", "energy"):
        count = result[f"{branch}_samples"]
        if type(count) is not int or count < 0:
            raise RunConflictError(f"{where}.{branch}_samples differs")
        enabled = branch in inputs.binning.branches
        if enabled and count < 1:
            raise RunConflictError(f"{where}.{branch}_samples must be positive")
        if enabled and count != _expected_samples(inputs, split, branch):
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


def _events(inputs: RunInputs) -> tuple[list[dict[str, Any]], TrainingState]:
    path = inputs.run_dir / "events.jsonl"
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise RunConflictError("events.jsonl is empty or has an incomplete tail")
    try:
        lines = raw.decode("utf-8").splitlines()
        events = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunConflictError(f"events.jsonl is corrupt: {error}") from error
    best_epoch: int | None = None
    best_loss = math.inf
    bad_epochs = 0
    terminal = False
    for epoch, event in enumerate(events):
        if type(event) is not dict or set(event) != _ALLOWED_EVENT_KEYS:
            raise RunConflictError(f"event {epoch} schema differs")
        if event["epoch"] != epoch:
            raise RunConflictError("event epochs must be contiguous from zero")
        if terminal:
            raise RunConflictError("events continue after a completed epoch")
        _metrics(
            event["train"],
            where=f"event {epoch}.train",
            split="train",
            inputs=inputs,
        )
        validation = _metrics(
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
        if (
            event["improved"] is not improved
            or event["best_epoch"] != best_epoch
            or float(event["best_validation_loss"]) != best_loss
            or event["bad_epochs"] != bad_epochs
            or event["completed"] is not completed
        ):
            raise RunConflictError(f"event {epoch} best/early-stop semantics differ")
        if event["overflow"] != inputs.overflow:
            raise RunConflictError(f"event {epoch} overflow diagnostics differ")
        terminal = completed
    if not events or not terminal or best_epoch is None:
        raise RunConflictError("event history is not a completed training run")
    state = TrainingState(
        len(events), best_epoch, best_loss, len(events) - best_epoch - 1, True
    )
    return events, state


def _validate_tensor_state(
    payload: Mapping[str, Any],
    expected_schema: dict[str, dict[str, Any]],
    *,
    where: str,
) -> None:
    if payload["parameter_schema"] != expected_schema:
        raise RunConflictError(f"{where} parameter schema differs")
    state = payload["model_state"]
    if type(state) is not dict or set(state) != set(expected_schema):
        raise RunConflictError(f"{where} model state names differ")
    for name, tensor in state.items():
        descriptor = expected_schema[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or list(tensor.shape) != descriptor["shape"]
            or str(tensor.dtype) != descriptor["dtype"]
        ):
            raise RunConflictError(f"{where} model state {name} metadata differs")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise RunConflictError(f"{where} model state contains non-finite tensors")


def _validate_checkpoint_base(
    payload: dict[str, Any],
    *,
    keys: frozenset[str],
    inputs: RunInputs,
    expected_schema: dict[str, dict[str, Any]],
    where: str,
) -> None:
    if set(payload) != set(keys):
        raise RunConflictError(f"{where} schema keys differ")
    if (
        payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or payload["formula_version"] != CHECKPOINT_FORMULA_VERSION
    ):
        raise RunConflictError(f"{where} version differs")
    if any(payload[name] != value for name, value in inputs.identity.items()):
        raise RunConflictError(f"{where} identity differs")
    created_at = payload["created_at"]
    try:
        parsed = datetime.fromisoformat(created_at)
    except (TypeError, ValueError) as error:
        raise RunConflictError(f"{where} timestamp differs") from error
    if parsed.tzinfo is None:
        raise RunConflictError(f"{where} timestamp has no timezone")
    _validate_tensor_state(payload, expected_schema, where=where)


def _validate_checkpoints(
    inputs: RunInputs,
    events: list[dict[str, Any]],
    state: TrainingState,
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
) -> bool:
    expected_schema = parameter_schema(model)
    best = load_torch_artifact(inputs.run_dir / "best.pt")
    _validate_checkpoint_base(
        best,
        keys=BEST_CHECKPOINT_KEYS,
        inputs=inputs,
        expected_schema=expected_schema,
        where="best.pt",
    )
    if best["epoch"] != state.best_epoch:
        raise RunConflictError(
            "best.pt does not correspond to the strict minimum event"
        )
    expected_best_metrics = events[state.best_epoch]["validation"]
    if best["validation_metrics"] != expected_best_metrics:
        raise RunConflictError("best.pt validation metrics differ from the best event")

    last_payload = load_torch_artifact(inputs.run_dir / "last.pt")
    _validate_checkpoint_base(
        last_payload,
        keys=LAST_CHECKPOINT_KEYS,
        inputs=inputs,
        expected_schema=expected_schema,
        where="last.pt",
    )
    try:
        restored = load_last(
            inputs.run_dir / "last.pt", model, optimizer, inputs.identity
        )
    except Exception as error:
        raise RunConflictError(f"last.pt is not fully restorable: {error}") from error
    if restored != state:
        raise RunConflictError("last.pt training state differs from events")
    if (
        last_payload["epoch"] != len(events) - 1
        or last_payload["validation_metrics"] != events[-1]["validation"]
        or last_payload["completed"] is not True
    ):
        raise RunConflictError("last.pt does not correspond to the final event")
    return inputs.config.energy_enabled


def _validate_summary(inputs: RunInputs, state: TrainingState) -> dict[str, Any]:
    path = inputs.run_dir / "training_summary.json"
    summary = _read_json(path)
    mode = summary.get("wandb_mode")
    allowed = (
        {"online", "offline"}
        if inputs.config.logging.wandb_mode == "auto"
        else {
            "disabled"
            if not inputs.config.logging.wandb
            else inputs.config.logging.wandb_mode
        }
    )
    if mode not in allowed:
        raise RunConflictError("training summary W&B mode differs")
    expected = _summary(inputs, state, wandb_mode=mode)
    if summary != expected:
        raise RunConflictError("training summary differs from events/checkpoints")
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RunConflictError(f"{path.name} is unreadable: {error}") from error
    if type(value) is not dict:
        raise RunConflictError(f"{path.name} must contain a JSON object")
    return value


def _support_files(inputs: RunInputs) -> dict[str, Path]:
    return {
        "source.yaml": inputs.run_root / "config" / "source.yaml",
        "resolved.json": inputs.run_root / "config" / "resolved.json",
        "run_identity.json": inputs.run_root / "identity" / "run_identity.json",
        "environment.json": inputs.run_root / "identity" / "environment.json",
        "git.json": inputs.run_root / "identity" / "git.json",
        "binning.pt": inputs.run_root / "binning" / "binning.pt",
        "binning_manifest.json": inputs.run_root / "binning" / "binning_manifest.json",
        "events.jsonl": inputs.run_dir / "events.jsonl",
        "best.pt": inputs.run_dir / "best.pt",
        "last.pt": inputs.run_dir / "last.pt",
        "training_summary.json": inputs.run_dir / "training_summary.json",
    }


def _file_hashes(inputs: RunInputs) -> dict[str, str]:
    paths = _support_files(inputs)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RunConflictError(f"training support artifacts are missing: {missing}")
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _validation_payload(
    inputs: RunInputs,
    *,
    events: list[dict[str, Any]],
    state: TrainingState,
    dtype: torch.dtype,
    energy_projection_updated: bool,
    file_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "formula_version": TRAINING_FORMULA_VERSION,
        **inputs.identity,
        "valid": True,
        "enabled_branches": _enabled(inputs.config),
        "event_count": len(events),
        "last_epoch": len(events) - 1,
        "best_epoch": state.best_epoch,
        "best_validation_loss": state.best_validation_loss,
        "overflow": dict(inputs.overflow),
        "feature_schema": {"total_dim": 640, "dtype": str(dtype)},
        "energy_projection_updated": energy_projection_updated,
        "file_sha256": file_hashes,
    }


def _manifest_payload(
    inputs: RunInputs,
    validation_path: Path,
    validation: dict[str, Any],
) -> dict[str, Any]:
    hashes = dict(validation["file_sha256"])
    hashes[_VALIDATION_FILENAME] = sha256_file(validation_path)
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "formula_version": TRAINING_FORMULA_VERSION,
        **inputs.identity,
        "validation_filename": _VALIDATION_FILENAME,
        "validation_sha256": hashes[_VALIDATION_FILENAME],
        "file_sha256": hashes,
    }


def run_check_training(config: ConfidenceHeadConfig) -> Path:
    """Validate a completed run and atomically commit its completion marker."""
    caller_rng = capture_rng_state()
    try:
        inputs = _load_run_inputs(config)
        if (
            not (inputs.run_root / "config").is_dir()
            or not (inputs.run_root / "identity").is_dir()
        ):
            raise RunConflictError("immutable training snapshots are missing")
        _commit_or_validate_snapshots(inputs)
        if not inputs.run_dir.is_dir():
            raise RunConflictError("training run directory is missing")
        unknown = {item.name for item in inputs.run_dir.iterdir()} - (
            _REQUIRED_RUN_FILES | {_VALIDATION_FILENAME, _MANIFEST_FILENAME, "wandb"}
        )
        if unknown:
            raise RunConflictError(
                f"run directory contains unknown entries: {sorted(unknown)}"
            )
        missing = {
            name
            for name in _REQUIRED_RUN_FILES
            if not (inputs.run_dir / name).is_file()
        }
        if missing:
            raise RunConflictError(
                f"training run is incomplete; missing {sorted(missing)}"
            )
        validation_path = inputs.run_dir / _VALIDATION_FILENAME
        manifest_path = inputs.run_dir / _MANIFEST_FILENAME
        if validation_path.exists() != manifest_path.exists():
            raise RunConflictError("training finalization is a partial artifact pair")

        events, state = _events(inputs)
        dtype = _feature_dtype(inputs.cache, config.trainer.batch_size)
        model, optimizer = _new_model_optimizer(
            inputs, device=torch.device("cpu"), dtype=dtype
        )
        energy_projection_updated = _validate_checkpoints(
            inputs, events, state, model, optimizer
        )
        _validate_summary(inputs, state)
        validation = _validation_payload(
            inputs,
            events=events,
            state=state,
            dtype=dtype,
            energy_projection_updated=energy_projection_updated,
            file_hashes=_file_hashes(inputs),
        )
        if validation_path.exists():
            if _read_json(validation_path) != validation:
                raise RunConflictError("existing training validation differs")
            expected_manifest = _manifest_payload(inputs, validation_path, validation)
            if _read_json(manifest_path) != expected_manifest:
                raise RunConflictError("existing training manifest differs")
            return validation_path

        atomic_json_dump(validation_path, validation)
        reloaded = _read_json(validation_path)
        if reloaded != validation:
            raise RunConflictError("new training validation failed verification")
        atomic_json_dump(
            manifest_path, _manifest_payload(inputs, validation_path, validation)
        )
        return validation_path
    finally:
        restore_rng_state(caller_rng)
