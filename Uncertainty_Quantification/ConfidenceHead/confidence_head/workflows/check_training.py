"""Validate completed ConfidenceHead training artifacts without inference."""

from __future__ import annotations

import json
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
    _validate_event_history,
)


_VALIDATION_FILENAME = "training_validation.json"
_MANIFEST_FILENAME = "training_manifest.json"
_REQUIRED_RUN_FILES = {
    "events.jsonl",
    "best.pt",
    "last.pt",
    "training_summary.json",
}


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


def _initial_model_optimizer(
    inputs: RunInputs,
    *,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, torch.optim.AdamW, dict[str, torch.Tensor]]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(inputs.config.runtime.seed)
        model, optimizer = _new_model_optimizer(
            inputs, device=torch.device("cpu"), dtype=dtype
        )
    initial_projection = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("energy_adapter.projection.")
    }
    return model, optimizer, initial_projection


def _validate_energy_projection(
    *,
    enabled: bool,
    best_model_state: Mapping[str, torch.Tensor],
    initial_projection: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
) -> bool:
    if not enabled:
        return False
    expected_names = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("energy_adapter.projection.")
    }
    if not expected_names or set(initial_projection) != expected_names:
        raise RunConflictError("energy projection parameter coverage differs")
    for name in sorted(expected_names):
        trained = best_model_state[name]
        initial = initial_projection[name]
        if not bool(torch.isfinite(trained).all().item()) or torch.equal(
            trained, initial
        ):
            raise RunConflictError(
                f"energy projection parameter {name} was not updated"
            )

    parameter_by_name = dict(model.named_parameters())
    for name in sorted(expected_names):
        state = optimizer.state.get(parameter_by_name[name])
        if type(state) is not dict:
            raise RunConflictError(
                f"energy projection optimizer state {name} is missing"
            )
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state.get(moment_name)
            if (
                not isinstance(moment, torch.Tensor)
                or not bool(torch.isfinite(moment).all().item())
                or not bool(torch.count_nonzero(moment).item())
            ):
                raise RunConflictError(
                    f"energy projection optimizer {moment_name} for {name} is invalid"
                )
    return True


def _validate_checkpoints(
    inputs: RunInputs,
    events: list[dict[str, Any]],
    state: TrainingState,
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    initial_projection: Mapping[str, torch.Tensor],
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
    return _validate_energy_projection(
        enabled=inputs.config.energy_enabled,
        best_model_state=best["model_state"],
        initial_projection=initial_projection,
        model=model,
        optimizer=optimizer,
    )


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
    failed = summary.get("wandb_failed")
    if type(failed) is not bool:
        raise RunConflictError("training summary W&B failure state differs")
    expected = _summary(inputs, state, wandb_mode=mode, wandb_failed=failed)
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

        events, state = _validate_event_history(inputs, require_completed=True)
        dtype = _feature_dtype(inputs.cache, config.trainer.batch_size)
        model, optimizer, initial_projection = _initial_model_optimizer(
            inputs, dtype=dtype
        )
        energy_projection_updated = _validate_checkpoints(
            inputs, events, state, model, optimizer, initial_projection
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
