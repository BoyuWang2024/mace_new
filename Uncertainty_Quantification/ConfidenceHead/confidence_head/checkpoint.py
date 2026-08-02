"""Strict portable ConfidenceHead checkpoints and epoch training state."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .artifacts import atomic_torch_save, load_torch_artifact
from .runtime import capture_rng_state, restore_rng_state


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_FORMULA_VERSION = "confidence_head_checkpoint_v1"
IDENTITY_KEYS = frozenset({"run_id", "experiment_id", "cache_id", "binning_id"})
BEST_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "formula_version",
        *IDENTITY_KEYS,
        "epoch",
        "validation_metrics",
        "model_state",
        "parameter_schema",
        "created_at",
    }
)
LAST_CHECKPOINT_KEYS = frozenset(
    {
        *BEST_CHECKPOINT_KEYS,
        "next_epoch",
        "optimizer_state",
        "early_stopping_state",
        "best_epoch",
        "best_validation_loss",
        "rng_state",
        "completed",
    }
)
_EARLY_STOPPING_KEYS = frozenset({"best_epoch", "best_validation_loss", "bad_epochs"})


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class EarlyStoppingState:
    """Immutable strict-improvement early-stopping state."""

    best_epoch: int | None
    best_validation_loss: float
    bad_epochs: int

    def __post_init__(self) -> None:
        if self.best_epoch is not None:
            _exact_int(self.best_epoch, "best_epoch")
            _finite_float(self.best_validation_loss, "best_validation_loss")
        elif self.best_validation_loss != math.inf:
            raise ValueError("absent best_epoch requires infinite initial best loss")
        _exact_int(self.bad_epochs, "bad_epochs")

    @classmethod
    def initial(cls) -> EarlyStoppingState:
        return cls(None, math.inf, 0)

    def observe(
        self, epoch: int, validation_total: float
    ) -> tuple[EarlyStoppingState, bool]:
        observed_epoch = _exact_int(epoch, "epoch")
        loss = _finite_float(validation_total, "validation total")
        if self.best_epoch is not None and observed_epoch <= self.best_epoch:
            # Epochs after the best are valid; an earlier/repeated observation is not.
            if observed_epoch < self.best_epoch:
                raise ValueError("epoch precedes the current best epoch")
        if loss < self.best_validation_loss:
            return EarlyStoppingState(observed_epoch, loss, 0), True
        return EarlyStoppingState(
            self.best_epoch, self.best_validation_loss, self.bad_epochs + 1
        ), False

    def should_stop(self, patience: int) -> bool:
        value = _exact_int(patience, "patience", minimum=1)
        return self.bad_epochs >= value

    def to_payload(self) -> dict[str, int | float]:
        if self.best_epoch is None or not math.isfinite(self.best_validation_loss):
            raise ValueError("committed early-stopping state requires a finite best")
        return {
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "bad_epochs": self.bad_epochs,
        }

    @classmethod
    def from_payload(cls, value: object) -> EarlyStoppingState:
        mapping = _exact_mapping(value, _EARLY_STOPPING_KEYS, "early-stopping state")
        return cls(
            _exact_int(mapping["best_epoch"], "early-stopping best_epoch"),
            _finite_float(
                mapping["best_validation_loss"], "early-stopping best_validation_loss"
            ),
            _exact_int(mapping["bad_epochs"], "early-stopping bad_epochs"),
        )


@dataclass(frozen=True)
class TrainingState:
    """State committed at an epoch boundary; ``next_epoch`` is uncommitted."""

    next_epoch: int
    best_epoch: int | None
    best_validation_loss: float
    bad_epochs: int
    completed: bool

    def __post_init__(self) -> None:
        _exact_int(self.next_epoch, "next_epoch")
        if self.best_epoch is None:
            if self.next_epoch != 0 or self.best_validation_loss != math.inf:
                raise ValueError("only initial training state may have no finite best")
        else:
            _exact_int(self.best_epoch, "best_epoch")
            _finite_float(self.best_validation_loss, "best_validation_loss")
            if self.best_epoch >= self.next_epoch:
                raise ValueError("best_epoch must precede next_epoch")
        _exact_int(self.bad_epochs, "bad_epochs")
        if type(self.completed) is not bool:
            raise ValueError("completed must be a boolean")

    @classmethod
    def initial(cls) -> TrainingState:
        return cls(0, None, math.inf, 0, False)

    @property
    def early_stopping(self) -> EarlyStoppingState:
        return EarlyStoppingState(
            self.best_epoch, self.best_validation_loss, self.bad_epochs
        )


def _exact_mapping(value: object, keys: frozenset[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} schema keys mismatch")
    return value


def _identity(value: object) -> dict[str, str]:
    mapping = _exact_mapping(value, IDENTITY_KEYS, "checkpoint identity")
    if any(not isinstance(item, str) or not item for item in mapping.values()):
        raise ValueError("checkpoint identity values must be non-empty strings")
    return {name: mapping[name] for name in sorted(IDENTITY_KEYS)}


def _metrics(value: object) -> dict[str, int | float]:
    if type(value) is not dict or "total_loss" not in value or not value:
        raise ValueError("validation metrics must be a plain mapping with total_loss")
    result: dict[str, int | float] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("validation metric names must be non-empty strings")
        finite = _finite_float(item, f"validation metric {name}")
        result[name] = item if type(item) is int else finite
    return result


def _portable(value: Any, name: str) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").clone()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"{name} contains non-finite tensors")
        return tensor
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, dict):
        return {
            _portable(key, name): _portable(item, name) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable(item, name) for item in value]
    if isinstance(value, tuple):
        return tuple(_portable(item, name) for item in value)
    raise TypeError(f"{name} contains unsupported value {type(value).__name__}")


def parameter_schema(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Describe every model-state entry with deterministic basic values."""
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(model.state_dict().items())
    }


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: _portable(tensor, "model state")
        for name, tensor in sorted(model.state_dict().items())
    }


def _base_payload(
    *,
    model: nn.Module,
    identity: Mapping[str, Any],
    epoch: int,
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    bound_identity = _identity(dict(identity))
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "formula_version": CHECKPOINT_FORMULA_VERSION,
        **bound_identity,
        "epoch": _exact_int(epoch, "epoch"),
        "validation_metrics": _metrics(dict(validation_metrics)),
        "model_state": _model_state(model),
        "parameter_schema": parameter_schema(model),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_best(
    path: Path,
    *,
    model: nn.Module,
    identity: Mapping[str, Any],
    epoch: int,
    validation_metrics: Mapping[str, Any],
) -> Path:
    """Atomically replace the caller-authorized rolling best checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        destination,
        _base_payload(
            model=model,
            identity=identity,
            epoch=epoch,
            validation_metrics=validation_metrics,
        ),
    )
    return destination


def save_last(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    identity: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    state: TrainingState,
) -> Path:
    """Atomically commit the complete resume state after an epoch event."""
    if not isinstance(state, TrainingState) or state.next_epoch < 1:
        raise ValueError("last checkpoint requires committed TrainingState")
    model_parameters = {id(parameter) for parameter in model.parameters()}
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if model_parameters != optimizer_parameters:
        raise ValueError("optimizer must own exactly the checkpoint model parameters")
    early = state.early_stopping.to_payload()
    payload = _base_payload(
        model=model,
        identity=identity,
        epoch=state.next_epoch - 1,
        validation_metrics=validation_metrics,
    )
    payload.update(
        {
            "next_epoch": state.next_epoch,
            "optimizer_state": _portable(optimizer.state_dict(), "optimizer state"),
            "early_stopping_state": early,
            "best_epoch": state.best_epoch,
            "best_validation_loss": state.best_validation_loss,
            "rng_state": _portable(capture_rng_state(), "RNG state"),
            "completed": state.completed,
        }
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(destination, payload)
    return destination


def _validate_versions(payload: Mapping[str, Any]) -> None:
    created_at = payload["created_at"]
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("checkpoint created_at must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError as error:
        raise ValueError("checkpoint created_at must be an ISO timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("checkpoint created_at must include a timezone")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is unsupported")
    if payload["formula_version"] != CHECKPOINT_FORMULA_VERSION:
        raise ValueError("checkpoint formula version is unsupported")


def _validate_saved_model(
    payload: Mapping[str, Any], model: nn.Module
) -> dict[str, torch.Tensor]:
    expected_schema = parameter_schema(model)
    schema = payload["parameter_schema"]
    if type(schema) is not dict or schema != expected_schema:
        raise ValueError("checkpoint parameter schema differs")
    raw_state = payload["model_state"]
    if type(raw_state) is not dict or set(raw_state) != set(expected_schema):
        raise ValueError("checkpoint model state names differ")
    checked: dict[str, torch.Tensor] = {}
    for name, tensor in raw_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint model state {name} is not a tensor")
        descriptor = expected_schema[name]
        if (
            list(tensor.shape) != descriptor["shape"]
            or str(tensor.dtype) != descriptor["dtype"]
        ):
            raise ValueError(f"checkpoint model state {name} differs from schema")
        if tensor.device.type != "cpu":
            raise ValueError("checkpoint model state must be portable CPU tensors")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError("checkpoint model state contains non-finite tensors")
        checked[name] = tensor
    return checked


def _validate_optimizer(
    value: object, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    state = _portable(value, "optimizer state")
    if type(state) is not dict or set(state) != {"state", "param_groups"}:
        raise ValueError("optimizer state schema differs")
    candidate = copy.deepcopy(optimizer)
    try:
        candidate.load_state_dict(copy.deepcopy(state))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"optimizer state is incompatible: {error}") from error
    return state


def _prevalidate_rng(value: object) -> dict[str, Any]:
    state = _portable(value, "RNG state")
    current = capture_rng_state()
    try:
        restore_rng_state(state)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"RNG state is invalid: {error}") from error
    finally:
        restore_rng_state(current)
    return state


def load_last(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_identity: Mapping[str, Any],
) -> TrainingState:
    """Validate fully, then restore model, optimizer and all RNGs exactly."""
    payload = load_torch_artifact(Path(path))
    _exact_mapping(payload, LAST_CHECKPOINT_KEYS, "last checkpoint")
    _validate_versions(payload)
    expected = _identity(dict(expected_identity))
    if {key: payload[key] for key in IDENTITY_KEYS} != expected:
        raise ValueError("checkpoint identity mismatch")
    epoch = _exact_int(payload["epoch"], "checkpoint epoch")
    next_epoch = _exact_int(payload["next_epoch"], "next_epoch", minimum=1)
    if next_epoch != epoch + 1:
        raise ValueError("checkpoint next_epoch is inconsistent")
    _metrics(payload["validation_metrics"])
    early = EarlyStoppingState.from_payload(payload["early_stopping_state"])
    best_epoch = _exact_int(payload["best_epoch"], "best_epoch")
    best_loss = _finite_float(payload["best_validation_loss"], "best_validation_loss")
    if best_epoch != early.best_epoch or best_loss != early.best_validation_loss:
        raise ValueError("early-stopping state differs from best metadata")
    if best_epoch >= next_epoch:
        raise ValueError("best_epoch must precede next_epoch")
    if type(payload["completed"]) is not bool:
        raise ValueError("completed must be a boolean")
    state = TrainingState(
        next_epoch, best_epoch, best_loss, early.bad_epochs, payload["completed"]
    )
    checked_model = _validate_saved_model(payload, model)
    checked_optimizer = _validate_optimizer(payload["optimizer_state"], optimizer)
    checked_rng = _prevalidate_rng(payload["rng_state"])

    old_model = _model_state(model)
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    old_rng = capture_rng_state()
    try:
        model.load_state_dict(checked_model, strict=True)
        optimizer.load_state_dict(checked_optimizer)
        restore_rng_state(checked_rng)
    except BaseException:
        model.load_state_dict(old_model, strict=True)
        optimizer.load_state_dict(old_optimizer)
        restore_rng_state(old_rng)
        raise
    return state
