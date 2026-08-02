"""Authoritative JSONL logging with an optional W&B mirror."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import LoggingConfig
from .identity import canonical_json
from .runtime import capture_rng_state, restore_rng_state


def _event_epoch(event: object) -> int:
    if type(event) is not dict:
        raise TypeError("event must be a plain mapping")
    epoch = event.get("epoch")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("event epoch must be a non-negative integer")
    return epoch


class JsonlLogger:
    """Append-only canonical JSON events with strict crash-tail recovery."""

    def __init__(self, path: Path, handle: Any, last_epoch: int | None) -> None:
        self.path = path
        self._handle = handle
        self.last_epoch = last_epoch
        self._closed = False

    @classmethod
    def resume(cls, path: Path) -> JsonlLogger:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch(exist_ok=True)
        raw = destination.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("events JSONL is not valid UTF-8") from error

        committed = text
        truncate_to: int | None = None
        if text and not text.endswith("\n"):
            final_newline = raw.rfind(b"\n")
            tail = text[text.rfind("\n") + 1 :]
            try:
                json.loads(tail)
            except json.JSONDecodeError:
                truncate_to = final_newline + 1
                committed = raw[:truncate_to].decode("utf-8")
            else:
                raise ValueError(
                    "events JSONL has a complete unterminated final record"
                )

        last_epoch: int | None = None
        for line_number, line in enumerate(committed.splitlines(), start=1):
            if not line:
                raise ValueError(f"events JSONL contains a blank line at {line_number}")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"events JSONL is corrupt at line {line_number}"
                ) from error
            try:
                epoch = _event_epoch(event)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"events JSONL has an invalid event at line {line_number}"
                ) from error
            if last_epoch is not None and epoch <= last_epoch:
                raise ValueError("events JSONL epochs must be strictly increasing")
            last_epoch = epoch

        handle = destination.open("r+b")
        if truncate_to is not None:
            handle.truncate(truncate_to)
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0, os.SEEK_END)
        return cls(destination, handle, last_epoch)

    def append(self, event: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("JSONL logger is closed")
        epoch = _event_epoch(event)
        if self.last_epoch is not None and epoch <= self.last_epoch:
            raise ValueError("event epoch must be strictly greater than the last epoch")
        try:
            line = canonical_json(event)
            # canonical_json permits NaN by default; formal logs do not.
            json.dumps(event, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"event is not finite JSON data: {error}") from error
        self._handle.write((line + "\n").encode("utf-8"))
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.last_epoch = epoch

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True


def _import_wandb() -> ModuleType | None:
    try:
        import wandb
    except ModuleNotFoundError as error:
        if error.name != "wandb":
            raise
        return None
    return wandb


class WandbMirror:
    """A non-authoritative mirror of locally committed epoch events."""

    def __init__(self, mode: str, run: Any | None) -> None:
        self.mode = mode
        self._run = run
        self._finished = False
        self._failed = False
        self._failure: str | None = None

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def failure(self) -> str | None:
        return self._failure

    def _disable(self, error: Exception) -> None:
        self._failed = True
        self._failure = f"{type(error).__name__}: {error}"
        self._run = None

    @classmethod
    def start(
        cls,
        logging_config: LoggingConfig,
        run_dir: Path,
        identity: Mapping[str, Any],
        *,
        wandb_module: Any | None = None,
        init_timeout_seconds: float = 30.0,
    ) -> WandbMirror:
        if not logging_config.wandb or logging_config.wandb_mode == "disabled":
            return cls("disabled", None)
        if (
            isinstance(init_timeout_seconds, bool)
            or not isinstance(init_timeout_seconds, (int, float))
            or init_timeout_seconds <= 0
        ):
            raise ValueError("W&B initialization timeout must be positive")
        module = wandb_module if wandb_module is not None else _import_wandb()
        if module is None:
            raise RuntimeError(
                "W&B logging is enabled but wandb is unavailable; install it with `pip install wandb`"
            )
        if type(identity) is not dict:
            identity = dict(identity)
        run_identity = identity.get("run_id")
        if not isinstance(run_identity, str) or not run_identity:
            raise ValueError("W&B identity requires a non-empty run_id")
        name = identity.get("name", run_identity)
        if not isinstance(name, str) or not name:
            raise ValueError("W&B identity name must be a non-empty string")
        destination = Path(run_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        settings = module.Settings(
            init_timeout=float(init_timeout_seconds),
            disable_git=True,
            disable_code=True,
        )

        def initialize(mode: str):
            return module.init(
                project=logging_config.wandb_project,
                dir=str(destination),
                id=run_identity,
                name=name,
                config=dict(identity),
                mode=mode,
                settings=settings,
                reinit="create_new",
            )

        requested = logging_config.wandb_mode
        if requested == "auto":
            communication_errors = (
                module.errors.CommError,
                module.errors.AuthenticationError,
            )
            try:
                return cls("online", initialize("online"))
            except communication_errors:
                return cls("offline", initialize("offline"))
        return cls(requested, initialize(requested))

    def log(self, event: Mapping[str, Any], *, step: int) -> None:
        if self._finished:
            raise RuntimeError("W&B mirror is finished")
        if self._run is not None:
            try:
                self._run.log(dict(event), step=step)
            except Exception as error:
                self._disable(error)

    def update_summary(self, values: Mapping[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("W&B mirror is finished")
        if self._run is not None:
            try:
                self._run.summary.update(dict(values))
            except Exception as error:
                self._disable(error)

    def finish(self) -> None:
        if self._finished:
            return
        try:
            if self._run is not None:
                self._run.finish()
        except Exception as error:
            self._disable(error)
        finally:
            self._finished = True


class TrainingLogger:
    """Commit authoritative events locally before mirroring them externally."""

    def __init__(self, local: JsonlLogger, mirror: WandbMirror) -> None:
        self.local = local
        self.mirror = mirror
        self._closed = False

    @classmethod
    def start(
        cls,
        events_path: Path,
        logging_config: LoggingConfig,
        identity: Mapping[str, Any],
        *,
        wandb_module: Any | None = None,
        init_timeout_seconds: float = 30.0,
    ) -> TrainingLogger:
        local = JsonlLogger.resume(events_path)
        rng_state = capture_rng_state()
        try:
            mirror = WandbMirror.start(
                logging_config,
                Path(events_path).parent,
                identity,
                wandb_module=wandb_module,
                init_timeout_seconds=init_timeout_seconds,
            )
        except BaseException:
            local.close()
            raise
        finally:
            restore_rng_state(rng_state)
        return cls(local, mirror)

    def append_epoch(self, event: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("training logger is closed")
        self.local.append(event)
        rng_state = capture_rng_state()
        try:
            self.mirror.log(event, step=_event_epoch(event))
        finally:
            restore_rng_state(rng_state)

    def update_summary(self, values: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("training logger is closed")
        rng_state = capture_rng_state()
        try:
            self.mirror.update_summary(values)
        finally:
            restore_rng_state(rng_state)

    def close(self) -> None:
        if not self._closed:
            try:
                rng_state = capture_rng_state()
                try:
                    self.mirror.finish()
                finally:
                    restore_rng_state(rng_state)
            finally:
                self.local.close()
                self._closed = True

    finish = close
