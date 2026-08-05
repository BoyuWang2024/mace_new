"""Best-effort W&B logging with an authoritative local JSONL fallback."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _finite_payload(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _finite_payload(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_payload(item) for item in value)
    return False


class WandbLogger:
    def __init__(self, run: Any, fallback_path: Path, warnings: list[dict[str, str]]) -> None:
        self._run = run
        self._fallback_path = fallback_path
        self._warnings = warnings
        self._warned = False

    def _fallback(self, row: Mapping[str, Any], reason: Exception | None = None) -> None:
        if not _finite_payload(row):
            raise ValueError("W&B fallback payload must be finite JSON")
        self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with self._fallback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        if not self._warned:
            self._warnings.append(
                {"code": "wandb_fallback", "message": str(reason or "W&B disabled")}
            )
            self._warned = True

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        row = {"event": "log", "metrics": dict(metrics), "step": step}
        if self._run is None:
            self._fallback(row)
            return
        try:
            self._run.log(dict(metrics), step=step)
        except Exception as exc:  # W&B is explicitly non-fatal
            self._run = None
            self._fallback(row, exc)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as exc:  # W&B is explicitly non-fatal
            self._run = None
            self._fallback({"event": "finish"}, exc)


def create_wandb_logger(
    config: Mapping[str, Any],
    work_dir: Path,
    *,
    warnings: list[dict[str, str]],
    wandb_module: Any | None = None,
) -> WandbLogger:
    """Initialize W&B when requested and downgrade every failure to fallback logging."""
    fallback = Path(work_dir) / "wandb" / "fallback_history.jsonl"
    if not config.get("enabled") or config.get("mode") == "disabled":
        return WandbLogger(None, fallback, warnings)
    try:
        module = wandb_module or importlib.import_module("wandb")
        run = module.init(
            project=config.get("project"),
            entity=config.get("entity") or None,
            mode=config.get("mode", "online"),
            dir=str(Path(work_dir) / "wandb"),
        )
        return WandbLogger(run, fallback, warnings)
    except Exception as exc:  # W&B is explicitly non-fatal
        logger = WandbLogger(None, fallback, warnings)
        logger._fallback({"event": "init_failure"}, exc)
        return logger
