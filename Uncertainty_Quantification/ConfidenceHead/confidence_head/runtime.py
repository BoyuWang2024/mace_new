"""Deterministic process runtime and reproducibility snapshots."""

from __future__ import annotations

import importlib.metadata
import math
import platform
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .identity import code_identity


RNG_SCHEMA_VERSION = 1
ENVIRONMENT_SCHEMA_VERSION = 1


def configure_runtime(seed: int, deterministic: bool, device: str) -> torch.device:
    """Validate and configure the process-wide training runtime."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a boolean")
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False
    return torch.device(device)


def capture_rng_state() -> dict[str, Any]:
    """Capture all supported RNGs using weights-only-safe values."""
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "schema_version": RNG_SCHEMA_VERSION,
        "python": random.getstate(),
        "numpy": {
            "algorithm": algorithm,
            "keys": torch.from_numpy(keys.copy()),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _exact_keys(value: object, expected: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{field} must be a plain dictionary")
    mapping = value
    if set(mapping) != expected:
        raise ValueError(f"{field} schema mismatch")
    return mapping


def _rng_tensor(value: object, field: str, *, expected_numel: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError(f"{field} must be a one-dimensional CPU uint8 tensor")
    if value.numel() != expected_numel:
        raise ValueError(
            f"{field} has {value.numel()} values; expected {expected_numel}"
        )
    return value


def restore_rng_state(state: object) -> None:
    """Validate and restore a state created by :func:`capture_rng_state`."""
    payload = _exact_keys(
        state,
        {"schema_version", "python", "numpy", "torch_cpu", "torch_cuda"},
        "RNG state",
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != RNG_SCHEMA_VERSION
    ):
        raise ValueError("RNG state schema version is unsupported")

    python_state = payload["python"]
    if not isinstance(python_state, tuple):
        raise TypeError("python RNG state must be a tuple")

    numpy_state = _exact_keys(
        payload["numpy"],
        {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"},
        "numpy RNG state",
    )
    algorithm = numpy_state["algorithm"]
    keys = numpy_state["keys"]
    position = numpy_state["position"]
    has_gauss = numpy_state["has_gauss"]
    cached_gaussian = numpy_state["cached_gaussian"]
    if algorithm != "MT19937":
        raise ValueError("numpy RNG algorithm must be MT19937")
    if (
        not isinstance(keys, torch.Tensor)
        or keys.device.type != "cpu"
        or keys.dtype != torch.uint32
        or keys.shape != (624,)
    ):
        raise ValueError("numpy RNG keys must be a length-624 CPU uint32 tensor")
    if type(position) is not int or not 0 <= position <= 624:
        raise ValueError("numpy RNG position is invalid")
    if type(has_gauss) is not int or has_gauss not in {0, 1}:
        raise ValueError("numpy RNG Gaussian flag is invalid")
    if type(cached_gaussian) is not float or not math.isfinite(cached_gaussian):
        raise ValueError("numpy RNG cached Gaussian must be finite")

    cpu_state = _rng_tensor(
        payload["torch_cpu"],
        "torch_cpu RNG state",
        expected_numel=torch.get_rng_state().numel(),
    )
    cuda_states = payload["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise TypeError("torch CUDA RNG state must be a list")
    current_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    if len(cuda_states) != len(current_cuda):
        raise RuntimeError(
            f"CUDA RNG state count {len(cuda_states)} is incompatible with {len(current_cuda)} devices"
        )
    checked_cuda = [
        _rng_tensor(
            item,
            f"CUDA RNG state {index}",
            expected_numel=current_cuda[index].numel(),
        )
        for index, item in enumerate(cuda_states)
    ]

    # Validation is complete before any process RNG is changed.
    try:
        random.Random().setstate(python_state)
    except (TypeError, ValueError) as error:
        raise ValueError("python RNG state is invalid") from error
    restored_numpy = (
        algorithm,
        keys.numpy().copy(),
        position,
        has_gauss,
        cached_gaussian,
    )
    try:
        np.random.RandomState().set_state(restored_numpy)
    except (TypeError, ValueError) as error:
        raise ValueError("numpy RNG state is invalid") from error

    random.setstate(python_state)
    np.random.set_state(restored_numpy)
    torch.set_rng_state(cpu_state)
    if checked_cuda:
        torch.cuda.set_rng_state_all(checked_cuda)


def _wandb_version() -> str | None:
    try:
        return importlib.metadata.version("wandb")
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_snapshot(repo_root: Path) -> dict[str, Any]:
    """Return a JSON-safe environment and source-control snapshot."""
    git = asdict(code_identity(Path(repo_root)))
    wandb_version = _wandb_version()
    wandb: dict[str, Any]
    if wandb_version is None:
        wandb = {"version": None, "available": False}
    else:
        wandb = {"version": wandb_version}
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cudnn_available": torch.backends.cudnn.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "wandb": wandb,
        "git": git,
    }
