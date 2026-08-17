"""Validated single-member MACE inference for one prepared graph batch."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .artifacts import sha256_file
from .errors import HardFailure


_SUPPORTED_DOMAINS = {
    ("energy", "forces"),
    ("energy", "forces", "stress"),
}


def load_member_model(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Load one hash-pinned full MACE model and configure it for inference."""
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise HardFailure(f"member model must be a regular file: {source}")
    digest = sha256_file(source)
    if digest != expected_sha256:
        raise HardFailure(f"member model SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    original_jit_load = torch.jit.load

    def load_embedded_jit_on_cpu(value: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs["map_location"] = torch.device("cpu")
        return original_jit_load(value, *args, **kwargs)

    torch.jit.load = load_embedded_jit_on_cpu
    try:
        model = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise HardFailure(f"could not load trusted member model {source}: {error}") from error
    finally:
        torch.jit.load = original_jit_load
    if not isinstance(model, nn.Module):
        raise HardFailure("trusted member model is not a torch.nn.Module")
    try:
        model.to(device=torch.device(device), dtype=dtype)
    except (RuntimeError, TypeError, ValueError) as error:
        raise HardFailure(f"could not configure member model for inference: {error}") from error
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _batch_shape(batch: Any) -> tuple[int, int, Mapping[str, Any]]:
    ptr = getattr(batch, "ptr", None)
    if not isinstance(ptr, Tensor) or ptr.ndim != 1 or ptr.numel() < 2:
        raise HardFailure("batch ptr must be a one-dimensional tensor")
    pointer = ptr.detach().cpu()
    if pointer.dtype == torch.bool or pointer.is_floating_point():
        raise HardFailure("batch ptr must contain integer offsets")
    if int(pointer[0].item()) != 0 or (pointer[1:] <= pointer[:-1]).any().item():
        raise HardFailure("batch ptr must start at zero and increase strictly")
    structures = pointer.numel() - 1
    atoms = int(pointer[-1].item())
    if getattr(batch, "num_graphs", structures) != structures:
        raise HardFailure("batch structure count does not match ptr")
    if getattr(batch, "num_nodes", atoms) != atoms:
        raise HardFailure("batch atom count does not match ptr")
    positions = getattr(batch, "positions", None)
    if isinstance(positions, Tensor) and (positions.ndim != 2 or positions.shape[0] != atoms):
        raise HardFailure("batch atom positions do not match ptr")
    try:
        data = batch.to_dict()
    except (AttributeError, TypeError, ValueError) as error:
        raise HardFailure("batch cannot be converted to a MACE input mapping") from error
    if not isinstance(data, Mapping):
        raise HardFailure("batch to_dict result must be a mapping")
    return structures, atoms, data


def _array(output: Mapping[str, Any], name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = output.get(name)
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise HardFailure(f"model {name} output has invalid shape; expected {shape}")
    array = value.detach().cpu().numpy()
    if not np.isfinite(array).all():
        raise HardFailure(f"model {name} output contains NaN or Inf")
    return array


def predict_batch(
    model: nn.Module,
    batch: Any,
    *,
    domains: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Predict energy, forces, and optional stress for one native MACE batch."""
    if domains not in _SUPPORTED_DOMAINS:
        raise HardFailure("domains must be energy/forces with optional stress")
    if not isinstance(model, nn.Module):
        raise HardFailure("model must be a torch.nn.Module")
    structures, atoms, data = _batch_shape(batch)
    compute_stress = "stress" in domains
    with torch.enable_grad():
        output = model(
            data,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=compute_stress,
        )
    if not isinstance(output, Mapping):
        raise HardFailure("model output must be a mapping")
    result = {
        "energy": _array(output, "energy", (structures,)),
        "forces": _array(output, "forces", (atoms, 3)),
    }
    if compute_stress:
        stress = output.get("stress")
        if isinstance(stress, Tensor) and tuple(stress.shape) == (structures, 1, 3, 3):
            output = dict(output)
            output["stress"] = stress[:, 0]
        result["stress"] = _array(output, "stress", (structures, 3, 3))
    return result


__all__ = ["load_member_model", "predict_batch"]
