"""Continuous cache-only error labels for ConfidenceHead branches."""

from __future__ import annotations

import torch


def _floating_pair(
    prediction: object,
    reference: object,
    *,
    name: str,
    ndim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(prediction, torch.Tensor) or not isinstance(
        reference, torch.Tensor
    ):
        raise ValueError(f"{name} prediction and reference must be tensors")
    if prediction.ndim != ndim or reference.ndim != ndim:
        raise ValueError(f"{name} prediction and reference shape is invalid")
    if prediction.shape != reference.shape:
        raise ValueError(f"{name} prediction and reference shape must match")
    if not prediction.is_floating_point() or not reference.is_floating_point():
        raise ValueError(f"{name} prediction and reference must be floating point")
    if prediction.device != reference.device or prediction.device.type == "meta":
        raise ValueError(f"{name} prediction and reference device must match")
    if not bool(torch.isfinite(prediction).all().item()) or not bool(
        torch.isfinite(reference).all().item()
    ):
        raise ValueError(f"{name} prediction and reference must be finite")
    return prediction, reference


def force_errors(
    prediction: object,
    reference: object,
    target_mode: str,
) -> torch.Tensor:
    """Return detached CPU float64 absolute force errors."""
    prediction_tensor, reference_tensor = _floating_pair(
        prediction, reference, name="force", ndim=2
    )
    if prediction_tensor.shape[1:] != (3,):
        raise ValueError("force prediction and reference shape must be [N_atom, 3]")
    if target_mode not in {"component", "atom_mean"}:
        raise ValueError("force target_mode must be component or atom_mean")
    absolute = (
        prediction_tensor.detach().to(device="cpu", dtype=torch.float64)
        - reference_tensor.detach().to(device="cpu", dtype=torch.float64)
    ).abs()
    return absolute if target_mode == "component" else absolute.mean(dim=1)


def energy_errors(
    prediction: object,
    reference: object,
    num_atoms: object,
) -> torch.Tensor:
    """Return detached CPU float64 absolute per-atom energy errors."""
    prediction_tensor, reference_tensor = _floating_pair(
        prediction, reference, name="energy", ndim=1
    )
    if not isinstance(num_atoms, torch.Tensor) or num_atoms.ndim != 1:
        raise ValueError("energy num_atoms shape must match structures")
    if num_atoms.shape != prediction_tensor.shape:
        raise ValueError("energy num_atoms shape must match structures")
    if num_atoms.device != prediction_tensor.device or num_atoms.device.type == "meta":
        raise ValueError("energy num_atoms device must match predictions")
    if num_atoms.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("energy num_atoms must contain positive integers")
    if not bool(torch.all(num_atoms > 0).item()):
        raise ValueError("energy num_atoms must contain positive integers")
    absolute = (
        prediction_tensor.detach().to(device="cpu", dtype=torch.float64)
        - reference_tensor.detach().to(device="cpu", dtype=torch.float64)
    ).abs()
    return absolute / num_atoms.detach().to(device="cpu", dtype=torch.float64)
