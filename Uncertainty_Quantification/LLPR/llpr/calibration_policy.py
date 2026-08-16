"""Stage-specific zero-q policy for force calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


ZERO_Q_POLICY = (
    "exact-zero-force-jacobian-structure-excluded-from-force-alpha-v1"
)
_VARIANTS = ("he", "hf", "hef")


@dataclass(frozen=True)
class ForceCalibrationDecision:
    """One cross-variant decision for all force rows in a structure."""

    exclude_structure: bool
    zero_rows: tuple[int, ...]
    zero_components: tuple[tuple[int, int], ...]


def classify_force_calibration_structure(
    g_forces: Tensor,
    q_by_variant: Mapping[str, Tensor],
    indices: Tensor,
) -> ForceCalibrationDecision:
    """Validate three force-q tensors and classify one whole structure."""
    gradients = g_forces.detach().to(device="cpu", dtype=torch.float64)
    force_indices = indices.detach().to(device="cpu")
    if gradients.ndim != 2 or gradients.shape[0] == 0:
        raise ValueError("force gradients must be a non-empty matrix")
    if not torch.isfinite(gradients).all():
        raise ValueError("force gradients must be finite")
    if force_indices.ndim != 1 or force_indices.numel() != gradients.shape[0]:
        raise ValueError("force indices must match force gradient rows")
    if force_indices.dtype == torch.bool or force_indices.is_floating_point():
        raise ValueError("force indices must contain integers")
    force_indices = force_indices.to(dtype=torch.long)
    if torch.any(force_indices < 0):
        raise ValueError("force indices must be non-negative")
    if torch.unique(force_indices).numel() != force_indices.numel():
        raise ValueError("force indices must be unique")
    if not isinstance(q_by_variant, Mapping) or set(q_by_variant) != set(_VARIANTS):
        raise ValueError("force q variants must be exactly he, hf, and hef")

    zero_masks: dict[str, Tensor] = {}
    for variant in _VARIANTS:
        values = q_by_variant[variant].detach().to(
            device="cpu", dtype=torch.float64
        )
        if values.ndim != 1 or values.numel() != gradients.shape[0]:
            raise ValueError(f"{variant} force q must match force gradient rows")
        if not torch.isfinite(values).all():
            raise ValueError(f"{variant} force q must be finite")
        if torch.any(values < 0.0):
            raise ValueError(f"{variant} force q must be non-negative")
        zero_masks[variant] = values == 0.0

    reference_zero_mask = zero_masks["he"]
    if any(
        not torch.equal(zero_masks[variant], reference_zero_mask)
        for variant in ("hf", "hef")
    ):
        raise ValueError("force zero-q rows must match across variants")

    zero_gradient_mask = torch.all(gradients == 0.0, dim=1)
    if not torch.equal(reference_zero_mask, zero_gradient_mask):
        raise ValueError(
            "force zero-q rows must match exact-zero force gradient rows"
        )

    zero_rows = tuple(
        int(row)
        for row in torch.nonzero(reference_zero_mask, as_tuple=False).reshape(-1)
    )
    zero_components = tuple(
        divmod(int(force_indices[row]), 3) for row in zero_rows
    )
    return ForceCalibrationDecision(
        exclude_structure=bool(zero_rows),
        zero_rows=zero_rows,
        zero_components=zero_components,
    )
