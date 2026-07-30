"""Fixed and condition-number ridge selection for LLPR curvature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from .config import RidgeConfig


@dataclass(frozen=True)
class RidgeRecord:
    mode: Literal["fixed", "condition_number"]
    value: float


def condition_number_ridge(eigenvalues: Tensor, kappa: float) -> float:
    """Return the smallest float64 ridge targeting condition number *kappa*."""
    values = eigenvalues.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        raise ValueError("eigenvalues must not be empty")
    if not torch.isfinite(values).all():
        raise ValueError("eigenvalues must be finite")
    if not np.isfinite(kappa) or kappa <= 1.0:
        raise ValueError("kappa must be finite and greater than 1")

    mu_min = float(values.min())
    mu_max = float(values.max())
    raw = max(0.0, (mu_max - kappa * mu_min) / (kappa - 1.0))
    return float(np.nextafter(np.float64(raw), np.inf)) if raw > 0.0 else 0.0


def choose_ridge(curvature: Tensor, config: RidgeConfig) -> RidgeRecord:
    """Choose the configured ridge without adding implicit jitter."""
    if config.mode == "fixed":
        return RidgeRecord(mode="fixed", value=float(config.value))
    if config.mode != "condition_number":
        raise ValueError(f"unsupported ridge mode: {config.mode}")

    matrix = curvature.detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("curvature must be a square matrix")
    eigenvalues = torch.linalg.eigvalsh(matrix)
    return RidgeRecord(
        mode="condition_number",
        value=condition_number_ridge(eigenvalues, config.max_condition_number),
    )
