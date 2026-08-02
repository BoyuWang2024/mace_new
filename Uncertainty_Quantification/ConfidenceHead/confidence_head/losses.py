"""Weighted hard cross-entropy losses and sample-weighted metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BranchLosses:
    """Per-branch mean losses and their configured weighted total."""

    force: torch.Tensor | None
    energy: torch.Tensor | None
    total: torch.Tensor


def _coefficients(force: object, energy: object) -> tuple[float, float]:
    values: list[float] = []
    for value in (force, energy):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("loss coefficients must be finite non-negative numbers")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError("loss coefficients must be finite non-negative numbers")
        values.append(number)
    if values == [0.0, 0.0]:
        raise ValueError("loss coefficients cannot both be zero")
    return values[0], values[1]


def _paired(
    branch: str,
    logits: torch.Tensor | None,
    labels: torch.Tensor | None,
    coefficient: float,
) -> bool:
    if coefficient == 0.0:
        if logits is not None or labels is not None:
            raise ValueError(f"disabled {branch} branch must be absent")
        return False
    if logits is None and labels is None:
        raise ValueError(f"enabled {branch} branch requires logits and labels")
    if logits is None or labels is None:
        raise ValueError(f"{branch} logits and labels must both be provided")
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError(f"{branch} logits and labels must both be tensors")
    return True


def _common_inputs(branch: str, logits: torch.Tensor, labels: torch.Tensor) -> None:
    if not logits.is_floating_point():
        raise ValueError(f"{branch} logits must have a floating dtype")
    if labels.dtype is not torch.long:
        raise ValueError(f"{branch} labels must have torch.long dtype")
    if logits.device != labels.device:
        raise ValueError(f"{branch} logits and labels must use the same device")
    if logits.shape[-1] < 3:
        raise ValueError(f"{branch} class dimension must be at least 3")
    if logits.shape[0] == 0:
        raise ValueError(f"{branch} sample axis cannot be empty")
    if not torch.isfinite(logits).all():
        raise ValueError(f"{branch} logits must be finite")
    if torch.any(labels < 0) or torch.any(labels >= logits.shape[-1]):
        raise ValueError(f"{branch} labels must be in class range")


def _force_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        if labels.ndim != 1:
            raise ValueError("atom-mean force labels shape must match logits samples")
        if labels.shape[0] != logits.shape[0]:
            raise ValueError("force logits and labels sample axes must match")
        _common_inputs("force", logits, labels)
        return F.cross_entropy(logits, labels, reduction="mean")
    if logits.ndim == 3:
        if logits.shape[1] != 3 or labels.ndim != 2 or labels.shape[1] != 3:
            raise ValueError(
                "component force logits and labels shape must be [N,3,B] and [N,3]"
            )
        if labels.shape != logits.shape[:2]:
            raise ValueError("force logits and labels sample axes must match")
        _common_inputs("force", logits, labels)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean"
        )
    raise ValueError("force logits rank must be 2 or 3")


def _energy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("energy logits rank must be 2")
    if labels.ndim != 1:
        raise ValueError("energy labels shape must be [N]")
    if labels.shape[0] != logits.shape[0]:
        raise ValueError("energy logits and labels sample axes must match")
    _common_inputs("energy", logits, labels)
    return F.cross_entropy(logits, labels, reduction="mean")


def joint_loss(
    force_logits: torch.Tensor | None,
    force_labels: torch.Tensor | None,
    energy_logits: torch.Tensor | None,
    energy_labels: torch.Tensor | None,
    *,
    force_coefficient: float,
    energy_coefficient: float,
) -> BranchLosses:
    """Mean each enabled hard-CE branch before applying its coefficient."""
    force_weight, energy_weight = _coefficients(force_coefficient, energy_coefficient)
    force_enabled = _paired("force", force_logits, force_labels, force_weight)
    energy_enabled = _paired("energy", energy_logits, energy_labels, energy_weight)

    force = _force_loss(force_logits, force_labels) if force_enabled else None
    energy = _energy_loss(energy_logits, energy_labels) if energy_enabled else None
    if force is not None and energy is not None and force.device != energy.device:
        raise ValueError("enabled branch losses must use the same device")
    total = force * force_weight if force is not None else energy * energy_weight
    if force is not None and energy is not None:
        total = force * force_weight + energy * energy_weight
    return BranchLosses(force=force, energy=energy, total=total)


class LossAccumulator:
    """Accumulate branch loss sums with their true sample counts."""

    def __init__(self) -> None:
        self._sums = {"force": 0.0, "energy": 0.0}
        self._counts = {"force": 0, "energy": 0}

    def update(
        self, branch: str, mean_loss: Real | torch.Tensor, sample_count: int
    ) -> None:
        if branch not in self._sums:
            raise ValueError("branch must be force or energy")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
        ):
            raise ValueError("sample_count must be a positive integer")
        if isinstance(mean_loss, torch.Tensor):
            if mean_loss.ndim != 0:
                raise ValueError("mean_loss must be a scalar numeric value")
            if mean_loss.is_complex() or mean_loss.dtype == torch.bool:
                raise ValueError("mean_loss must be a real numeric value")
            value = float(mean_loss.detach().cpu().item())
        elif isinstance(mean_loss, bool) or not isinstance(mean_loss, Real):
            raise ValueError("mean_loss must be a scalar numeric value")
        else:
            value = float(mean_loss)
        if not math.isfinite(value):
            raise ValueError("mean_loss must be finite")
        self._sums[branch] += value * sample_count
        self._counts[branch] += sample_count

    def mean(self, branch: str) -> float:
        if branch not in self._sums:
            raise ValueError("branch must be force or energy")
        if self._counts[branch] == 0:
            raise ValueError(f"no accumulated samples for {branch}")
        return self._sums[branch] / self._counts[branch]

    def total(self, force_coefficient: float, energy_coefficient: float) -> float:
        force_weight, energy_weight = _coefficients(
            force_coefficient, energy_coefficient
        )
        total = 0.0
        if force_weight > 0.0:
            total += force_weight * self.mean("force")
        if energy_weight > 0.0:
            total += energy_weight * self.mean("energy")
        return total
