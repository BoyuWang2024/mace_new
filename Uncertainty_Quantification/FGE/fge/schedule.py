"""Step-wise asymmetric triangular learning-rate schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import HardFailure


@dataclass(frozen=True)
class AsymmetricTriangularLR:
    steps_per_cycle: int
    lr_min: float
    lr_max: float
    rise_fraction: float

    def __post_init__(self) -> None:
        if type(self.steps_per_cycle) is not int or self.steps_per_cycle < 2:
            raise HardFailure("steps_per_cycle must be an integer >= 2")
        if not all(math.isfinite(value) for value in (self.lr_min, self.lr_max, self.rise_fraction)):
            raise HardFailure("learning-rate schedule values must be finite")
        if self.lr_min <= 0 or self.lr_max <= self.lr_min:
            raise HardFailure("learning rates must satisfy 0 < lr_min < lr_max")
        if not 0.0 < self.rise_fraction < 1.0:
            raise HardFailure("rise_fraction must be in (0, 1)")

    @property
    def peak_step(self) -> int:
        return min(
            self.steps_per_cycle - 2,
            max(1, round((self.steps_per_cycle - 1) * self.rise_fraction)),
        )

    def value(self, step: int) -> float:
        if type(step) is not int or step < 0:
            raise HardFailure("schedule step must be a nonnegative integer")
        position = step % self.steps_per_cycle
        if position <= self.peak_step:
            fraction = position / self.peak_step
        else:
            fraction = (self.steps_per_cycle - 1 - position) / (
                self.steps_per_cycle - 1 - self.peak_step
            )
        return self.lr_min + max(0.0, fraction) * (self.lr_max - self.lr_min)
