"""Deterministic N-out-of-N bootstrap sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .errors import HardFailure


@dataclass(frozen=True)
class BootstrapSample:
    seed: int
    indices: NDArray[np.int64]
    oob_indices: NDArray[np.int64]


def draw_bootstrap_sample(*, size: int, seed: int, sample_size: int | None = None) -> BootstrapSample:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise HardFailure("bootstrap population size must be at least 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise HardFailure("bootstrap seed must be a non-negative integer")
    count = size if sample_size is None else sample_size
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise HardFailure("bootstrap sample_size must be at least 1")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, size, size=count, dtype=np.int64)
    present = np.zeros(size, dtype=bool)
    present[np.unique(indices)] = True
    oob = np.flatnonzero(~present).astype(np.int64, copy=False)
    return BootstrapSample(seed=seed, indices=indices, oob_indices=oob)
