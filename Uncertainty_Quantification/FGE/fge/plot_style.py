"""Immutable publication-style settings shared by FGE figures."""

from __future__ import annotations

from dataclasses import dataclass


ORANGE = "#f28e2b"


@dataclass(frozen=True)
class PlotConfig:
    dpi: int = 300
    scatter_max_points: int = 20_000
    scatter_size: float = 2.0
    scatter_alpha: float = 0.035
    random_seed: int = 20260714
    grid_size: int = 160
    gaussian_sigma: float = 1.2
    contour_masses: tuple[float, ...] = (0.50, 0.70, 0.85, 0.95, 0.99)

