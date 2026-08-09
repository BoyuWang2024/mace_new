"""Deterministic sampling and density analysis for FGE plots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from torch import Tensor

from .errors import HardFailure
from .plot_style import PlotConfig


@dataclass(frozen=True)
class PanelAnalysis:
    log_uncertainty: Tensor
    log_residual: Tensor
    sample_indices: Tensor
    density: np.ndarray
    uncertainty_edges: np.ndarray
    residual_edges: np.ndarray
    contour_levels: tuple[float, ...]
    pearson_log10: float
    spearman: float
    excluded: dict[str, int]

    @property
    def valid_count(self) -> int:
        return int(self.log_uncertainty.numel())


def deterministic_indices(count: int, maximum: int, seed: int) -> Tensor:
    if count < 0 or maximum <= 0:
        raise ValueError("count must be nonnegative and maximum must be positive")
    if count <= maximum:
        return torch.arange(count, dtype=torch.int64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(count, generator=generator, dtype=torch.int64)[:maximum]


def _contour_levels(density: np.ndarray, masses: tuple[float, ...]) -> tuple[float, ...]:
    positive = density[density > 0]
    if positive.size == 0:
        return ()
    descending = np.sort(positive)[::-1]
    cumulative = np.cumsum(descending) / descending.sum()
    thresholds = []
    for mass in masses:
        index = min(int(np.searchsorted(cumulative, mass, side="left")), descending.size - 1)
        thresholds.append(float(descending[index]))
    return tuple(float(value) for value in np.unique(thresholds))


def analyze_log_panel(
    uncertainty: Tensor, residual: Tensor, config: PlotConfig
) -> PanelAnalysis:
    uncertainty = torch.as_tensor(uncertainty, dtype=torch.float64, device="cpu").reshape(-1)
    residual = torch.as_tensor(residual, dtype=torch.float64, device="cpu").reshape(-1)
    if uncertainty.shape != residual.shape:
        raise HardFailure("uncertainty and residual arrays have mismatched shapes")

    nan = torch.isnan(uncertainty) | torch.isnan(residual)
    inf = (~nan) & (torch.isinf(uncertainty) | torch.isinf(residual))
    finite = ~(nan | inf)
    zero = finite & ((uncertainty == 0) | (residual == 0))
    negative = finite & (~zero) & ((uncertainty < 0) | (residual < 0))
    valid = finite & (uncertainty > 0) & (residual > 0)
    excluded = {
        "zero": int(zero.sum().item()),
        "negative": int(negative.sum().item()),
        "nan": int(nan.sum().item()),
        "inf": int(inf.sum().item()),
    }
    if int(valid.sum().item()) < 2:
        raise HardFailure("insufficient positive finite points for log plotting")

    log_uncertainty = torch.log10(uncertainty[valid])
    log_residual = torch.log10(residual[valid])
    x = log_uncertainty.numpy()
    y = log_residual.numpy()
    density, x_edges, y_edges = np.histogram2d(x, y, bins=config.grid_size)
    density = gaussian_filter(density, sigma=config.gaussian_sigma)
    pearson = float(pearsonr(x, y).statistic)
    spearman = float(spearmanr(x, y).statistic)
    return PanelAnalysis(
        log_uncertainty=log_uncertainty,
        log_residual=log_residual,
        sample_indices=deterministic_indices(
            log_uncertainty.numel(), config.scatter_max_points, config.random_seed
        ),
        density=density,
        uncertainty_edges=x_edges,
        residual_edges=y_edges,
        contour_levels=_contour_levels(density, config.contour_masses),
        pearson_log10=pearson,
        spearman=spearman,
        excluded=excluded,
    )
