"""Continuous error-pair density plots for ConfidenceHead evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr


class DensityPlotError(ValueError):
    """Raised when density-plot inputs cannot be represented safely."""


@dataclass(frozen=True)
class ErrorPairs:
    expected: np.ndarray
    actual: np.ndarray
    expected_log10: np.ndarray
    actual_log10: np.ndarray
    sample_ids: list[Any]
    atom_indices: list[int] | None
    unit: str
    task: str
    order: int | None
    audit: dict[str, int]

    @property
    def log_expected(self) -> np.ndarray:
        return self.expected_log10

    @property
    def log_actual(self) -> np.ndarray:
        return self.actual_log10


@dataclass(frozen=True)
class DensityMetrics:
    valid_count: int
    spearman_rho: float
    log10_pearson_r: float
    total_count: int | None = None
    excluded_count: int = 0


@dataclass(frozen=True)
class DensityGrid:
    density: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    contour_levels: tuple[float, ...]


@dataclass(frozen=True)
class DensityPlotSettings:
    dpi: int = 300
    figure_size: tuple[float, float] = (7.0, 7.0)
    scatter_max_points: int = 20000
    scatter_seed: int = 20260714
    scatter_size: float = 2.0
    scatter_alpha: float = 0.035
    grid_size: int = 160
    gaussian_sigma: float = 1.2
    contour_masses: tuple[float, ...] = (0.50, 0.70, 0.85, 0.95, 0.99)
    log_margin: float = 0.05


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)


def build_error_pairs(expected: Any, actual: Any, sample_ids: Sequence[Any],
                      atom_indices: Sequence[int] | None, unit: str, task: str,
                      order: int | None) -> ErrorPairs:
    x, y = _array(expected), _array(actual)
    total = len(x)
    if len(y) != total or len(sample_ids) != total or (atom_indices is not None and len(atom_indices) != total):
        raise DensityPlotError(f"{task} order={order}: input length mismatch")
    finite = np.isfinite(x) & np.isfinite(y)
    positive = (x > 0) & (y > 0)
    valid = finite & positive
    audit = {"total_count": total, "valid_count": int(valid.sum()),
             "excluded_nonfinite": int((~finite).sum()),
             "excluded_nonpositive": int((finite & ~positive).sum())}
    if not valid.any():
        raise DensityPlotError(f"{task} order={order}: no valid positive finite error pairs")
    valid_x, valid_y = x[valid].astype(float), y[valid].astype(float)
    return ErrorPairs(valid_x, valid_y, np.log10(valid_x), np.log10(valid_y),
                      [sample_ids[i] for i in np.flatnonzero(valid)],
                      ([atom_indices[i] for i in np.flatnonzero(valid)] if atom_indices is not None else None),
                      unit, task, order, audit)


def _validate_xy(expected: Any, actual: Any, *, context: str = "") -> tuple[np.ndarray, np.ndarray]:
    x, y = _array(expected), _array(actual)
    if len(x) != len(y):
        raise DensityPlotError(f"{context} length mismatch")
    if len(x) < 2:
        raise DensityPlotError(f"{context} requires at least two points")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise DensityPlotError(f"{context} values must be finite")
    if not (x > 0).all() or not (y > 0).all():
        raise DensityPlotError(f"{context} values must be positive")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise DensityPlotError(f"{context} coordinate must not be constant")
    return x, y


def compute_log_metrics(expected: Any, actual: Any) -> DensityMetrics:
    x, y = _validate_xy(expected, actual)
    lx, ly = np.log10(x), np.log10(y)
    rho = float(spearmanr(lx, ly).statistic)
    pearson = float(pearsonr(lx, ly).statistic)
    if not np.isfinite(rho) or not np.isfinite(pearson):
        raise DensityPlotError("log metrics are non-finite")
    return DensityMetrics(len(x), rho, pearson)


def sample_scatter_indices(count: int, max_points: int = 20000, seed: int = 20260714) -> np.ndarray:
    if count < 0 or max_points < 0:
        raise DensityPlotError("count and max_points must be non-negative")
    if count <= max_points:
        return np.arange(count, dtype=int)
    return np.random.default_rng(seed).choice(count, size=max_points, replace=False)


def compute_density_grid(expected: Any, actual: Any, grid_size: int = 160,
                         sigma: float = 1.2, contour_masses: Sequence[float] = (0.50, 0.70, 0.85, 0.95, 0.99),
                         log_margin: float = 0.05) -> DensityGrid:
    x, y = _validate_xy(expected, actual)
    if grid_size < 2:
        raise DensityPlotError("grid_size must be at least two")
    if not np.isfinite(sigma) or sigma < 0:
        raise DensityPlotError("sigma must be finite and non-negative")
    if not np.isfinite(log_margin) or log_margin < 0:
        raise DensityPlotError("log_margin must be finite and non-negative")
    if not contour_masses:
        raise DensityPlotError("at least one contour mass is required")
    lx, ly = np.log10(x), np.log10(y)
    x_margin, y_margin = log_margin * np.ptp(lx), log_margin * np.ptp(ly)
    x_edges = np.linspace(lx.min() - x_margin, lx.max() + x_margin, grid_size + 1)
    y_edges = np.linspace(ly.min() - y_margin, ly.max() + y_margin, grid_size + 1)
    hist, _, _ = np.histogram2d(lx, ly, bins=(x_edges, y_edges))
    density = gaussian_filter(hist.astype(float), sigma=sigma, mode="nearest")
    total = density.sum()
    if not np.isfinite(total) or total <= 0:
        raise DensityPlotError("density is empty or non-finite")
    density /= total
    positive = np.sort(density[density > 0])[::-1]
    if not len(positive):
        raise DensityPlotError("density has no positive bins")
    flat = np.cumsum(positive)
    levels = []
    for mass in contour_masses:
        if not 0 < mass <= 1:
            raise DensityPlotError("contour masses must be in (0, 1]")
        levels.append(float(positive[min(np.searchsorted(flat, mass, side="left"), len(positive)-1)]))
    levels = np.sort(np.asarray(levels))
    # Matplotlib requires strictly increasing levels; preserve duplicate mass thresholds.
    for i in range(1, len(levels)):
        if levels[i] <= levels[i - 1]:
            levels[i] = np.nextafter(levels[i - 1], np.inf)
    return DensityGrid(density, (x_edges[:-1] + x_edges[1:]) / 2,
                       (y_edges[:-1] + y_edges[1:]) / 2, tuple(levels.tolist()))


def render_density_plot(output_stem: str | Path, pairs: ErrorPairs, density: DensityGrid,
                        metrics: DensityMetrics, settings: DensityPlotSettings | None = None) -> tuple[Path, Path]:
    settings = settings or DensityPlotSettings()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stem = Path(output_stem)
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    if png.exists() or pdf.exists():
        raise DensityPlotError(f"refusing to overwrite existing output for {stem}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    for suffix in (".png", ".pdf"):
        handle = tempfile.NamedTemporaryFile(prefix=f".{stem.name}.", suffix=suffix,
                                             dir=stem.parent, delete=False)
        handle.close()
        temporary.append(Path(handle.name))
    idx = sample_scatter_indices(len(pairs.expected), settings.scatter_max_points, settings.scatter_seed)
    fig, ax = plt.subplots(figsize=settings.figure_size, dpi=settings.dpi)
    try:
        gx, gy = 10 ** density.x_centers, 10 ** density.y_centers
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        ax.contour(X, Y, density.density, levels=density.contour_levels, colors="#c45a19", linewidths=0.9)
        ax.scatter(pairs.expected[idx], pairs.actual[idx], s=settings.scatter_size, alpha=settings.scatter_alpha,
                   color="#e67e22", rasterized=True, edgecolors="none")
        lo = min(float(pairs.expected.min()), float(pairs.actual.min()))
        hi = max(float(pairs.expected.max()), float(pairs.actual.max()))
        xs = np.logspace(np.log10(lo), np.log10(hi), 200)
        ax.fill_between(xs, xs * 0.9, xs * 1.1, color="0.88", zorder=0)
        ax.plot(xs, xs, "--", color="0.35", linewidth=0.9)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_aspect("equal", adjustable="box")
        ax.grid(True, which="both", color="0.85", linewidth=0.5)
        if pairs.task.lower() == "energy":
            xlabel = ylabel = "Error (eV/atom)"
        else:
            xlabel = ylabel = "Absolute error (eV/Angstrom)"
        ax.set_xlabel(f"Expected {xlabel}"); ax.set_ylabel(f"Actual {ylabel}")
        excluded = pairs.audit.get("total_count", len(pairs.expected)) - len(pairs.expected)
        text = (f"Spearman rho = {metrics.spearman_rho:.3f}\n"
                f"log10 Pearson r = {metrics.log10_pearson_r:.3f}\n"
                f"valid/total = {len(pairs.expected)}/{pairs.audit.get('total_count', len(pairs.expected))}\n"
                f"excluded = {excluded}")
        ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9})
        fig.tight_layout()
        fig.savefig(temporary[0], dpi=settings.dpi, format="png")
        fig.savefig(temporary[1], format="pdf")
        if any(path.stat().st_size == 0 for path in temporary):
            raise DensityPlotError("rendered temporary output is empty")
        os.replace(temporary[0], png)
        os.replace(temporary[1], pdf)
    except Exception as exc:
        plt.close(fig)
        for path in (*temporary, png, pdf):
            if path.exists():
                path.unlink()
        raise DensityPlotError(f"failed to render density plot: {exc}") from exc
    plt.close(fig)
    if not png.exists() or png.stat().st_size == 0 or not pdf.exists() or pdf.stat().st_size == 0:
        raise DensityPlotError("rendered output is missing or empty")
    return png, pdf
