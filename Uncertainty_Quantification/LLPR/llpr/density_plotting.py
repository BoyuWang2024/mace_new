"""Independent numerical contract for MACE LLPR density publication plots."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


CARNET_SELECTED = (
    ("he", "energy"),
    ("hf", "forces"),
    ("hef", "energy"),
    ("hef", "forces"),
)

CARNET_DENSITY_CONFIG = {
    "grid_size": 160,
    "gaussian_sigma": 1.2,
    "contour_masses": (0.5, 0.7, 0.85, 0.95, 0.99),
    "scatter_max_points": 20000,
    "random_seed": 20260714,
    "log_margin": 0.05,
    "figure_size": (7.0, 7.0),
    "color": "#f28e2b",
    "scatter_size": 12.0,
    "scatter_alpha": 0.04,
    "dpi": 300,
}

CARNET_FIGURE_STEMS = {
    ("he", "energy"): "llpr_he_energy_uncertainty_vs_residual",
    ("hf", "forces"): "llpr_hf_force_uncertainty_vs_residual",
    ("hef", "energy"): "llpr_hef_energy_uncertainty_vs_residual",
    ("hef", "forces"): "llpr_hef_force_uncertainty_vs_residual",
}

CARNET_STATISTICS_FIELDS = (
    "variant",
    "target",
    "unit",
    "rows",
    "finite_rows",
    "nonfinite_rows",
    "nonpositive_std_rows",
    "zero_absolute_residual_rows",
    "metric_rows",
    "log_plot_rows",
    "excluded_from_log_rows",
    "pearson_log",
    "spearman_log",
    "correlation_rows",
    "correlation_degenerate",
    "axis_min",
    "axis_max",
)


def deterministic_sample_indices(total: int, maximum: int, seed: int) -> np.ndarray:
    """Return reproducible duplicate-free sample indices."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (total, maximum, seed)):
        raise ValueError("sampling arguments must be integers")
    if total < 0 or maximum <= 0:
        raise ValueError("sampling total must be non-negative and maximum positive")
    if total <= maximum:
        return np.arange(total, dtype=np.int64)
    return np.random.default_rng(seed).choice(total, size=maximum, replace=False)


def _pair_limits(
    panels: Mapping[tuple[str, str], Any],
    paths: tuple[tuple[str, str], ...],
    margin: float,
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for path in paths:
        panel = panels[path]
        for source in (panel.uncertainty, panel.absolute_residual):
            array = np.asarray(source, dtype=np.float64)
            positive = array[np.isfinite(array) & (array > 0.0)]
            if positive.size:
                values.append(positive)
    if not values:
        return (1.0e-12, 1.0)
    pooled = np.concatenate(values)
    low = float(np.log10(np.min(pooled)))
    high = float(np.log10(np.max(pooled)))
    padding = max(margin * (high - low), margin)
    return (10.0 ** (low - padding), 10.0 ** (high + padding))


def carnet_shared_limits(
    panels: Mapping[tuple[str, str], Any], *, margin: float
) -> dict[tuple[str, str], tuple[float, float]]:
    """Return local paired limits for He/Hef energy and Hf/Hef forces."""
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("log margin must be positive and finite")
    missing = set(CARNET_SELECTED) - set(panels)
    if missing:
        raise ValueError(f"missing density panels: {sorted(missing)}")
    energy_paths = (("he", "energy"), ("hef", "energy"))
    force_paths = (("hf", "forces"), ("hef", "forces"))
    energy = _pair_limits(panels, energy_paths, margin)
    forces = _pair_limits(panels, force_paths, margin)
    return {
        ("he", "energy"): energy,
        ("hf", "forces"): forces,
        ("hef", "energy"): energy,
        ("hef", "forces"): forces,
    }


def _log_mask(panel: Any) -> np.ndarray:
    std = np.asarray(panel.uncertainty, dtype=np.float64)
    residual = np.asarray(panel.absolute_residual, dtype=np.float64)
    return np.isfinite(std) & np.isfinite(residual) & (std > 0.0) & (residual > 0.0)


def _correlations(panel: Any, mask: np.ndarray) -> tuple[float, float, int, int]:
    count = int(np.count_nonzero(mask))
    if count < 2:
        return (0.0, 0.0, count, 1)
    x_values = np.log10(np.asarray(panel.uncertainty, dtype=np.float64)[mask])
    y_values = np.log10(np.asarray(panel.absolute_residual, dtype=np.float64)[mask])
    if np.ptp(x_values) == 0.0 or np.ptp(y_values) == 0.0:
        return (0.0, 0.0, count, 1)
    pearson = float(np.corrcoef(x_values, y_values)[0, 1])
    rank_x = np.argsort(np.argsort(x_values, kind="stable"), kind="stable")
    rank_y = np.argsort(np.argsort(y_values, kind="stable"), kind="stable")
    spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return (pearson, spearman, count, int(not (math.isfinite(pearson) and math.isfinite(spearman))))


def carnet_statistics(panels: Mapping[tuple[str, str], Any], limits: Mapping[tuple[str, str], tuple[float, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, target in CARNET_SELECTED:
        panel = panels[(variant, target)]
        std = np.asarray(panel.uncertainty, dtype=np.float64)
        residual = np.asarray(panel.absolute_residual, dtype=np.float64)
        finite = np.isfinite(std) & np.isfinite(residual)
        metric = finite & (std > 0.0)
        mask = metric & (residual > 0.0)
        pearson, spearman, correlation_rows, degenerate = _correlations(panel, mask)
        lower, upper = limits[(variant, target)]
        rows.append({"variant": variant, "target": target, "unit": str(panel.unit), "rows": int(std.size), "finite_rows": int(np.count_nonzero(finite)), "nonfinite_rows": int(np.count_nonzero(~finite)), "nonpositive_std_rows": int(np.count_nonzero(finite & (std <= 0.0))), "zero_absolute_residual_rows": int(np.count_nonzero(finite & (residual == 0.0))), "metric_rows": int(np.count_nonzero(metric)), "log_plot_rows": int(np.count_nonzero(mask)), "excluded_from_log_rows": int(std.size - np.count_nonzero(mask)), "pearson_log": pearson, "spearman_log": spearman, "correlation_rows": correlation_rows, "correlation_degenerate": degenerate, "axis_min": lower, "axis_max": upper})
    return rows


def write_carnet_statistics(path: Any, rows: list[dict[str, Any]]) -> None:
    import csv
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CARNET_STATISTICS_FIELDS), lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _render_panel(panel: Any, path: tuple[str, str], limits: tuple[float, float], staging: Any, dpi: int) -> None:
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    figure, axis = plt.subplots(figsize=CARNET_DENSITY_CONFIG["figure_size"])
    try:
        lower, upper = limits
        diagonal = np.geomspace(lower, upper, 256)
        axis.fill_between(diagonal, lower, diagonal, color="0.90")
        mask = _log_mask(panel)
        candidates = np.flatnonzero(mask)
        sample = candidates[deterministic_sample_indices(int(candidates.size), int(CARNET_DENSITY_CONFIG["scatter_max_points"]), int(CARNET_DENSITY_CONFIG["random_seed"]))]
        axis.scatter(np.asarray(panel.uncertainty)[sample], np.asarray(panel.absolute_residual)[sample], color=CARNET_DENSITY_CONFIG["color"], s=CARNET_DENSITY_CONFIG["scatter_size"], alpha=CARNET_DENSITY_CONFIG["scatter_alpha"], edgecolors="none", rasterized=True)
        edges = np.linspace(math.log10(lower), math.log10(upper), 161)
        if candidates.size:
            histogram, _, _ = np.histogram2d(np.log10(np.asarray(panel.uncertainty)[mask]), np.log10(np.asarray(panel.absolute_residual)[mask]), bins=(edges, edges))
            density = gaussian_filter(histogram.T, sigma=1.2)
            total = density.sum()
            if total > 0:
                sorted_density = np.sort(density.ravel())[::-1]
                cumulative = np.cumsum(sorted_density) / total
                levels = np.unique([sorted_density[min(int(np.searchsorted(cumulative, mass)), sorted_density.size - 1)] for mass in CARNET_DENSITY_CONFIG["contour_masses"]])
                if levels.size:
                    centers = (edges[:-1] + edges[1:]) / 2.0
                    axis.contour(10.0 ** centers, 10.0 ** centers, density, levels=levels, colors=CARNET_DENSITY_CONFIG["color"])
        axis.plot(diagonal, diagonal, color="black")
        pearson, spearman, _, _ = _correlations(panel, mask)
        axis.text(0.04, 0.96, f"Pearson={pearson:.3f}\nSpearman={spearman:.3f}", transform=axis.transAxes, va="top")
        axis.set_xscale("log"); axis.set_yscale("log"); axis.set_xlim(lower, upper); axis.set_ylim(lower, upper); axis.set_box_aspect(1)
        axis.set_xlabel(f"LLPR uncertainty ({panel.unit})"); axis.set_ylabel(f"Absolute residual ({panel.unit})"); axis.grid(False)
        stem = CARNET_FIGURE_STEMS[path]
        figure.savefig(staging / f"{stem}.png", dpi=dpi, facecolor="white")
        figure.savefig(staging / f"{stem}.pdf", dpi=dpi, facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    finally:
        plt.close(figure)


def render_carnet_density(panels: Mapping[tuple[str, str], Any], staging: Any, *, dpi: int) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[float, float]]]:
    limits = carnet_shared_limits(panels, margin=float(CARNET_DENSITY_CONFIG["log_margin"]))
    for path in CARNET_SELECTED:
        _render_panel(panels[path], path, limits[path], staging, dpi)
    return carnet_statistics(panels, limits), limits
