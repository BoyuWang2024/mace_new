"""Independent numerical contract for MACE LLPR density publication plots."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr


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
    "font_family": "DejaVu Sans",
    "title_font_size": 26.0,
    "axis_label_font_size": 22.0,
    "tick_label_font_size": 18.0,
    "annotation_font_size": 16.0,
    "line_width": 1.5,
    "spine_width": 1.5,
    "constrained_layout": True,
    "titles": {
        "he/energy": "He energy",
        "hf/forces": "Hf forces",
        "hef/energy": "Hef energy",
        "hef/forces": "Hef forces",
    },
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
    "correlation_status",
    "axis_min",
    "axis_max",
)


@dataclass(frozen=True, slots=True)
class DensityPanel:
    """The only two numeric arrays needed by one density panel."""

    uncertainty: np.ndarray
    absolute_residual: np.ndarray
    target: str
    unit: str


@dataclass(frozen=True, slots=True)
class _PanelAnalysis:
    panel: Any
    path: tuple[str, str]
    log_mask: np.ndarray
    candidate_indices: np.ndarray
    statistics: dict[str, Any]


def deterministic_sample_indices(total: int, maximum: int, seed: int) -> np.ndarray:
    """Return reproducible duplicate-free sample indices."""
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (total, maximum, seed)
    ):
        raise ValueError("sampling arguments must be integers")
    if total < 0 or maximum <= 0:
        raise ValueError("sampling total must be non-negative and maximum positive")
    if total <= maximum:
        return np.arange(total, dtype=np.int64)
    return np.random.default_rng(seed).choice(total, size=maximum, replace=False)


def _panel_arrays(panel: Any) -> tuple[np.ndarray, np.ndarray]:
    uncertainty = np.asarray(panel.uncertainty, dtype=np.float64)
    residual = np.asarray(panel.absolute_residual, dtype=np.float64)
    if uncertainty.ndim != 1 or residual.ndim != 1 or uncertainty.shape != residual.shape:
        raise ValueError("density panel arrays must be aligned one-dimensional arrays")
    return uncertainty, residual


def _log_mask(panel: Any) -> np.ndarray:
    uncertainty, residual = _panel_arrays(panel)
    return (
        np.isfinite(uncertainty)
        & np.isfinite(residual)
        & (uncertainty > 0.0)
        & (residual > 0.0)
    )


def _pair_limits(
    panels: Mapping[tuple[str, str], Any],
    paths: tuple[tuple[str, str], ...],
    margin: float,
) -> tuple[float, float]:
    minimum = math.inf
    maximum = -math.inf
    for path in paths:
        panel = panels[path]
        uncertainty, residual = _panel_arrays(panel)
        valid = _log_mask(panel)
        if np.any(valid):
            minimum = min(
                minimum,
                float(np.min(uncertainty, where=valid, initial=math.inf)),
                float(np.min(residual, where=valid, initial=math.inf)),
            )
            maximum = max(
                maximum,
                float(np.max(uncertainty, where=valid, initial=-math.inf)),
                float(np.max(residual, where=valid, initial=-math.inf)),
            )
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        return (1.0e-12, 1.0)
    low = math.log10(minimum)
    high = math.log10(maximum)
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


def _correlations(
    panel: Any, mask: np.ndarray
) -> tuple[float | None, float | None, int, int, str]:
    count = int(np.count_nonzero(mask))
    if count < 2:
        return (None, None, count, 1, "undefined_insufficient_rows")
    uncertainty, residual = _panel_arrays(panel)
    x_values = np.log10(uncertainty[mask])
    y_values = np.log10(residual[mask])
    if np.ptp(x_values) == 0.0 or np.ptp(y_values) == 0.0:
        return (None, None, count, 1, "undefined_constant")
    pearson = float(pearsonr(x_values, y_values).statistic)
    spearman = float(spearmanr(x_values, y_values).statistic)
    if not (math.isfinite(pearson) and math.isfinite(spearman)):
        return (None, None, count, 1, "undefined_nonfinite")
    return (pearson, spearman, count, 0, "ok")


def _analyze_panel(
    panel: Any,
    path: tuple[str, str],
    limits: tuple[float, float],
) -> _PanelAnalysis:
    uncertainty, residual = _panel_arrays(panel)
    finite = np.isfinite(uncertainty) & np.isfinite(residual)
    metric = finite & (uncertainty > 0.0)
    log_mask = metric & (residual > 0.0)
    pearson, spearman, correlation_rows, degenerate, status = _correlations(
        panel, log_mask
    )
    lower, upper = limits
    row = {
        "variant": path[0],
        "target": path[1],
        "unit": str(panel.unit),
        "rows": int(uncertainty.size),
        "finite_rows": int(np.count_nonzero(finite)),
        "nonfinite_rows": int(np.count_nonzero(~finite)),
        "nonpositive_std_rows": int(
            np.count_nonzero(finite & (uncertainty <= 0.0))
        ),
        "zero_absolute_residual_rows": int(
            np.count_nonzero(finite & (residual == 0.0))
        ),
        "metric_rows": int(np.count_nonzero(metric)),
        "log_plot_rows": int(np.count_nonzero(log_mask)),
        "excluded_from_log_rows": int(
            uncertainty.size - np.count_nonzero(log_mask)
        ),
        "pearson_log": pearson,
        "spearman_log": spearman,
        "correlation_rows": correlation_rows,
        "correlation_degenerate": degenerate,
        "correlation_status": status,
        "axis_min": float(lower),
        "axis_max": float(upper),
    }
    return _PanelAnalysis(
        panel=panel,
        path=path,
        log_mask=log_mask,
        candidate_indices=np.flatnonzero(log_mask),
        statistics=row,
    )


def carnet_statistics(
    panels: Mapping[tuple[str, str], Any],
    limits: Mapping[tuple[str, str], tuple[float, float]],
) -> list[dict[str, Any]]:
    return [
        _analyze_panel(panels[path], path, limits[path]).statistics
        for path in CARNET_SELECTED
    ]


def write_carnet_statistics(path: Any, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(CARNET_STATISTICS_FIELDS),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def _render_panel(
    analysis: _PanelAnalysis,
    staging: Any,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    config = CARNET_DENSITY_CONFIG
    rc = {
        "font.family": config["font_family"],
        "font.size": config["tick_label_font_size"],
        "axes.titlesize": config["title_font_size"],
        "axes.labelsize": config["axis_label_font_size"],
        "xtick.labelsize": config["tick_label_font_size"],
        "ytick.labelsize": config["tick_label_font_size"],
        "savefig.bbox": None,
        "pdf.compression": 6,
    }
    with plt.rc_context(rc):
        figure, axis = plt.subplots(
            figsize=config["figure_size"],
            constrained_layout=bool(config["constrained_layout"]),
        )
        try:
            panel = analysis.panel
            path = analysis.path
            lower = float(analysis.statistics["axis_min"])
            upper = float(analysis.statistics["axis_max"])
            diagonal = np.geomspace(lower, upper, 256)
            axis.fill_between(diagonal, lower, diagonal, color="0.90")
            sample = analysis.candidate_indices[
                deterministic_sample_indices(
                    int(analysis.candidate_indices.size),
                    int(config["scatter_max_points"]),
                    int(config["random_seed"]),
                )
            ]
            uncertainty, residual = _panel_arrays(panel)
            axis.scatter(
                uncertainty[sample],
                residual[sample],
                color=config["color"],
                s=config["scatter_size"],
                alpha=config["scatter_alpha"],
                edgecolors="none",
                rasterized=True,
            )
            edges = np.linspace(
                math.log10(lower),
                math.log10(upper),
                int(config["grid_size"]) + 1,
            )
            if analysis.candidate_indices.size:
                histogram, _, _ = np.histogram2d(
                    np.log10(uncertainty[analysis.log_mask]),
                    np.log10(residual[analysis.log_mask]),
                    bins=(edges, edges),
                )
                density = gaussian_filter(
                    histogram.T, sigma=float(config["gaussian_sigma"])
                )
                total = float(density.sum())
                if total > 0.0:
                    sorted_density = np.sort(density.ravel())[::-1]
                    cumulative = np.cumsum(sorted_density) / total
                    levels = np.unique(
                        [
                            sorted_density[
                                min(
                                    int(np.searchsorted(cumulative, mass)),
                                    sorted_density.size - 1,
                                )
                            ]
                            for mass in config["contour_masses"]
                        ]
                    )
                    levels = levels[levels > 0.0]
                    if levels.size:
                        centers = (edges[:-1] + edges[1:]) / 2.0
                        axis.contour(
                            10.0 ** centers,
                            10.0 ** centers,
                            density,
                            levels=levels,
                            colors=config["color"],
                            linewidths=config["line_width"],
                        )
            axis.plot(
                diagonal,
                diagonal,
                color="black",
                linestyle="--",
                linewidth=config["line_width"],
                label="1:1 reference",
            )
            pearson = analysis.statistics["pearson_log"]
            spearman = analysis.statistics["spearman_log"]
            pearson_text = "n/a" if pearson is None else f"{pearson:.3f}"
            spearman_text = "n/a" if spearman is None else f"{spearman:.3f}"
            axis.text(
                0.04,
                0.96,
                f"Pearson={pearson_text}\nSpearman={spearman_text}",
                transform=axis.transAxes,
                va="top",
                fontsize=config["annotation_font_size"],
            )
            axis.set_title(
                config["titles"][f"{path[0]}/{path[1]}"],
                fontsize=config["title_font_size"],
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_box_aspect(1)
            axis.set_xlabel(
                f"LLPR uncertainty ({panel.unit})",
                fontsize=config["axis_label_font_size"],
            )
            axis.set_ylabel(
                f"Absolute residual ({panel.unit})",
                fontsize=config["axis_label_font_size"],
            )
            axis.tick_params(
                axis="both",
                labelsize=config["tick_label_font_size"],
                width=config["line_width"],
            )
            for spine in axis.spines.values():
                spine.set_linewidth(config["spine_width"])
            axis.grid(False)
            stem = CARNET_FIGURE_STEMS[path]
            figure.savefig(
                staging / f"{stem}.png",
                dpi=dpi,
                facecolor="white",
            )
            figure.savefig(
                staging / f"{stem}.pdf",
                dpi=dpi,
                facecolor="white",
                metadata={"CreationDate": None, "ModDate": None},
            )
        finally:
            plt.close(figure)


def render_carnet_density(
    panels: Mapping[tuple[str, str], Any],
    staging: Any,
    *,
    dpi: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[float, float]]]:
    limits = carnet_shared_limits(
        panels, margin=float(CARNET_DENSITY_CONFIG["log_margin"])
    )
    analyses = [
        _analyze_panel(panels[path], path, limits[path])
        for path in CARNET_SELECTED
    ]
    for analysis in analyses:
        _render_panel(analysis, staging, dpi)
    return [analysis.statistics for analysis in analyses], limits
