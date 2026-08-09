"""Publication rendering for one canonical FGE experiment."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

from .errors import HardFailure
from .plot_data import PlotBranch, PlotRun
from .plot_density import analyze_log_panel, deterministic_indices
from .plot_style import ORANGE, PlotConfig


@dataclass(frozen=True)
class FigureRecord:
    experiment: str
    branch: str
    logical_name: str
    png: Path
    pdf: Path
    excluded: dict[str, int] | None = None


def _save_pair(figure: Figure, stem: Path, config: PlotConfig) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = (stem.with_suffix(".png"), stem.with_suffix(".pdf"))
    temporary: list[Path] = []
    try:
        for path in paths:
            temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            figure.savefig(
                temp,
                format=path.suffix[1:],
                dpi=config.dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            if not temp.is_file() or temp.stat().st_size == 0:
                raise HardFailure(f"failed to write figure: {path}")
            temporary.append(temp)
        for temp, path in zip(temporary, paths, strict=True):
            os.replace(temp, path)
    except Exception as exc:
        for temp in temporary:
            temp.unlink(missing_ok=True)
        if isinstance(exc, HardFailure):
            raise
        raise HardFailure(f"failed to publish figure pair: {stem}") from exc
    return paths


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def _parity_figure(reference, prediction, title: str, unit: str, config: PlotConfig) -> Figure:
    x = reference.numpy()
    y = prediction.numpy()
    low = float(min(x.min(), y.min()))
    high = float(max(x.max(), y.max()))
    padding = max((high - low) * 0.04, 1.0e-12)
    low -= padding
    high += padding
    density, x_edges, y_edges = np.histogram2d(x, y, bins=config.grid_size)
    density = gaussian_filter(density, sigma=config.gaussian_sigma)
    positive = density[density > 0]
    levels = np.unique(np.quantile(positive, (0.20, 0.45, 0.70, 0.88))) if positive.size else ()
    indices = deterministic_indices(len(x), config.scatter_max_points, config.random_seed).numpy()

    figure, axis = plt.subplots(figsize=(5.1, 5.0), constrained_layout=True)
    axis.scatter(x[indices], y[indices], s=3.0, alpha=0.08, color=ORANGE, linewidths=0, rasterized=True)
    if len(levels):
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        axis.contour(x_centers, y_centers, density.T, levels=levels, colors="#8c4b16", linewidths=0.8)
    axis.plot([low, high], [low, high], "k--", linewidth=1.0)
    axis.set(xlim=(low, high), ylim=(low, high), aspect="equal", title=title)
    axis.set_xlabel(f"Reference {unit}")
    axis.set_ylabel(f"FGE prediction {unit}")
    _style_axis(axis)
    return figure


def _uncertainty_figure(uncertainty, residual, title: str, unit: str, config: PlotConfig):
    analysis = analyze_log_panel(uncertainty, residual, config)
    x = torch_values = 10.0 ** analysis.log_uncertainty.numpy()
    y = 10.0 ** analysis.log_residual.numpy()
    indices = analysis.sample_indices.numpy()
    log_low = float(min(analysis.log_uncertainty.min(), analysis.log_residual.min()))
    log_high = float(max(analysis.log_uncertainty.max(), analysis.log_residual.max()))
    padding = max((log_high - log_low) * 0.05, 0.05)
    low, high = 10.0 ** (log_low - padding), 10.0 ** (log_high + padding)

    figure, axis = plt.subplots(figsize=(5.1, 5.0), constrained_layout=True)
    diagonal = np.geomspace(low, high, 256)
    axis.fill_between(diagonal, low, diagonal, color="#d9d9d9", alpha=0.55, zorder=0)
    axis.scatter(
        x[indices], y[indices], s=config.scatter_size, alpha=config.scatter_alpha,
        color=ORANGE, linewidths=0, rasterized=True, zorder=2,
    )
    if analysis.contour_levels:
        x_centers = 10.0 ** ((analysis.uncertainty_edges[:-1] + analysis.uncertainty_edges[1:]) / 2)
        y_centers = 10.0 ** ((analysis.residual_edges[:-1] + analysis.residual_edges[1:]) / 2)
        axis.contour(
            x_centers, y_centers, analysis.density.T, levels=analysis.contour_levels,
            colors="#7f3f00", linewidths=0.8, zorder=3,
        )
    axis.plot(diagonal, diagonal, "k--", linewidth=1.0, zorder=4)
    axis.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high), aspect="equal")
    axis.set_title(title)
    axis.set_xlabel(f"Predicted uncertainty {unit}")
    axis.set_ylabel(f"Absolute residual {unit}")
    axis.text(
        0.04, 0.96,
        f"Spearman $\\rho$ = {analysis.spearman:.3f}\nPearson(log$_{{10}}$) = {analysis.pearson_log10:.3f}",
        transform=axis.transAxes, va="top", ha="left", fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#bdbdbd"},
    )
    _style_axis(axis)
    return figure, analysis.excluded


def _risk_figure(branch: PlotBranch, title: str) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), constrained_layout=True)
    for axis, metric, label in zip(
        axes,
        ("energy_per_atom_std", "force_component_std"),
        ("Energy", "Force"),
        strict=True,
    ):
        rows = [row for row in branch.risk_coverage if row.get("metric") == metric]
        if not rows:
            raise HardFailure(f"risk-coverage rows are missing for {metric}")
        try:
            points = sorted((float(row["coverage"]), float(row["risk"])) for row in rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise HardFailure(f"invalid risk-coverage values for {metric}") from exc
        if not np.isfinite(points).all():
            raise HardFailure(f"risk-coverage contains NaN or Inf for {metric}")
        coverage, risk = zip(*points, strict=True)
        axis.plot(coverage, risk, color=ORANGE, marker="o", markersize=3.5, linewidth=1.6)
        axis.set_title(label)
        axis.set_xlabel("Coverage")
        axis.set_ylabel("RMSE of retained samples")
        axis.set_xlim(0.0, 1.02)
        _style_axis(axis)
    figure.suptitle(title)
    return figure


def render_single_run(
    run: PlotRun, output_root: Path, config: PlotConfig = PlotConfig()
) -> tuple[FigureRecord, ...]:
    records: list[FigureRecord] = []
    for branch_name, branch in run.branches.items():
        destination = Path(output_root) / run.name / branch_name
        title_suffix = branch_name.replace("_", " ").title()
        specifications = (
            ("energy_parity", _parity_figure(branch.energy_reference, branch.energy_prediction, f"Energy parity — {title_suffix}", "(eV/atom)", config), None),
            ("force_parity", _parity_figure(branch.force_reference, branch.force_prediction, f"Force parity — {title_suffix}", "(eV/Å)", config), None),
        )
        for logical_name, figure, excluded in specifications:
            png, pdf = _save_pair(figure, destination / logical_name, config)
            plt.close(figure)
            records.append(FigureRecord(run.name, branch_name, logical_name, png, pdf, excluded))

        for logical_name, values in (
            ("energy_uncertainty_residual", (branch.energy_uncertainty, branch.energy_residual, "Energy uncertainty vs residual", "(eV/atom)")),
            ("force_uncertainty_residual", (branch.force_uncertainty, branch.force_residual, "Force uncertainty vs residual", "(eV/Å)")),
        ):
            figure, excluded = _uncertainty_figure(*values, config)
            figure.axes[0].set_title(f"{values[2]} — {title_suffix}")
            png, pdf = _save_pair(figure, destination / logical_name, config)
            plt.close(figure)
            records.append(FigureRecord(run.name, branch_name, logical_name, png, pdf, excluded))

        figure = _risk_figure(branch, f"Risk–coverage — {title_suffix}")
        png, pdf = _save_pair(figure, destination / "risk_coverage", config)
        plt.close(figure)
        records.append(FigureRecord(run.name, branch_name, "risk_coverage", png, pdf))
    return tuple(records)
