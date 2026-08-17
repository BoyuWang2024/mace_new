"""Cross-experiment comparison figures for canonical FGE results."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .errors import HardFailure
from .plot_data import BRANCHES, PlotRun
from .plot_single import FigureRecord, _save_pair, _style_axis
from .plot_style import PlotConfig


COLORS = {"equal_weight": "#4e79a7", "validation_weighted": "#f28e2b"}


def _finite_float(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HardFailure(f"invalid numeric value for {context}") from exc
    if not np.isfinite(result):
        raise HardFailure(f"NaN or Inf in {context}")
    return result


def _correlation(run: PlotRun, branch: str, metric: str, coefficient: str) -> float:
    rows = [row for row in run.branches[branch].correlations if row.get("metric") == metric]
    if len(rows) != 1:
        raise HardFailure(f"missing unique correlation row for {metric}")
    if rows[0].get(coefficient) in (None, ""):
        warnings.warn(
            f"undefined correlation omitted: {run.name} {branch} {metric} {coefficient}",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    return _finite_float(rows[0].get(coefficient), f"{metric} {coefficient}")


def _comparison_panels(runs: Mapping[str, PlotRun]) -> list[tuple[str, str, str]]:
    modes = {run.has_stress for run in runs.values()}
    if len(modes) != 1:
        raise HardFailure("all four experiments must expose the same observables")
    panels = [
        ("energy_per_atom", "Energy", "eV/atom"),
        ("force_component", "Force", "eV/Angstrom"),
    ]
    if modes == {True}:
        panels.append(("stress_component", "Stress", "eV/Angstrom^3"))
    return panels


def _rmse_figure(runs: Mapping[str, PlotRun]):
    labels = tuple(runs)
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    panels = _comparison_panels(runs)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.1 * len(panels), 4.2),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, (metric, observable, unit) in zip(axes, panels, strict=True):
        for offset, branch in zip((-width / 2, width / 2), BRANCHES, strict=True):
            values = [
                _finite_float(
                    runs[label].branches[branch].metrics[metric]["rmse"],
                    f"{label} {metric} RMSE",
                )
                for label in labels
            ]
            axis.bar(
                x + offset,
                values,
                width,
                label=branch.replace("_", " ").title(),
                color=COLORS[branch],
            )
        axis.set_title(f"{observable} RMSE")
        axis.set_ylabel(unit)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        _style_axis(axis)
    axes[0].legend(frameon=False, fontsize=8)
    return figure


def _correlation_figure(runs: Mapping[str, PlotRun]):
    labels = tuple(runs)
    x = np.arange(len(labels), dtype=float)
    panels = _comparison_panels(runs)
    figure, axes = plt.subplots(
        len(panels),
        2,
        figsize=(10.2, 3.7 * len(panels)),
        constrained_layout=True,
        sharex=True,
        squeeze=False,
    )
    for row, (metric, observable, _unit) in enumerate(panels):
        uncertainty_metric = f"{metric}_std"
        for column, coefficient in enumerate(("pearson", "spearman")):
            axis = axes[row, column]
            for branch in BRANCHES:
                values = [
                    _correlation(
                        runs[label],
                        branch,
                        uncertainty_metric,
                        coefficient,
                    )
                    for label in labels
                ]
                axis.plot(
                    x,
                    values,
                    marker="o",
                    linewidth=1.5,
                    color=COLORS[branch],
                    label=branch.replace("_", " ").title(),
                )
            axis.set_title(f"{observable} - {coefficient.title()}")
            axis.set_ylim(-1.05, 1.05)
            axis.set_xticks(x, labels, rotation=18, ha="right")
            _style_axis(axis)
    axes[0, 0].legend(frameon=False, fontsize=8)
    return figure


def _risk_figure(runs: Mapping[str, PlotRun]):
    panels = _comparison_panels(runs)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.1 * len(panels), 4.2),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    linestyles = {"equal_weight": "-", "validation_weighted": "--"}
    palette = plt.get_cmap("tab10")
    for axis, (metric, observable, _unit) in zip(axes, panels, strict=True):
        uncertainty_metric = f"{metric}_std"
        for run_index, (label, run) in enumerate(runs.items()):
            for branch in BRANCHES:
                rows = [
                    row
                    for row in run.branches[branch].risk_coverage
                    if row.get("metric") == uncertainty_metric
                ]
                if not rows:
                    raise HardFailure(
                        f"risk-coverage rows are missing for "
                        f"{label} {uncertainty_metric}"
                    )
                points = sorted(
                    (
                        _finite_float(row.get("coverage"), f"{label} coverage"),
                        _finite_float(row.get("risk"), f"{label} risk"),
                    )
                    for row in rows
                )
                coverage, risk = zip(*points, strict=True)
                axis.plot(
                    coverage,
                    risk,
                    linestyle=linestyles[branch],
                    color=palette(run_index),
                    linewidth=1.5,
                    label=f"{label} - {branch.replace('_', ' ')}",
                )
        axis.set_title(observable)
        axis.set_xlabel("Coverage")
        axis.set_ylabel("RMSE of retained samples")
        axis.set_xlim(0.0, 1.02)
        _style_axis(axis)
    axes[-1].legend(
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )
    return figure


def render_comparisons(
    runs: Mapping[str, PlotRun], output_root: Path, config: PlotConfig
) -> tuple[FigureRecord, ...]:
    records = []
    destination = Path(output_root) / "comparison"
    for logical_name, figure in (
        ("rmse_comparison", _rmse_figure(runs)),
        ("correlation_comparison", _correlation_figure(runs)),
        ("risk_coverage_comparison", _risk_figure(runs)),
    ):
        png, pdf = _save_pair(figure, destination / logical_name, config)
        plt.close(figure)
        records.append(FigureRecord("comparison", "all", logical_name, png, pdf))
    return tuple(records)
