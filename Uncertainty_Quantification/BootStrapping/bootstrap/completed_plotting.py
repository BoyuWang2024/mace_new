"""Deterministic STD-versus-residual plots for completed MACE ensembles."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

from .errors import HardFailure
from .inference_config import CompletedInferenceConfig


def symmetric_voigt(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape[-2:] != (3, 3):
        raise HardFailure("stress array must end with 3 x 3")
    symmetric = 0.5 * (array + np.swapaxes(array, -1, -2))
    return symmetric[..., [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]


def _load(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise HardFailure(f"could not load plot input {path}: {error}") from error


def dataset_panels(
    *,
    targets: dict[str, np.ndarray],
    ensemble: dict[str, np.ndarray],
    uncertainty: dict[str, np.ndarray],
    domains: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    counts = np.asarray(targets["num_atoms"], dtype=float)
    panels = {
        "energy": (
            np.asarray(uncertainty["energy_per_atom_std"], dtype=float),
            np.abs(np.asarray(ensemble["energy"], dtype=float) / counts - targets["energy"] / counts),
        ),
        "force": (
            np.asarray(uncertainty["force_std"], dtype=float),
            np.abs(np.asarray(ensemble["forces"], dtype=float) - targets["forces"]),
        ),
    }
    if "stress" in domains:
        stress_std = np.asarray(uncertainty["stress_std"], dtype=float)
        if stress_std.shape[-2:] == (3, 3):
            stress_std = symmetric_voigt(stress_std)
        panels["stress"] = (
            stress_std,
            np.abs(symmetric_voigt(ensemble["stress"]) - symmetric_voigt(targets["stress"])),
        )
    return panels


def _artifacts(
    config: CompletedInferenceConfig, dataset_name: str, inference_root: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if dataset_name == "matpes_test":
        run = config.source.run
        targets = _load(run / "predictions" / "test" / "targets.npz")
        ensemble = _load(run / "ensemble" / "test" / "raw.npz")
        uncertainty = _load(run / "uncertainty" / "test" / "raw.npz")
        stress_members = np.stack(
            [
                symmetric_voigt(
                    _load(
                        run
                        / "predictions"
                        / "test"
                        / "members"
                        / f"member_{index:03d}"
                        / "raw.npz"
                    )["stress"]
                )
                for index in range(config.source.member_count)
            ],
            axis=0,
        )
        uncertainty["stress_std"] = np.std(stress_members, axis=0, ddof=1)
        return targets, ensemble, uncertainty
    root = inference_root / dataset_name / "analysis"
    return (
        _load(root / "targets.npz"),
        _load(root / "ensemble.npz"),
        _load(root / "uncertainty.npz"),
    )


def _filtered(uncertainty: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray(uncertainty, dtype=float).reshape(-1)
    y = np.asarray(residual, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise HardFailure("plot uncertainty and residual shapes differ")
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(mask.sum()) < 2:
        raise HardFailure("plot requires at least two positive finite pairs")
    return x[mask], y[mask], int(x.size - mask.sum())


def _contour_levels(grid: np.ndarray, masses: tuple[float, ...]) -> tuple[float, ...]:
    descending = np.sort(grid.reshape(-1))[::-1]
    cumulative = np.cumsum(descending)
    values = []
    for mass in masses:
        index = min(int(np.searchsorted(cumulative, mass)), descending.size - 1)
        values.append(float(descending[index]))
    return tuple(sorted(set(values)))


def _render_panel(
    name: str,
    dataset_name: str,
    uncertainty: np.ndarray,
    residual: np.ndarray,
    config: CompletedInferenceConfig,
    destination: Path,
) -> dict[str, object]:
    x, y, excluded = _filtered(uncertainty, residual)
    lx = np.log10(x)
    ly = np.log10(y)
    low = float(min(lx.min(), ly.min()))
    high = float(max(lx.max(), ly.max()))
    margin = max((high - low) * 0.04, 0.08)
    low -= margin
    high += margin
    histogram, x_edges, y_edges = np.histogram2d(
        lx,
        ly,
        bins=config.plot.grid_size,
        range=((low, high), (low, high)),
    )
    density = gaussian_filter(histogram, sigma=config.plot.gaussian_sigma, mode="nearest")
    density /= density.sum()
    levels = _contour_levels(density, config.plot.contour_masses)
    if len(levels) < 2:
        levels = tuple(np.linspace(float(density[density > 0].min()), float(density.max()), 3))
    rng = np.random.default_rng(config.plot.scatter_seed)
    if x.size > config.plot.scatter_max_points:
        indices = np.sort(rng.choice(x.size, config.plot.scatter_max_points, replace=False))
    else:
        indices = np.arange(x.size)
    units = {"energy": "eV/atom", "force": "eV/Angstrom", "stress": "eV/Angstrom^3"}
    colors = {"energy": "#E15759", "force": "#4E79A7", "stress": "#59A14F"}
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        figure, axis = plt.subplots(figsize=config.plot.figure_size)
        lower, upper = 10.0**low, 10.0**high
        diagonal = np.geomspace(lower, upper, 512)
        axis.fill_between(diagonal, lower, diagonal, color="#D9D9D9", alpha=0.45)
        axis.scatter(x[indices], y[indices], s=7, alpha=0.28, color=colors[name], edgecolors="none", rasterized=True)
        centers_x = 10.0 ** ((x_edges[:-1] + x_edges[1:]) * 0.5)
        centers_y = 10.0 ** ((y_edges[:-1] + y_edges[1:]) * 0.5)
        axis.contour(centers_x, centers_y, density.T, levels=levels, colors=colors[name], linewidths=0.9)
        axis.plot(diagonal, diagonal, color="black", linewidth=1.1)
        axis.set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper))
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"Bootstrap STD ({units[name]})")
        axis.set_ylabel(f"Absolute residual ({units[name]})")
        axis.set_title(f"MACE BootStrapping | {dataset_name} | {name}")
        axis.text(
            0.03,
            0.97,
            f"Spearman(log) = {spearmanr(lx, ly).statistic:.4f}\nPearson(log10) = {pearsonr(lx, ly).statistic:.4f}\nvalid = {x.size:,}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#AAAAAA", "alpha": 0.92},
        )
        figure.tight_layout()
        stem = destination / f"mace_bootstrap_{name}_uncertainty_vs_residual"
        figure.savefig(stem.with_suffix(".png"), dpi=config.plot.dpi)
        figure.savefig(stem.with_suffix(".pdf"), dpi=config.plot.dpi, metadata={"CreationDate": None, "ModDate": None})
        plt.close(figure)
    return {
        "dataset": dataset_name,
        "target": name,
        "original_count": int(np.asarray(uncertainty).size),
        "valid_count": int(x.size),
        "excluded_count": excluded,
        "scatter_count": int(indices.size),
        "spearman_log": float(spearmanr(lx, ly).statistic),
        "pearson_log10": float(pearsonr(lx, ly).statistic),
    }


def render_dataset_plots(
    config: CompletedInferenceConfig,
    dataset_name: str,
    *,
    inference_root: str | Path,
) -> Path:
    dataset = config.datasets[dataset_name]
    targets, ensemble, uncertainty = _artifacts(config, dataset_name, Path(inference_root))
    panels = dataset_panels(
        targets=targets, ensemble=ensemble, uncertainty=uncertainty, domains=dataset.domains
    )
    destination = config.plot.output_root / dataset_name
    destination.mkdir(parents=True, exist_ok=True)
    rows = [
        _render_panel(name, dataset_name, values[0], values[1], config, destination)
        for name, values in panels.items()
    ]
    (destination / "plot_manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    with (destination / "plot_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


__all__ = ["dataset_panels", "render_dataset_plots", "symmetric_voigt"]
