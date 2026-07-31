"""Deterministic publication figures from validated LLPR result CSVs only."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import FORMULA_VERSION, SCHEMA_VERSION, sha256_file
from .validation import validate_publication_root


_VARIANTS = ("he", "hf", "hef")
_VARIANT_LABELS = {"he": r"$H_e$", "hf": r"$H_f$", "hef": r"$H_{ef}$"}
_VARIANT_COLORS = {"he": "#0072B2", "hf": "#D55E00", "hef": "#009E73"}
_ZERO_STRATEGY = (
    "exclude non-positive coordinates from logarithmic rendering only; "
    "retain every validated row in statistics"
)
_DEFAULT_DPI = 180

DEFAULT_SELECTED = (
    ("he", "energy"),
    ("hf", "forces"),
    ("hef", "energy"),
    ("hef", "forces"),
)

FIGURE_STEMS = (
    "selected_uncertainty_residual",
    "energy_comparison",
    "force_component_comparison",
    "force_structure_comparison",
    "reliability",
    "standardized_residual_cdf",
)

PLOTTING_STATISTICS_FIELDS = (
    "variant",
    "target",
    "data_level",
    "unit",
    "rows",
    "log_plot_rows",
    "zero_std_rows",
    "zero_absolute_residual_rows",
    "mae",
    "rmse",
    "mean_std",
    "median_std",
    "coverage_1sigma",
    "coverage_2sigma",
    "coverage_3sigma",
    "standardized_residual_rows",
    "mean_absolute_standardized_residual",
    "uncertainty_residual_correlation",
    "shared_axis_min",
    "shared_axis_max",
)

_INPUT_NAMES = (
    "manifest.json",
    "validation.json",
    *(f"{variant}/{filename}" for variant in _VARIANTS for filename in (
        "energy.csv",
        "force_components.csv",
        "force_structure.csv",
        "summary.json",
    )),
)


@dataclass(frozen=True)
class _PanelData:
    uncertainty: np.ndarray
    absolute_residual: np.ndarray
    signed_residual: np.ndarray
    mae_values: np.ndarray
    squared_error_values: np.ndarray
    target: str
    data_level: str
    unit: str


def _as_float(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].to_numpy(dtype=np.float64, copy=True)


def _load_plot_data(root: Path) -> dict[str, dict[str, _PanelData]]:
    result: dict[str, dict[str, _PanelData]] = {
        "energy": {},
        "force_component": {},
        "force_structure": {},
    }
    for variant in _VARIANTS:
        energy = pd.read_csv(root / variant / "energy.csv")
        energy_residual = _as_float(energy, "residual")
        result["energy"][variant] = _PanelData(
            uncertainty=_as_float(energy, "std"),
            absolute_residual=np.abs(energy_residual),
            signed_residual=energy_residual,
            mae_values=np.abs(energy_residual),
            squared_error_values=energy_residual**2,
            target="energy",
            data_level="energy",
            unit="eV/atom",
        )

        components = pd.read_csv(root / variant / "force_components.csv")
        force_residual = _as_float(components, "residual")
        result["force_component"][variant] = _PanelData(
            uncertainty=_as_float(components, "std"),
            absolute_residual=np.abs(force_residual),
            signed_residual=force_residual,
            mae_values=np.abs(force_residual),
            squared_error_values=force_residual**2,
            target="forces",
            data_level="force_component",
            unit="eV/?",
        )

        structures = pd.read_csv(root / variant / "force_structure.csv")
        structure_rmse = _as_float(structures, "rmse")
        result["force_structure"][variant] = _PanelData(
            uncertainty=np.sqrt(_as_float(structures, "mean_variance")),
            absolute_residual=structure_rmse,
            signed_residual=structure_rmse,
            mae_values=_as_float(structures, "mae"),
            squared_error_values=structure_rmse**2,
            target="forces",
            data_level="force_structure",
            unit="eV/?",
        )
    return result


def _shared_log_limits(panels: Mapping[str, _PanelData]) -> tuple[float, float]:
    positive: list[np.ndarray] = []
    for panel in panels.values():
        for values in (panel.uncertainty, panel.absolute_residual):
            selected = values[np.isfinite(values) & (values > 0.0)]
            if selected.size:
                positive.append(selected)
    if not positive:
        return (1.0e-12, 1.0)
    pooled = np.concatenate(positive)
    low_log = float(np.log10(np.min(pooled)))
    high_log = float(np.log10(np.max(pooled)))
    span = high_log - low_log
    padding = max(0.05 * span, 0.05)
    return (10.0 ** (low_log - padding), 10.0 ** (high_log + padding))


def _all_shared_limits(
    data: Mapping[str, Mapping[str, _PanelData]],
) -> dict[str, tuple[float, float]]:
    return {level: _shared_log_limits(data[level]) for level in data}


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _statistics_row(
    variant: str,
    panel: _PanelData,
    limits: tuple[float, float],
) -> dict[str, Any]:
    uncertainty = panel.uncertainty
    error = panel.absolute_residual
    log_mask = (uncertainty > 0.0) & (error > 0.0)
    standardized_mask = uncertainty > 0.0
    standardized = error[standardized_mask] / uncertainty[standardized_mask]
    return {
        "variant": variant,
        "target": panel.target,
        "data_level": panel.data_level,
        "unit": panel.unit,
        "rows": int(error.size),
        "log_plot_rows": int(np.count_nonzero(log_mask)),
        "zero_std_rows": int(np.count_nonzero(uncertainty == 0.0)),
        "zero_absolute_residual_rows": int(np.count_nonzero(error == 0.0)),
        "mae": float(np.mean(panel.mae_values)),
        "rmse": float(np.sqrt(np.mean(panel.squared_error_values))),
        "mean_std": float(np.mean(uncertainty)),
        "median_std": float(np.median(uncertainty)),
        "coverage_1sigma": float(np.mean(error <= uncertainty)),
        "coverage_2sigma": float(np.mean(error <= 2.0 * uncertainty)),
        "coverage_3sigma": float(np.mean(error <= 3.0 * uncertainty)),
        "standardized_residual_rows": int(standardized.size),
        "mean_absolute_standardized_residual": (
            float(np.mean(standardized)) if standardized.size else None
        ),
        "uncertainty_residual_correlation": _correlation(uncertainty, error),
        "shared_axis_min": limits[0],
        "shared_axis_max": limits[1],
    }


def _write_statistics(
    path: Path,
    data: Mapping[str, Mapping[str, _PanelData]],
    shared_limits: Mapping[str, tuple[float, float]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(PLOTTING_STATISTICS_FIELDS),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for level in ("energy", "force_component", "force_structure"):
            for variant in _VARIANTS:
                writer.writerow(
                    _statistics_row(variant, data[level][variant], shared_limits[level])
                )


def _style_log_axis(
    axis: Any,
    panel: _PanelData,
    limits: tuple[float, float],
    *,
    variant: str,
    title: str,
) -> None:
    low, high = limits
    diagonal = np.geomspace(low, high, 256)
    axis.fill_between(diagonal, low, diagonal, color="0.88", zorder=0)
    visible = (panel.uncertainty > 0.0) & (panel.absolute_residual > 0.0)
    axis.scatter(
        panel.uncertainty[visible],
        panel.absolute_residual[visible],
        s=8.0,
        alpha=0.42,
        color=_VARIANT_COLORS[variant],
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    axis.plot(diagonal, diagonal, color="black", linestyle="--", linewidth=1.0, zorder=3)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_box_aspect(1)
    axis.set_title(title)
    axis.set_xlabel(f"LLPR standard deviation ({panel.unit})")
    y_name = "Structure force RMSE" if panel.data_level == "force_structure" else "Absolute residual"
    axis.set_ylabel(f"{y_name} ({panel.unit})")
    axis.grid(False)
    axis.text(
        0.04,
        0.96,
        f"N={panel.absolute_residual.size:,}\nlog N={np.count_nonzero(visible):,}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.0},
    )


def _save_figure(figure: Any, staging: Path, stem: str, dpi: int) -> None:
    figure.savefig(
        staging / f"{stem}.png",
        format="png",
        dpi=dpi,
        facecolor="white",
        metadata={"Software": "MACE LLPR"},
    )
    figure.savefig(
        staging / f"{stem}.pdf",
        format="pdf",
        dpi=dpi,
        facecolor="white",
        metadata={
            "Creator": "MACE LLPR",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )


def _selected_level(target: str) -> str:
    return "energy" if target == "energy" else "force_component"


def _render_selected(
    data: Mapping[str, Mapping[str, _PanelData]],
    limits: Mapping[str, tuple[float, float]],
    selected: Sequence[tuple[str, str]],
    staging: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), layout="constrained")
    try:
        for axis, (variant, target) in zip(axes.reshape(-1), selected):
            level = _selected_level(target)
            panel = data[level][variant]
            target_label = "Energy" if target == "energy" else "Force component"
            _style_log_axis(
                axis,
                panel,
                limits[level],
                variant=variant,
                title=f"{_VARIANT_LABELS[variant]} ? {target_label}",
            )
        figure.suptitle("Selected LLPR uncertainty and residual paths")
        _save_figure(figure, staging, "selected_uncertainty_residual", dpi)
    finally:
        plt.close(figure)


def _render_comparison(
    data: Mapping[str, Mapping[str, _PanelData]],
    limits: Mapping[str, tuple[float, float]],
    level: str,
    stem: str,
    heading: str,
    staging: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), layout="constrained")
    try:
        for axis, variant in zip(axes, _VARIANTS):
            _style_log_axis(
                axis,
                data[level][variant],
                limits[level],
                variant=variant,
                title=_VARIANT_LABELS[variant],
            )
        figure.suptitle(heading)
        _save_figure(figure, staging, stem, dpi)
    finally:
        plt.close(figure)


def _reliability_points(panel: _PanelData) -> tuple[np.ndarray, np.ndarray]:
    valid = panel.uncertainty > 0.0
    std = panel.uncertainty[valid]
    residual = panel.signed_residual[valid]
    if not std.size:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    order = np.argsort(std, kind="stable")
    bins = np.array_split(order, min(10, order.size))
    return (
        np.asarray([np.sqrt(np.mean(std[index] ** 2)) for index in bins]),
        np.asarray([np.sqrt(np.mean(residual[index] ** 2)) for index in bins]),
    )


def _diagnostic_limits(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    values = [
        array[array > 0.0]
        for pair in pairs
        for array in pair
        if np.any(array > 0.0)
    ]
    if not values:
        return (1.0e-12, 1.0)
    pooled = np.concatenate(values)
    low_log = float(np.log10(np.min(pooled)))
    high_log = float(np.log10(np.max(pooled)))
    padding = max((high_log - low_log) * 0.05, 0.05)
    return (10.0 ** (low_log - padding), 10.0 ** (high_log + padding))


def _render_reliability(
    data: Mapping[str, Mapping[str, _PanelData]], staging: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), layout="constrained")
    try:
        for axis, level, heading in zip(
            axes,
            ("energy", "force_component"),
            ("Energy", "Force component"),
        ):
            pairs = [_reliability_points(data[level][variant]) for variant in _VARIANTS]
            low, high = _diagnostic_limits(pairs)
            diagonal = np.geomspace(low, high, 256)
            for variant, (x_values, y_values) in zip(_VARIANTS, pairs):
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    markersize=4,
                    linewidth=1.2,
                    color=_VARIANT_COLORS[variant],
                    label=_VARIANT_LABELS[variant],
                )
            axis.plot(diagonal, diagonal, color="black", linestyle="--", linewidth=1.0)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
            axis.set_box_aspect(1)
            axis.set_title(f"{heading} reliability")
            axis.set_xlabel(f"Predicted RMS std ({data[level]['he'].unit})")
            axis.set_ylabel(f"Empirical RMSE ({data[level]['he'].unit})")
            axis.legend(frameon=False, fontsize=8)
            axis.grid(False)
        figure.suptitle("LLPR reliability")
        _save_figure(figure, staging, "reliability", dpi)
    finally:
        plt.close(figure)


def _cdf_grid(panels: Mapping[str, _PanelData]) -> np.ndarray:
    positive: list[np.ndarray] = []
    for panel in panels.values():
        valid = panel.uncertainty > 0.0
        z = np.abs(panel.signed_residual[valid] / panel.uncertainty[valid])
        if np.any(z > 0.0):
            positive.append(z[z > 0.0])
    if not positive:
        return np.geomspace(1.0e-2, 1.0e2, 256)
    pooled = np.concatenate(positive)
    low_log = float(np.log10(np.min(pooled)))
    high_log = float(np.log10(np.max(pooled)))
    padding = max((high_log - low_log) * 0.05, 0.05)
    base = np.geomspace(10.0 ** (low_log - padding), 10.0 ** (high_log + padding), 256)
    return np.unique(np.concatenate((base, np.asarray([1.0, 2.0, 3.0]))))


def _empirical_cdf(panel: _PanelData, grid: np.ndarray) -> np.ndarray:
    valid = panel.uncertainty > 0.0
    if not np.any(valid):
        return np.zeros_like(grid)
    z = np.sort(np.abs(panel.signed_residual[valid] / panel.uncertainty[valid]))
    return np.searchsorted(z, grid, side="right") / z.size


def _render_cdf(
    data: Mapping[str, Mapping[str, _PanelData]], staging: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), layout="constrained")
    try:
        for axis, level, heading in zip(
            axes,
            ("energy", "force_component"),
            ("Energy", "Force component"),
        ):
            grid = _cdf_grid(data[level])
            half_normal = np.asarray([math.erf(value / math.sqrt(2.0)) for value in grid])
            axis.plot(
                grid,
                half_normal,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="Half-normal reference",
            )
            for variant in _VARIANTS:
                axis.plot(
                    grid,
                    _empirical_cdf(data[level][variant], grid),
                    color=_VARIANT_COLORS[variant],
                    linewidth=1.2,
                    label=_VARIANT_LABELS[variant],
                )
            for threshold in (1.0, 2.0, 3.0):
                axis.axvline(threshold, color="0.72", linestyle=":", linewidth=0.8)
            axis.set_xscale("log")
            axis.set_ylim(0.0, 1.01)
            axis.set_title(f"{heading} standardized residual CDF")
            axis.set_xlabel(r"Threshold $k$ for $|r|/\sigma$")
            axis.set_ylabel(r"$P(|r|/\sigma \leq k)$")
            axis.legend(frameon=False, fontsize=7)
            axis.grid(False)
        figure.suptitle("LLPR standardized residual empirical CDF")
        _save_figure(figure, staging, "standardized_residual_cdf", dpi)
    finally:
        plt.close(figure)


def _render_all_figures(
    data: Mapping[str, Mapping[str, _PanelData]],
    shared_limits: Mapping[str, tuple[float, float]],
    selected: Sequence[tuple[str, str]],
    staging: Path,
    dpi: int,
) -> None:
    _render_selected(data, shared_limits, selected, staging, dpi)
    _render_comparison(
        data, shared_limits, "energy", "energy_comparison",
        "Energy uncertainty comparison", staging, dpi,
    )
    _render_comparison(
        data, shared_limits, "force_component", "force_component_comparison",
        "Force-component uncertainty comparison", staging, dpi,
    )
    _render_comparison(
        data, shared_limits, "force_structure", "force_structure_comparison",
        "Structure-level force uncertainty comparison", staging, dpi,
    )
    _render_reliability(data, staging, dpi)
    _render_cdf(data, staging, dpi)


def _strict_json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _input_hashes(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in _INPUT_NAMES}


def _figure_manifest(
    selected: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "selected_uncertainty_residual": {
            "layout": [2, 2],
            "paths": [list(item) for item in selected],
        },
        "energy_comparison": {"layout": [1, 3], "data_level": "energy"},
        "force_component_comparison": {
            "layout": [1, 3], "data_level": "force_component",
        },
        "force_structure_comparison": {
            "layout": [1, 3], "data_level": "force_structure",
        },
        "reliability": {"layout": [1, 2], "paths": "all_six"},
        "standardized_residual_cdf": {"layout": [1, 2], "paths": "all_six"},
    }


def _validate_selection(selected: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    normalized = tuple((str(variant), str(target)) for variant, target in selected)
    if len(normalized) != 4 or len(set(normalized)) != len(normalized):
        raise ValueError("selected plotting paths must contain four unique entries")
    if any(variant not in _VARIANTS or target not in {"energy", "forces"} for variant, target in normalized):
        raise ValueError("selected plotting path is invalid")
    return normalized


def _safe_paths(publication_root: Path, output_dir: Path) -> tuple[Path, Path]:
    root = Path(publication_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"publication root is not a directory: {root}")
    destination = Path(output_dir).resolve(strict=False)
    if destination == root or root.is_relative_to(destination):
        raise ValueError("output_dir cannot equal or contain the publication root")
    if destination.is_relative_to(root):
        relative = destination.relative_to(root)
        if not relative.parts or relative.parts[0] != "plots":
            raise ValueError("output_dir cannot replace canonical publication data")
    if destination.exists() and not destination.is_dir():
        raise ValueError("output_dir must be a directory path")
    return root, destination


def _validate_staging(staging: Path) -> None:
    expected = {
        *(f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "pdf")),
        "plotting_statistics.csv",
        "plotting_manifest.json",
    }
    actual = {path.name for path in staging.iterdir()}
    if actual != expected:
        raise RuntimeError(
            f"plot staging files mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for path in staging.iterdir():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"plot staging output is empty: {path.name}")
    for stem in FIGURE_STEMS:
        if not (staging / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"invalid PNG output: {stem}.png")
        pdf = (staging / f"{stem}.pdf").read_bytes()
        if not pdf.startswith(b"%PDF-") or not pdf.rstrip().endswith(b"%%EOF"):
            raise RuntimeError(f"invalid PDF output: {stem}.pdf")
    with (staging / "plotting_statistics.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames != list(PLOTTING_STATISTICS_FIELDS) or len(rows) != 9:
        raise RuntimeError("plotting statistics schema or row count mismatch")
    json.loads(
        (staging / "plotting_manifest.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _promote_directory(staging: Path, destination: Path) -> None:
    backup: Path | None = None
    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent)
        )
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def run_plot(
    publication_root: Path,
    *,
    output_dir: Path,
    selected: Sequence[tuple[str, str]] = DEFAULT_SELECTED,
    dpi: int = _DEFAULT_DPI,
) -> Path:
    """Render deterministic publication artifacts from validated formal CSVs."""
    root, destination = _safe_paths(publication_root, output_dir)
    selected_paths = _validate_selection(selected)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("plot dpi must be a positive integer")

    validate_publication_root(root)
    data = _load_plot_data(root)
    shared_limits = _all_shared_limits(data)

    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "savefig.bbox": None,
            "pdf.compression": 6,
        }
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        _render_all_figures(data, shared_limits, selected_paths, staging, dpi)
        _write_statistics(staging / "plotting_statistics.csv", data, shared_limits)
        outputs = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "formula_version": FORMULA_VERSION,
            "status": "complete",
            "inputs": _input_hashes(root),
            "outputs": outputs,
            "config": {
                "dpi": dpi,
                "formats": ["png", "pdf"],
                "font_family": "DejaVu Sans",
                "variant_order": list(_VARIANTS),
                "colors": dict(_VARIANT_COLORS),
                "zero_strategy": _ZERO_STRATEGY,
            },
            "selected": [list(item) for item in selected_paths],
            "units": {"energy": "eV/atom", "forces": "eV/?"},
            "figures": _figure_manifest(selected_paths),
            "shared_axes": {
                level: {"minimum": bounds[0], "maximum": bounds[1]}
                for level, bounds in shared_limits.items()
            },
        }
        _strict_json_dump(staging / "plotting_manifest.json", manifest)
        _validate_staging(staging)
        _promote_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
