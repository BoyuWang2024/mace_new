"""Deterministic publication figures from validated LLPR result CSVs only."""

from __future__ import annotations

import csv
import ctypes
import errno
import fcntl
import json
import hashlib
import math
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

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
    "undefined_standardized_residual_rows",
    "zero_zero_standardized_residual_rows",
    "uncertainty_residual_correlation",
    "correlation_valid_rows",
    "correlation_degenerate",
    "shared_axis_min",
    "shared_axis_max",
)

_SNAPSHOT_SOURCE_NAMES = (
    "progress.pt",
    "manifest.json",
    *(f"{variant}/{filename}" for variant in _VARIANTS for filename in (
        "energy.csv",
        "force_components.csv",
        "force_structure.csv",
        "summary.json",
    )),
)
_INPUT_NAMES = (*_SNAPSHOT_SOURCE_NAMES, "validation.json")


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
            unit="eV/\u00c5",
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
            unit="eV/\u00c5",
        )
    return result


def _load_carnet_panels(
    root: Path,
) -> dict[tuple[str, str], Any]:
    """Load only q, std, and residual for the four density panels."""
    from .density_plotting import CARNET_SELECTED, DensityPanel

    panels: dict[tuple[str, str], Any] = {}
    for variant, target in CARNET_SELECTED:
        filename = "energy.csv" if target == "energy" else "force_components.csv"
        frame = pd.read_csv(
            root / variant / filename,
            usecols=("q", "std", "residual"),
        )
        uncertainty = frame["std"].to_numpy(dtype=np.float64, copy=True)
        residual = frame["residual"].to_numpy(dtype=np.float64, copy=True)
        np.abs(residual, out=residual)
        panels[(variant, target)] = DensityPanel(
            uncertainty=uncertainty,
            q=frame["q"].to_numpy(dtype=np.float64, copy=True),
            absolute_residual=residual,
            target=target,
            unit="eV/atom" if target == "energy" else "eV/\u00c5",
        )
    return panels


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


def _standardized_residuals(
    panel: _PanelData,
) -> tuple[np.ndarray, int, int]:
    positive_std = panel.uncertainty > 0.0
    zero_zero = (panel.uncertainty == 0.0) & (panel.signed_residual == 0.0)
    undefined = (panel.uncertainty == 0.0) & (panel.signed_residual != 0.0)
    values = np.abs(
        panel.signed_residual[positive_std] / panel.uncertainty[positive_std]
    )
    return values, int(np.count_nonzero(undefined)), int(np.count_nonzero(zero_zero))


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, int, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    valid_left = left[valid]
    valid_right = right[valid]
    count = int(valid_left.size)
    if count < 2 or np.ptp(valid_left) == 0.0 or np.ptp(valid_right) == 0.0:
        return 0.0, count, 1
    value = float(np.corrcoef(valid_left, valid_right)[0, 1])
    if not math.isfinite(value):
        return 0.0, count, 1
    return value, count, 0


def _statistics_row(
    variant: str,
    panel: _PanelData,
    limits: tuple[float, float],
) -> dict[str, Any]:
    uncertainty = panel.uncertainty
    error = panel.absolute_residual
    log_mask = (uncertainty > 0.0) & (error > 0.0)
    standardized, undefined_rows, zero_zero_rows = _standardized_residuals(panel)
    correlation, correlation_rows, correlation_degenerate = _correlation(
        uncertainty, error
    )
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
            float(np.mean(standardized)) if standardized.size else 0.0
        ),
        "undefined_standardized_residual_rows": undefined_rows,
        "zero_zero_standardized_residual_rows": zero_zero_rows,
        "uncertainty_residual_correlation": correlation,
        "correlation_valid_rows": correlation_rows,
        "correlation_degenerate": correlation_degenerate,
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
                title=f"{_VARIANT_LABELS[variant]} / {target_label}",
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
        z, _, _ = _standardized_residuals(panel)
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
    z, _, _ = _standardized_residuals(panel)
    if not z.size:
        return np.zeros_like(grid)
    z = np.sort(z)
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


def _relative_input_path(name: str) -> Path:
    relative = Path(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"publication input path escapes its root: {name}")
    return relative


def _regular_state(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_regular_input(root: Path, name: str) -> tuple[int, os.stat_result]:
    relative = _relative_input_path(name)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    directory_descriptor: int | None = None
    descriptor: int | None = None
    try:
        directory_descriptor = os.open(root, directory_flags)
        for part in relative.parts[:-1]:
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            relative.parts[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ValueError(
            f"publication input must be a safe regular file: {name}"
        ) from error
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    state = os.fstat(descriptor)
    if not stat.S_ISREG(state.st_mode):
        os.close(descriptor)
        raise ValueError(f"publication input must be a regular file: {name}")
    return descriptor, state


def _assert_regular_unchanged(
    descriptor: int, before: os.stat_result, name: str
) -> None:
    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(after.st_mode)
        or _regular_state(after) != _regular_state(before)
    ):
        raise ValueError(f"publication input changed while reading: {name}")


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _regular_file_hash(root: Path, name: str) -> str:
    descriptor, before = _open_regular_input(root, name)
    try:
        digest = _hash_descriptor(descriptor)
        _assert_regular_unchanged(descriptor, before, name)
        return digest
    finally:
        os.close(descriptor)


def _regular_file_hashes(root: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: _regular_file_hash(root, name) for name in names}


def _snapshot_target(snapshot: Path, name: str) -> Path:
    relative = _relative_input_path(name)
    snapshot_state = snapshot.lstat()
    if stat.S_ISLNK(snapshot_state.st_mode) or not stat.S_ISDIR(
        snapshot_state.st_mode
    ):
        raise ValueError("plot input snapshot root must be a real directory")
    current = snapshot
    for part in relative.parts[:-1]:
        current = current / part
        current.mkdir(mode=0o700, exist_ok=True)
        state = current.lstat()
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
            raise ValueError(f"plot input snapshot path is unsafe: {name}")
    return current / relative.parts[-1]


def _copy_snapshot_input(source: Path, snapshot: Path, name: str) -> None:
    source_descriptor, source_before = _open_regular_input(source, name)
    target = _snapshot_target(snapshot, name)
    target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        target_flags |= os.O_NOFOLLOW
    target_descriptor: int | None = None
    digest = hashlib.sha256()
    try:
        target_descriptor = os.open(target, target_flags, 0o600)
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(target_descriptor, remaining)
                remaining = remaining[written:]
        os.fsync(target_descriptor)
        target_state = os.fstat(target_descriptor)
        if (
            not stat.S_ISREG(target_state.st_mode)
            or target_state.st_size != source_before.st_size
        ):
            raise ValueError(f"plot input snapshot copy is invalid: {name}")
        _assert_regular_unchanged(source_descriptor, source_before, name)
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        os.close(source_descriptor)
    if _regular_file_hash(snapshot, name) != digest.hexdigest():
        raise ValueError(f"plot input snapshot copy hash mismatch: {name}")


def _input_hashes(root: Path) -> dict[str, str]:
    return _regular_file_hashes(root, _INPUT_NAMES)


def _snapshot_source_hashes(root: Path) -> dict[str, str]:
    return _regular_file_hashes(root, _SNAPSHOT_SOURCE_NAMES)


def _standardized_residual_counts(
    data: Mapping[str, Mapping[str, _PanelData]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for level in ("energy", "force_component"):
        for variant in _VARIANTS:
            values, undefined_rows, zero_zero_rows = _standardized_residuals(
                data[level][variant]
            )
            result[f"{variant}/{level}"] = {
                "ecdf_rows": int(values.size),
                "undefined_rows": undefined_rows,
                "zero_zero_rows": zero_zero_rows,
            }
    return result


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


def _absolute_without_resolution(path: Path) -> Path:
    expanded = Path(os.path.expanduser(os.fspath(path)))
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _reject_existing_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise ValueError(f"output_dir contains a symlink component: {current}")


def _safe_paths(publication_root: Path, output_dir: Path) -> tuple[Path, Path]:
    root_lexical = Path(os.path.abspath(_absolute_without_resolution(Path(publication_root))))
    root = root_lexical.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"publication root is not a directory: {root}")
    for name in _VARIANTS:
        child = root_lexical / name
        if child.is_symlink():
            raise ValueError(f"publication canonical child must not be a symlink: {child}")

    output_raw = _absolute_without_resolution(Path(output_dir))
    _reject_existing_symlink_components(output_raw)
    destination_lexical = Path(os.path.abspath(output_raw))
    destination_resolved = destination_lexical.resolve(strict=False)
    for checked_root, checked_output in (
        (root_lexical, destination_lexical),
        (root, destination_resolved),
    ):
        if checked_output == checked_root or checked_root.is_relative_to(checked_output):
            raise ValueError("output_dir cannot equal or contain the publication root")
        if checked_output.is_relative_to(checked_root):
            raise ValueError("output_dir cannot be inside the publication root")
    if destination_lexical.exists() and not destination_lexical.is_dir():
        raise ValueError("output_dir must be a directory path")
    return root, destination_lexical


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
    numeric_fields = PLOTTING_STATISTICS_FIELDS[4:]
    for row_index, row in enumerate(rows):
        for field in numeric_fields:
            try:
                value = float(row[field])
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"plotting statistics row {row_index} field {field} must be numeric"
                ) from error
            if not math.isfinite(value):
                raise RuntimeError(
                    f"plotting statistics row {row_index} field {field} must be finite"
                )
    json.loads(
        (staging / "plotting_manifest.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _parse_carnet_statistics(path: Path) -> list[dict[str, Any]]:
    from .density_plotting import CARNET_SELECTED, CARNET_STATISTICS_FIELDS

    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise RuntimeError("carnet statistics cannot be parsed") from error
    if reader.fieldnames != list(CARNET_STATISTICS_FIELDS) or len(rows) != 4:
        raise RuntimeError("carnet statistics schema or row count mismatch")
    integer_fields = {
        "rows", "finite_rows", "nonfinite_rows", "nonpositive_std_rows",
        "zero_q_rows", "zero_q_zero_residual_rows",
        "zero_q_nonzero_residual_rows",
        "zero_absolute_residual_rows", "metric_rows", "log_plot_rows",
        "excluded_from_log_rows", "correlation_rows", "correlation_degenerate",
    }
    nullable_fields = {"pearson_log", "spearman_log"}
    parsed: list[dict[str, Any]] = []
    for index, (raw, expected_path) in enumerate(zip(rows, CARNET_SELECTED)):
        if (raw["variant"], raw["target"]) != expected_path:
            raise RuntimeError("carnet statistics panel order mismatch")
        row: dict[str, Any] = {}
        for field in CARNET_STATISTICS_FIELDS:
            value = raw[field]
            if field in integer_fields:
                try:
                    numeric = int(value)
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"carnet statistics row {index} field {field} must be an integer"
                    ) from error
                if str(numeric) != value or numeric < 0:
                    raise RuntimeError(
                        f"carnet statistics row {index} field {field} is invalid"
                    )
                row[field] = numeric
            elif field in nullable_fields and value == "":
                row[field] = None
            elif field in nullable_fields or field in {"axis_min", "axis_max"}:
                try:
                    numeric_float = float(value)
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"carnet statistics row {index} field {field} must be numeric"
                    ) from error
                if not math.isfinite(numeric_float):
                    raise RuntimeError(
                        f"carnet statistics row {index} field {field} must be finite"
                    )
                row[field] = numeric_float
            else:
                if not isinstance(value, str) or not value:
                    raise RuntimeError(
                        f"carnet statistics row {index} field {field} is invalid"
                    )
                row[field] = value
        if row["finite_rows"] + row["nonfinite_rows"] != row["rows"]:
            raise RuntimeError("carnet statistics finite counts are inconsistent")
        if (
            row["zero_q_zero_residual_rows"]
            + row["zero_q_nonzero_residual_rows"]
            != row["zero_q_rows"]
            or row["zero_q_rows"] > row["nonpositive_std_rows"]
        ):
            raise RuntimeError(
                "carnet statistics zero-q counts are inconsistent"
            )
        if row["log_plot_rows"] + row["excluded_from_log_rows"] != row["rows"]:
            raise RuntimeError("carnet statistics exclusion counts are inconsistent")
        if row["correlation_rows"] != row["log_plot_rows"]:
            raise RuntimeError("carnet statistics correlation count is inconsistent")
        if not (0.0 < row["axis_min"] < row["axis_max"]):
            raise RuntimeError("carnet statistics axis bounds are invalid")
        defined = row["pearson_log"] is not None and row["spearman_log"] is not None
        if row["correlation_status"] == "ok":
            if row["correlation_degenerate"] != 0 or not defined:
                raise RuntimeError("carnet statistics correlation status is inconsistent")
        elif (
            not row["correlation_status"].startswith("undefined_")
            or row["correlation_degenerate"] != 1
            or defined
        ):
            raise RuntimeError("carnet statistics correlation status is inconsistent")
        parsed.append(row)
    return parsed


def _validate_carnet_staging(staging: Path) -> None:
    from .density_plotting import (
        CARNET_DENSITY_CONFIG,
        CARNET_FIGURE_STEMS,
        CARNET_SELECTED,
    )

    expected = {
        *(
            f"{stem}.{suffix}"
            for stem in CARNET_FIGURE_STEMS.values()
            for suffix in ("png", "pdf")
        ),
        "plotting_statistics.csv",
        "plotting_manifest.json",
    }
    actual = {path.name for path in staging.iterdir()}
    if actual != expected:
        raise RuntimeError(
            f"carnet staging files mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for path in staging.iterdir():
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"carnet staging output is not a regular file: {path.name}")
        if path.stat().st_size == 0:
            raise RuntimeError(f"carnet staging output is empty: {path.name}")
    for stem in CARNET_FIGURE_STEMS.values():
        png = (staging / f"{stem}.png").read_bytes()
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"invalid PNG output: {stem}.png")
        pdf = (staging / f"{stem}.pdf").read_bytes()
        if not pdf.startswith(b"%PDF-") or not pdf.rstrip().endswith(b"%%EOF"):
            raise RuntimeError(f"invalid PDF output: {stem}.pdf")

    rows = _parse_carnet_statistics(staging / "plotting_statistics.csv")
    try:
        manifest = json.loads(
            (staging / "plotting_manifest.json").read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("carnet manifest is not strict JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "formula_version", "status", "style", "inputs",
        "outputs", "config", "selected", "shared_axes", "statistics",
    }:
        raise RuntimeError("carnet manifest fields mismatch")
    if manifest["status"] != "complete" or manifest["style"] != "carnet_density":
        raise RuntimeError("carnet manifest status or style mismatch")
    if manifest["selected"] != [list(item) for item in CARNET_SELECTED]:
        raise RuntimeError("carnet manifest selection mismatch")
    if not isinstance(manifest["inputs"], dict) or set(manifest["inputs"]) != set(_INPUT_NAMES):
        raise RuntimeError("carnet manifest inputs mismatch")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in manifest["inputs"].values()
    ):
        raise RuntimeError("carnet manifest input SHA256 is invalid")

    config = manifest["config"]
    if not isinstance(config, dict):
        raise RuntimeError("carnet manifest config is invalid")
    expected_config = {
        **CARNET_DENSITY_CONFIG,
        "figure_size": list(CARNET_DENSITY_CONFIG["figure_size"]),
        "contour_masses": list(CARNET_DENSITY_CONFIG["contour_masses"]),
        "formats": ["png", "pdf"],
        "dpi": config.get("dpi"),
    }
    if (
        isinstance(config.get("dpi"), bool)
        or not isinstance(config.get("dpi"), int)
        or config["dpi"] <= 0
        or config != expected_config
    ):
        raise RuntimeError("carnet manifest config mismatch")

    expected_statistics = {
        f"{row['variant']}/{row['target']}": row for row in rows
    }
    if manifest["statistics"] != expected_statistics:
        raise RuntimeError("carnet manifest statistics mismatch")
    axes = manifest["shared_axes"]
    if not isinstance(axes, dict) or set(axes) != set(expected_statistics):
        raise RuntimeError("carnet manifest shared axes mismatch")
    for key, row in expected_statistics.items():
        if axes[key] != {
            "minimum": row["axis_min"],
            "maximum": row["axis_max"],
        }:
            raise RuntimeError("carnet manifest shared axes disagree with statistics")

    expected_outputs = expected - {"plotting_manifest.json"}
    outputs = manifest["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
        raise RuntimeError("carnet manifest outputs mismatch")
    for name, record in outputs.items():
        path = staging / name
        if (
            not isinstance(record, dict)
            or set(record) != {"size", "sha256"}
            or isinstance(record["size"], bool)
            or not isinstance(record["size"], int)
            or record["size"] != path.stat().st_size
            or record["sha256"] != sha256_file(path)
        ):
            raise RuntimeError(f"carnet manifest size or SHA256 mismatch: {name}")


@contextmanager
def _output_lock(destination: Path) -> Iterator[None]:
    lock_path = destination.parent / f".{destination.name}.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _rename_exchange(first: Path, second: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(first),
        -100,
        os.fsencode(second),
        2,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _best_effort_remove(path: Path) -> None:
    try:
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    except Exception:
        return


def _cleanup_stale_directories(destination: Path) -> None:
    for stale in destination.parent.glob(f".{destination.name}.stale-*"):
        _best_effort_remove(stale)


_EXCHANGE_FALLBACK_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOSYS,
        errno.EXDEV,
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        getattr(errno, "ENOTSUP", errno.EINVAL),
    }
)


def _directory_identity(state: os.stat_result) -> tuple[int, int]:
    return state.st_dev, state.st_ino


def _directory_path_identity(path: Path, *, role: str) -> tuple[int, int]:
    try:
        state = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"{role} changed during plot publication") from error
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise RuntimeError(f"{role} changed during plot publication")
    return _directory_identity(state)


def _optional_directory_path_identity(
    path: Path, *, role: str
) -> tuple[int, int] | None:
    try:
        state = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise RuntimeError(f"{role} changed during plot publication")
    return _directory_identity(state)


def _open_directory_identity(
    path: Path, *, role: str
) -> tuple[int, tuple[int, int]]:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{role} changed during plot publication") from error
    try:
        state = os.fstat(descriptor)
        if not stat.S_ISDIR(state.st_mode):
            raise RuntimeError(f"{role} changed during plot publication")
        identity = _directory_identity(state)
        if _directory_path_identity(path, role=role) != identity:
            raise RuntimeError(f"{role} changed during plot publication")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _assert_directory_identity(
    path: Path, expected: tuple[int, int], *, role: str
) -> None:
    if _directory_path_identity(path, role=role) != expected:
        raise RuntimeError(f"{role} changed during plot publication")


def _assert_path_missing(path: Path, *, role: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise RuntimeError(f"{role} changed during plot publication")


def _rollback_backup_promotion(
    staging: Path,
    destination: Path,
    backup: Path,
    *,
    staging_identity: tuple[int, int],
    destination_identity: tuple[int, int],
) -> None:
    live_destination = _optional_directory_path_identity(
        destination, role="plot destination"
    )
    if live_destination == staging_identity:
        if _optional_directory_path_identity(
            staging, role="plot staging directory"
        ) is not None:
            raise RuntimeError(
                "plot staging directory changed; rollback would overwrite it"
            )
        os.replace(destination, staging)
        _assert_directory_identity(
            staging, staging_identity, role="plot staging directory"
        )
    elif live_destination is not None:
        raise RuntimeError(
            "plot destination changed; rollback would overwrite it"
        )

    _assert_directory_identity(
        backup, destination_identity, role="plot backup directory"
    )
    _assert_path_missing(destination, role="plot destination")
    os.replace(backup, destination)
    _assert_directory_identity(
        destination, destination_identity, role="plot destination"
    )


def _promote_directory_with_backup(
    staging: Path,
    destination: Path,
    *,
    staging_identity: tuple[int, int],
    destination_identity: tuple[int, int],
) -> None:
    backup_container = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.backup-",
            dir=destination.parent,
        )
    )
    backup = backup_container / "previous"
    old_moved = False
    try:
        _assert_directory_identity(
            staging, staging_identity, role="plot staging directory"
        )
        _assert_directory_identity(
            destination, destination_identity, role="plot destination"
        )
        os.replace(destination, backup)
        old_moved = True
        _assert_directory_identity(
            backup, destination_identity, role="plot backup directory"
        )
        _assert_path_missing(destination, role="plot destination")
        _assert_directory_identity(
            staging, staging_identity, role="plot staging directory"
        )
        os.replace(staging, destination)
        _assert_directory_identity(
            destination, staging_identity, role="plot destination"
        )
        _assert_path_missing(staging, role="plot staging directory")
        _assert_directory_identity(
            backup, destination_identity, role="plot backup directory"
        )
    except BaseException:
        if old_moved:
            try:
                _rollback_backup_promotion(
                    staging,
                    destination,
                    backup,
                    staging_identity=staging_identity,
                    destination_identity=destination_identity,
                )
            except BaseException as rollback_error:
                raise RuntimeError(
                    "plot fallback publication failed and could not safely "
                    f"restore the old output; it remains at {backup}"
                ) from rollback_error
        _best_effort_remove(backup_container)
        raise
    _best_effort_remove(backup_container)


def _promote_existing_directory(staging: Path, destination: Path) -> None:
    if _directory_path_identity(
        staging.parent, role="plot staging parent"
    ) != _directory_path_identity(
        destination.parent, role="plot destination parent"
    ):
        raise RuntimeError("plot staging and destination must share one directory")
    staging_descriptor, staging_identity = _open_directory_identity(
        staging, role="plot staging directory"
    )
    destination_descriptor: int | None = None
    try:
        destination_descriptor, destination_identity = _open_directory_identity(
            destination, role="plot destination"
        )
        try:
            _rename_exchange(staging, destination)
        except OSError as error:
            if error.errno not in _EXCHANGE_FALLBACK_ERRNOS:
                raise
            _assert_directory_identity(
                staging, staging_identity, role="plot staging directory"
            )
            _assert_directory_identity(
                destination, destination_identity, role="plot destination"
            )
            _promote_directory_with_backup(
                staging,
                destination,
                staging_identity=staging_identity,
                destination_identity=destination_identity,
            )
            return
        _assert_directory_identity(
            destination, staging_identity, role="plot destination"
        )
        _assert_directory_identity(
            staging, destination_identity, role="plot staging directory"
        )
    finally:
        try:
            if destination_descriptor is not None:
                os.close(destination_descriptor)
        finally:
            os.close(staging_descriptor)
    if staging.exists():
        _best_effort_remove(staging)


def _promote_directory(staging: Path, destination: Path) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    _promote_existing_directory(staging, destination)


def _create_input_snapshot(root: Path, destination: Path) -> Path:
    snapshot = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.snapshot-",
            dir=destination.parent,
        )
    )
    try:
        for name in _SNAPSHOT_SOURCE_NAMES:
            _copy_snapshot_input(root, snapshot, name)
        validate_publication_root(snapshot)
        return snapshot
    except BaseException:
        _best_effort_remove(snapshot)
        raise


def _run_plot_locked(
    snapshot_root: Path,
    live_root: Path,
    destination: Path,
    selected_paths: Sequence[tuple[str, str]],
    dpi: int,
    snapshot_source_hashes: Mapping[str, str],
    input_hashes: Mapping[str, str],
) -> Path:
    data = _load_plot_data(snapshot_root)
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

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stale-", dir=destination.parent)
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
            "inputs": input_hashes,
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
            "units": {"energy": "eV/atom", "forces": "eV/\u00c5"},
            "figures": _figure_manifest(selected_paths),
            "shared_axes": {
                level: {"minimum": bounds[0], "maximum": bounds[1]}
                for level, bounds in shared_limits.items()
            },
            "standardized_residual_counts": _standardized_residual_counts(data),
        }
        _strict_json_dump(staging / "plotting_manifest.json", manifest)
        _validate_staging(staging)
        if _snapshot_source_hashes(live_root) != snapshot_source_hashes:
            raise ValueError("publication input changed during plotting")
        _promote_directory(staging, destination)
    finally:
        _best_effort_remove(staging)
    return destination


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_components(destination)
    with _output_lock(destination):
        _cleanup_stale_directories(destination)
        snapshot: Path | None = None
        try:
            snapshot = _create_input_snapshot(root, destination)
            snapshot_source_hashes = _snapshot_source_hashes(snapshot)
            input_hashes = _input_hashes(snapshot)
            return _run_plot_locked(
                snapshot,
                root,
                destination,
                selected_paths,
                dpi,
                snapshot_source_hashes,
                input_hashes,
            )
        finally:
            if snapshot is not None:
                _best_effort_remove(snapshot)


_run_plot_diagnostic_suite = run_plot


def run_plot(
    publication_root: Path,
    *,
    output_dir: Path,
    selected: Sequence[tuple[str, str]] = DEFAULT_SELECTED,
    style: str = "diagnostic_suite",
    dpi: int | None = None,
) -> Path:
    """Render validated CSV snapshots with either supported publication style."""
    if style == "diagnostic_suite":
        return _run_plot_diagnostic_suite(
            publication_root,
            output_dir=output_dir,
            selected=selected,
            dpi=_DEFAULT_DPI if dpi is None else dpi,
        )
    if style != "carnet_density":
        raise ValueError("plot style is invalid")
    if tuple(selected) != DEFAULT_SELECTED:
        raise ValueError(
            "carnet_density selected paths must exactly match the four required paths"
        )
    from .density_plotting import (
        CARNET_DENSITY_CONFIG,
        CARNET_SELECTED,
        render_carnet_density,
        write_carnet_statistics,
    )

    root, destination = _safe_paths(publication_root, output_dir)
    render_dpi = CARNET_DENSITY_CONFIG["dpi"] if dpi is None else dpi
    if isinstance(render_dpi, bool) or not isinstance(render_dpi, int) or render_dpi <= 0:
        raise ValueError("plot dpi must be a positive integer")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_components(destination)
    with _output_lock(destination):
        _cleanup_stale_directories(destination)
        snapshot: Path | None = None
        staging: Path | None = None
        try:
            snapshot = _create_input_snapshot(root, destination)
            snapshot_source_hashes = _snapshot_source_hashes(snapshot)
            input_hashes = _input_hashes(snapshot)
            panels = _load_carnet_panels(snapshot)
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.stale-",
                    dir=destination.parent,
                )
            )
            statistics, limits = render_carnet_density(
                panels, staging, dpi=render_dpi
            )
            write_carnet_statistics(
                staging / "plotting_statistics.csv", statistics
            )
            outputs = {
                path.name: {
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(staging.iterdir(), key=lambda item: item.name)
            }
            config = {
                **CARNET_DENSITY_CONFIG,
                "figure_size": list(CARNET_DENSITY_CONFIG["figure_size"]),
                "contour_masses": list(CARNET_DENSITY_CONFIG["contour_masses"]),
                "formats": ["png", "pdf"],
                "dpi": render_dpi,
            }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "formula_version": FORMULA_VERSION,
                "status": "complete",
                "style": "carnet_density",
                "inputs": input_hashes,
                "outputs": outputs,
                "config": config,
                "selected": [list(item) for item in CARNET_SELECTED],
                "shared_axes": {
                    f"{variant}/{target}": {
                        "minimum": bound[0],
                        "maximum": bound[1],
                    }
                    for (variant, target), bound in limits.items()
                },
                "statistics": {
                    f"{row['variant']}/{row['target']}": row
                    for row in statistics
                },
            }
            _strict_json_dump(staging / "plotting_manifest.json", manifest)
            _validate_carnet_staging(staging)
            if _snapshot_source_hashes(root) != snapshot_source_hashes:
                raise ValueError("publication input changed during plotting")
            _promote_directory(staging, destination)
            return destination
        finally:
            if snapshot is not None:
                _best_effort_remove(snapshot)
            if staging is not None:
                _best_effort_remove(staging)
