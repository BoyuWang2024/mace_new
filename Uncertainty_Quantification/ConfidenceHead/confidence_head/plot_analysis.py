"""Statistics, CSV, and Matplotlib helpers for argmax-bin analysis."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import torch

from .binning import BranchBinning


matplotlib.use("Agg")
from matplotlib import pyplot as plt


CSV_FIELDS = (
    "bin_index",
    "left_edge",
    "right_edge",
    "physical_width",
    "representative",
    "train_label_count",
    "test_predicted_count",
    "mean",
    "q1",
    "median",
    "q3",
    "iqr",
    "lower_whisker",
    "upper_whisker",
    "minimum",
    "maximum",
)
_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


class PlotConflictError(RuntimeError):
    """Plot evidence is partial, malformed, or bound to other inputs."""


@dataclass(frozen=True)
class BoxplotScale:
    """Shared non-negative symlog parameters."""

    linthresh: float
    y_max: float


def _validated_observations(
    logits: object,
    errors: object,
    binning: BranchBinning,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(binning, BranchBinning):
        raise ValueError("binning must be a BranchBinning")
    bins = binning.num_bins
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or logits.shape[0] == 0
        or logits.shape[1] != bins
        or not logits.dtype.is_floating_point
    ):
        raise ValueError("logits sample/bins shape or dtype differs")
    if (
        not isinstance(errors, torch.Tensor)
        or errors.ndim != 1
        or errors.numel() != logits.shape[0]
        or not errors.dtype.is_floating_point
    ):
        raise ValueError("errors sample shape or dtype differs")
    bound_logits = logits.detach().to(device="cpu", dtype=torch.float64)
    bound_errors = errors.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(bound_logits).all()) or not bool(
        torch.isfinite(bound_errors).all()
    ):
        raise ValueError("logits and errors must be finite")
    if bool(torch.any(bound_errors < 0)):
        raise ValueError("errors must be non-negative")
    thresholds = binning.thresholds
    representatives = binning.representatives
    counts = binning.counts
    if (
        not isinstance(thresholds, torch.Tensor)
        or thresholds.ndim != 1
        or thresholds.numel() != bins - 1
        or thresholds.dtype != torch.float64
        or thresholds.device.type != "cpu"
        or not bool(torch.isfinite(thresholds).all())
        or (
            thresholds.numel() > 0
            and (
                float(thresholds[0]) <= 0.0
                or not bool(torch.all(thresholds[1:] > thresholds[:-1]))
            )
        )
    ):
        raise ValueError("binning thresholds differ")
    if (
        not isinstance(representatives, torch.Tensor)
        or representatives.shape != (bins,)
        or representatives.dtype != torch.float64
        or representatives.device.type != "cpu"
        or not bool(torch.isfinite(representatives).all())
        or bool(torch.any(representatives < 0))
    ):
        raise ValueError("binning representatives differ")
    if (
        not isinstance(counts, torch.Tensor)
        or counts.shape != (bins,)
        or counts.dtype not in _INTEGER_DTYPES
        or counts.device.type != "cpu"
        or bool(torch.any(counts < 0))
    ):
        raise ValueError("binning training counts differ")
    return (
        bound_logits.argmax(dim=-1).numpy(),
        bound_errors.numpy(),
    )


def _physical_rows(binning: BranchBinning) -> list[dict[str, Any]]:
    thresholds = binning.thresholds.tolist()
    rows: list[dict[str, Any]] = []
    for index in range(binning.num_bins):
        left = 0.0 if index == 0 else float(thresholds[index - 1])
        right = (
            math.inf
            if index == binning.num_bins - 1
            else float(thresholds[index])
        )
        rows.append(
            {
                "bin_index": index,
                "left_edge": left,
                "right_edge": right,
                "physical_width": math.inf if math.isinf(right) else right - left,
                "representative": float(binning.representatives[index]),
                "train_label_count": int(binning.counts[index]),
            }
        )
    return rows


def _distribution_statistics(values: np.ndarray) -> dict[str, float]:
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    iqr = float(q3 - q1)
    lower_fence = float(q1) - 1.5 * iqr
    upper_fence = float(q3) + 1.5 * iqr
    inside = values[(values >= lower_fence) & (values <= upper_fence)]
    return {
        "mean": float(np.mean(values)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqr": iqr,
        "lower_whisker": float(np.min(inside)),
        "upper_whisker": float(np.max(inside)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def argmax_bin_statistics(
    logits: torch.Tensor,
    errors: torch.Tensor,
    binning: BranchBinning,
) -> list[dict[str, Any]]:
    """Summarize continuous observed errors for every physical predicted bin."""
    predicted, observed = _validated_observations(logits, errors, binning)
    rows = _physical_rows(binning)
    empty = {
        name: math.nan
        for name in (
            "mean",
            "q1",
            "median",
            "q3",
            "iqr",
            "lower_whisker",
            "upper_whisker",
            "minimum",
            "maximum",
        )
    }
    for row in rows:
        values = observed[predicted == row["bin_index"]]
        row["test_predicted_count"] = int(values.size)
        row.update(_distribution_statistics(values) if values.size else empty)
    return rows


def _serialize(value: Any) -> str | int:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return ""
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return format(number, ".12g")
    return value


def _atomic_path(destination: Path, *, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=suffix,
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return path


def _fsync_replace(temporary: Path, destination: Path) -> None:
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _serialize(row[field]) for field in CSV_FIELDS})
    return handle.getvalue().encode("utf-8")


def write_statistics_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Atomically write one stable row per physical bin."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _csv_bytes(rows)
    temporary = _atomic_path(destination, suffix=".csv")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def shared_boxplot_scale(
    row_groups: Sequence[list[dict[str, Any]]],
) -> BoxplotScale:
    """Choose a shared symlog scale from non-empty observed distributions."""
    nonempty = [
        row
        for rows in row_groups
        for row in rows
        if int(row["test_predicted_count"]) > 0
    ]
    positive = [
        float(value)
        for row in nonempty
        for value in (row["q1"], row["median"], row["q3"])
        if float(value) > 0.0
    ]
    linthresh = (
        max(min(positive) * 0.5, np.finfo(float).tiny)
        if positive
        else 1e-6
    )
    maximum = max(
        (
            max(float(row["maximum"]), float(row["representative"]))
            for row in nonempty
        ),
        default=1.0,
    )
    return BoxplotScale(
        linthresh=linthresh,
        y_max=maximum * 1.08 if maximum > 0.0 else 1.0,
    )


def draw_argmax_bin_boxplot(
    axis: Any,
    branch: str,
    rows: list[dict[str, Any]],
    experiment_label: str,
    *,
    scale: BoxplotScale | None = None,
) -> None:
    """Draw one Carnet-aligned non-negative symlog argmax-bin panel."""
    if branch not in {"force", "energy"}:
        raise ValueError("branch must be force or energy")
    nonempty = [row for row in rows if row["test_predicted_count"] > 0]
    stats = [
        {
            "label": str(row["bin_index"]),
            "mean": row["mean"],
            "med": row["median"],
            "q1": row["q1"],
            "q3": row["q3"],
            "whislo": row["lower_whisker"],
            "whishi": row["upper_whisker"],
            "fliers": [],
        }
        for row in nonempty
    ]
    if stats:
        axis.bxp(
            stats,
            positions=[row["bin_index"] for row in nonempty],
            widths=0.62,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#4C78A8", "alpha": 0.72},
            medianprops={"color": "#D62728", "linewidth": 1.5},
        )
    positions = np.arange(len(rows))
    axis.scatter(
        positions,
        [row["representative"] for row in rows],
        marker="D",
        s=18,
        color="#444444",
        alpha=0.72,
        label="train-bin representative",
        zorder=3,
    )
    axis.set_xticks(
        positions,
        labels=[
            f"{row['bin_index']}\nn={row['test_predicted_count']}" for row in rows
        ],
        rotation=90,
        fontsize=7,
    )
    axis.set_xlim(-0.75, len(rows) - 0.25)
    bound_scale = scale or shared_boxplot_scale([rows])
    axis.set_yscale("symlog", linthresh=bound_scale.linthresh)
    axis.set_ylim(0.0, bound_scale.y_max)
    if branch == "force":
        title = "Per-atom mean force error"
        unit = "eV/Å"
    else:
        title = "Per-atom energy error"
        unit = "eV/atom"
    suffix = f" ({experiment_label})" if experiment_label else ""
    axis.set_xlabel("Predicted argmax bin (bin index / test count)")
    axis.set_ylabel(f"Absolute error ({unit})")
    axis.set_title(f"{title} grouped by predicted argmax bin{suffix}")
    axis.grid(axis="y", which="both", alpha=0.25)
    axis.legend(loc="upper left", fontsize=8)


def build_argmax_bin_boxplot_figure(
    branch: str,
    rows: list[dict[str, Any]],
    experiment_label: str = "",
    *,
    scale: BoxplotScale | None = None,
) -> tuple[Any, Any]:
    """Build one wide publication boxplot figure."""
    figure, axis = plt.subplots(figsize=(24, 10))
    draw_argmax_bin_boxplot(
        axis,
        branch,
        rows,
        experiment_label,
        scale=scale,
    )
    figure.tight_layout()
    return figure, axis


def render_argmax_bin_boxplot(
    path_stem: Path,
    branch: str,
    rows: list[dict[str, Any]],
    experiment_label: str,
    *,
    scale: BoxplotScale | None = None,
) -> tuple[Path, Path]:
    """Atomically render PNG and PDF from precomputed statistics."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    temporary_png = _atomic_path(png, suffix=".png")
    temporary_pdf = _atomic_path(pdf, suffix=".pdf")
    figure, _ = build_argmax_bin_boxplot_figure(
        branch,
        rows,
        experiment_label,
        scale=scale,
    )
    try:
        figure.savefig(temporary_png, dpi=300, format="png")
        figure.savefig(temporary_pdf, format="pdf")
        _fsync_replace(temporary_png, png)
        _fsync_replace(temporary_pdf, pdf)
    finally:
        plt.close(figure)
        temporary_png.unlink(missing_ok=True)
        temporary_pdf.unlink(missing_ok=True)
    return png, pdf


def statistics_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Return the canonical CSV bytes used for strict reuse checks."""
    return _csv_bytes(rows)
