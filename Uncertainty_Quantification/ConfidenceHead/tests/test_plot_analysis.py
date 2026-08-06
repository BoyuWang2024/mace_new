"""Tests for native MACE argmax-bin statistics and per-run plots."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
import pytest
import torch

from confidence_head.binning import BranchBinning
from confidence_head.plot_analysis import (
    CSV_FIELDS,
    PlotConflictError,
    argmax_bin_statistics,
    render_argmax_bin_boxplot,
    write_statistics_csv,
)
from confidence_head.workflows.evaluate import run_evaluate
from confidence_head.workflows.plot_argmax_bin_boxplots import (
    run_plot_argmax_bin_boxplots,
)

from test_evaluate_workflow import _completed_run


def _binning() -> BranchBinning:
    return BranchBinning(
        algorithm="fixed_linear_v1",
        thresholds=torch.tensor([1.0, 2.0], dtype=torch.float64),
        representatives=torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64),
        counts=torch.tensor([10, 0, 5], dtype=torch.int64),
        representative_sources=("analytic_center",) * 3,
        overflow_count=1,
        max_error=3.0,
        bin_width=1.0,
    )


def test_argmax_bin_statistics_retain_empty_physical_bins() -> None:
    logits = torch.tensor(
        [[4.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 4.0]]
    )
    errors = torch.tensor([0.1, 0.3, 2.8])

    rows = argmax_bin_statistics(logits, errors, _binning())

    assert len(rows) == 3
    assert [row["bin_index"] for row in rows] == [0, 1, 2]
    assert rows[0]["left_edge"] == 0.0
    assert rows[0]["right_edge"] == 1.0
    assert rows[0]["median"] == pytest.approx(0.2)
    assert rows[0]["minimum"] == pytest.approx(0.1)
    assert rows[0]["maximum"] == pytest.approx(0.3)
    assert rows[1]["test_predicted_count"] == 0
    assert math.isnan(rows[1]["median"])
    assert math.isinf(rows[2]["right_edge"])
    assert rows[2]["train_label_count"] == 5


def test_argmax_bin_statistics_use_observed_tukey_whiskers() -> None:
    logits = torch.tensor([[5.0, 0.0, 0.0]] * 5)
    errors = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0])

    row = argmax_bin_statistics(logits, errors, _binning())[0]

    assert row["q1"] == pytest.approx(1.0)
    assert row["median"] == pytest.approx(2.0)
    assert row["q3"] == pytest.approx(3.0)
    assert row["lower_whisker"] == pytest.approx(0.0)
    assert row["upper_whisker"] == pytest.approx(3.0)
    assert row["maximum"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("logits", "errors", "message"),
    [
        (torch.ones(2, 2), torch.ones(2), "bins"),
        (torch.ones(2, 3), torch.ones(1), "sample"),
        (torch.tensor([[1.0, float("nan"), 0.0]]), torch.ones(1), "finite"),
        (torch.ones(1, 3), torch.tensor([-1.0]), "non-negative"),
    ],
)
def test_argmax_bin_statistics_reject_malformed_inputs(
    logits: torch.Tensor, errors: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        argmax_bin_statistics(logits, errors, _binning())


def test_csv_and_rendered_plot_are_complete_and_parseable(tmp_path: Path) -> None:
    rows = argmax_bin_statistics(
        torch.tensor([[4.0, 0.0, 0.0], [0.0, 0.0, 4.0]]),
        torch.tensor([0.2, 2.5]),
        _binning(),
    )
    csv_path = tmp_path / "statistics.csv"
    write_statistics_csv(csv_path, rows)
    png_path, pdf_path = render_argmax_bin_boxplot(
        tmp_path / "boxplot",
        "force",
        rows,
        "synthetic / fixed_linear_v1",
    )

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        parsed = list(reader)
        assert tuple(reader.fieldnames or ()) == CSV_FIELDS
    assert len(parsed) == 3
    assert parsed[1]["median"] == ""
    assert matplotlib.get_backend().lower() == "agg"
    assert png_path.stat().st_size > 1000
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_per_run_plot_workflow_writes_exact_force_outputs_and_reuses(
    tmp_path: Path,
) -> None:
    config, run_root = _completed_run(tmp_path, branch="force")
    run_evaluate(config)

    csv_path = run_plot_argmax_bin_boxplots(config)
    output_dir = run_root / "plots" / "argmax_bin_boxplots"
    png_path = output_dir / "test_force_argmax_bin_boxplot.png"
    pdf_path = output_dir / "test_force_argmax_bin_boxplot.pdf"
    before = {path: path.stat().st_mtime_ns for path in (csv_path, png_path, pdf_path)}

    assert csv_path == output_dir / "test_force_argmax_bin_statistics.csv"
    assert len(list(csv.DictReader(csv_path.open(encoding="utf-8")))) == 3
    assert png_path.stat().st_size > 1000
    assert pdf_path.stat().st_size > 1000
    assert run_plot_argmax_bin_boxplots(config) == csv_path
    assert before == {path: path.stat().st_mtime_ns for path in before}


def test_per_run_plot_workflow_rejects_partial_outputs(tmp_path: Path) -> None:
    config, run_root = _completed_run(tmp_path, branch="energy")
    run_evaluate(config)
    csv_path = run_plot_argmax_bin_boxplots(config)
    pdf_path = csv_path.with_name("test_energy_argmax_bin_boxplot.pdf")
    pdf_path.unlink()

    with pytest.raises(PlotConflictError, match="partial"):
        run_plot_argmax_bin_boxplots(config)
    assert csv_path.is_file()
