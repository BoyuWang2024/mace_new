from __future__ import annotations

from pathlib import Path

import torch

from Uncertainty_Quantification.FGE.fge.plot_data import PlotBranch, PlotRun
from Uncertainty_Quantification.FGE.fge.plot_single import render_single_run
from Uncertainty_Quantification.FGE.fge.plot_style import PlotConfig


def _branch(scale: float) -> PlotBranch:
    energy_reference = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float64)
    energy_prediction = energy_reference + scale * torch.tensor(
        [0.2, -0.1, 0.3, -0.2], dtype=torch.float64
    )
    force_reference = torch.linspace(-2.0, 2.0, 24, dtype=torch.float64)
    force_prediction = force_reference + scale * torch.linspace(0.05, 0.3, 24)
    risk_rows = tuple(
        {"metric": metric, "coverage": str(coverage), "risk": str(scale * risk), "count": "4"}
        for metric in ("energy_per_atom_std", "force_component_std")
        for coverage, risk in ((1.0, 0.3), (0.5, 0.1))
    )
    return PlotBranch(
        energy_reference=energy_reference,
        energy_prediction=energy_prediction,
        energy_residual=(energy_prediction - energy_reference).abs(),
        energy_uncertainty=torch.tensor([0.1, 0.2, 0.4, 0.3]) * scale,
        force_reference=force_reference,
        force_prediction=force_prediction,
        force_residual=(force_prediction - force_reference).abs(),
        force_uncertainty=torch.linspace(0.05, 0.4, 24) * scale,
        metrics={
            "energy_per_atom": {"rmse": 0.2 * scale, "count": 4},
            "force_component": {"rmse": 0.1 * scale, "count": 24},
        },
        correlations=(),
        risk_coverage=risk_rows,
    )


def test_render_single_run_writes_five_png_pdf_pairs_per_branch(tmp_path: Path) -> None:
    run = PlotRun(
        name="experiment",
        root=tmp_path / "formal-result",
        branches={"equal_weight": _branch(1.0), "validation_weighted": _branch(0.8)},
    )

    records = render_single_run(run, tmp_path / "figures", PlotConfig(dpi=72, grid_size=16))
    images = tuple((tmp_path / "figures").rglob("*.*"))
    png = tuple(path for path in images if path.suffix == ".png")
    pdf = tuple(path for path in images if path.suffix == ".pdf")

    assert len(records) == 10
    assert len(png) == 10
    assert len(pdf) == 10
    assert all(path.stat().st_size > 0 for path in (*png, *pdf))
