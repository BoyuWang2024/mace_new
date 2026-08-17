from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.plot_data import PlotRun
from Uncertainty_Quantification.FGE.fge.plot_style import PlotConfig
from Uncertainty_Quantification.FGE.fge.plot_workflow import render_dataset_figures
from Uncertainty_Quantification.FGE.tests.test_plot_single import _branch
from Uncertainty_Quantification.FGE.tests.test_plot_stress import _stress_branch


def _four_runs(tmp_path: Path, has_stress: bool) -> dict[str, PlotRun]:
    factory = _stress_branch if has_stress else _branch
    metrics = ["energy_per_atom_std", "force_component_std"]
    if has_stress:
        metrics.append("stress_component_std")

    def branch(scale: float):
        return replace(
            factory(scale),
            correlations=tuple(
                {"metric": metric, "pearson": "0.5", "spearman": "0.6"}
                for metric in metrics
            ),
        )

    return {
        f"experiment_{index}": PlotRun(
            name=f"experiment_{index}",
            root=tmp_path / f"run_{index}",
            branches={
                "equal_weight": branch(1.0 + index * 0.1),
                "validation_weighted": branch(0.8 + index * 0.1),
            },
        )
        for index in range(4)
    }


@pytest.mark.parametrize(("has_stress", "logical_count"), [(False, 43), (True, 59)])
def test_render_dataset_figures_enforces_contract(
    tmp_path: Path, has_stress: bool, logical_count: int
) -> None:
    output = render_dataset_figures(
        _four_runs(tmp_path, has_stress),
        tmp_path / "figures",
        dataset="case",
        config=PlotConfig(dpi=36, grid_size=8),
    )

    audit = json.loads((output / "plot_audit.json").read_text(encoding="utf-8"))
    assert output == tmp_path / "figures" / "case"
    assert audit["dataset"] == "case"
    assert audit["logical_figure_count"] == logical_count
    assert len(tuple(output.rglob("*.png"))) == logical_count
    assert len(tuple(output.rglob("*.pdf"))) == logical_count
