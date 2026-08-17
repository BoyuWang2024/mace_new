from __future__ import annotations

from pathlib import Path

from Uncertainty_Quantification.FGE.fge.dataset_evaluation import evaluate_dataset
from Uncertainty_Quantification.FGE.fge.plot_data import load_dataset_plot_run
from Uncertainty_Quantification.FGE.tests.test_dataset_evaluation import _prepare


def test_load_dataset_plot_run_concatenates_shards_with_stress(
    tmp_path: Path,
) -> None:
    config, layout = _prepare(tmp_path)
    evaluate_dataset(config, layout)

    run = load_dataset_plot_run(layout.root)

    assert run.name == "mace_fge"
    assert run.has_stress
    for branch in run.branches.values():
        assert branch.energy_reference.shape == (2,)
        assert branch.force_reference.shape == (6,)
        assert branch.stress_reference is not None
        assert branch.stress_reference.shape == (18,)
        assert branch.stress_uncertainty is not None
        assert branch.stress_uncertainty.shape == (18,)
