from __future__ import annotations

import pytest

from Uncertainty_Quantification.FGE.scripts import plot_dataset


def test_plot_dataset_script_requires_explicit_results_and_output() -> None:
    with pytest.raises(SystemExit) as exc:
        plot_dataset.main([])

    assert exc.value.code == 2
    arguments = plot_dataset.build_parser().parse_args(
        [
            "--dataset-label",
            "matpes_test",
            "--result",
            "case",
            "/derived/case",
            "--output-root",
            "/figures",
        ]
    )
    assert arguments.result == [["case", "/derived/case"]]
