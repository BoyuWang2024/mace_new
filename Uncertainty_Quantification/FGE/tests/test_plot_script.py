from __future__ import annotations

import pytest

from Uncertainty_Quantification.FGE.scripts import plot


def test_plot_script_requires_explicit_results_and_output() -> None:
    with pytest.raises(SystemExit) as exc:
        plot.main([])
    assert exc.value.code == 2
    arguments = plot.build_parser().parse_args(
        ["--result", "case", "/result", "--output-root", "/figures"]
    )
    assert arguments.result == [["case", "/result"]]
    assert str(arguments.output_root) == "/figures"
