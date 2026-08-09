from __future__ import annotations

import json
from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.plot_data import load_plot_run
from Uncertainty_Quantification.FGE.fge.validation import validate_result
from Uncertainty_Quantification.FGE.tests.test_validation import _make_result


def _formal_result(root: Path) -> Path:
    config = _make_result(root)
    validate_result(config, root)
    return root


def test_load_plot_run_reads_both_branches_in_canonical_units(tmp_path: Path) -> None:
    run = load_plot_run(_formal_result(tmp_path))

    assert set(run.branches) == {"equal_weight", "validation_weighted"}
    equal = run.branches["equal_weight"]
    assert equal.energy_reference.tolist() == [1.5, 4.0]
    assert equal.energy_prediction.tolist() == [1.5, 4.0]
    assert equal.energy_residual.tolist() == [0.0, 0.0]
    assert equal.energy_uncertainty.shape == (2,)
    assert equal.force_reference.shape == (6,)
    assert equal.force_prediction.shape == (6,)
    assert equal.force_residual.shape == (6,)
    assert equal.force_uncertainty.shape == (6,)
    assert equal.metrics["energy_per_atom"]["count"] == 2
    assert equal.correlations
    assert equal.risk_coverage


@pytest.mark.parametrize("marker", ["validation.json", "result_manifest.json"])
def test_load_plot_run_rejects_non_pass_formal_marker(tmp_path: Path, marker: str) -> None:
    root = _formal_result(tmp_path)
    path = root / marker
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "FAIL"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HardFailure, match="PASS"):
        load_plot_run(root)
