from __future__ import annotations

from pathlib import Path

import torch

from Uncertainty_Quantification.FGE.fge.artifacts import (
    atomic_torch_save,
    atomic_write_json,
    sha256_file,
)
from Uncertainty_Quantification.FGE.fge.evaluation import evaluate_prediction
from Uncertainty_Quantification.FGE.fge.manifests import build_prediction_manifest
from Uncertainty_Quantification.FGE.fge.plot_style import PlotConfig
from Uncertainty_Quantification.FGE.fge.plot_workflow import render_all_figures
from Uncertainty_Quantification.FGE.fge.validation import validate_result
from Uncertainty_Quantification.FGE.tests.test_validation import _make_result


def _nonzero_formal_result(root: Path) -> Path:
    config = _make_result(root)
    prediction_path = root / "prediction" / "test_raw.pt"
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    payload["energy_reference"] = torch.tensor([1.0, 4.5], dtype=torch.float64)
    payload["forces_reference"] = torch.tensor(
        [[0.5, 0.1, -0.1], [3.5, -0.2, 0.2]], dtype=torch.float64
    )
    atomic_torch_save(prediction_path, payload)
    atomic_write_json(
        root / "prediction" / "manifest.json",
        build_prediction_manifest(
            root=root,
            prediction_path=prediction_path,
            member_count=2,
            structure_count=2,
            atom_count=2,
            observables=("energy", "forces"),
        ),
    )
    evaluate_prediction(config, root)
    validate_result(config, root)
    return root


def test_workflow_publishes_exact_contract_without_modifying_results(tmp_path: Path) -> None:
    roots = {
        f"experiment_{index}": _nonzero_formal_result(tmp_path / f"result_{index}")
        for index in range(4)
    }
    before = {
        label: sha256_file(root / "result_manifest.json") for label, root in roots.items()
    }

    final = render_all_figures(
        roots, tmp_path / "figures", config=PlotConfig(dpi=72, grid_size=16)
    )

    assert len(tuple(final.rglob("*.png"))) == 43
    assert len(tuple(final.rglob("*.pdf"))) == 43
    assert (final / "plot_audit.json").is_file()
    assert before == {
        label: sha256_file(root / "result_manifest.json") for label, root in roots.items()
    }
