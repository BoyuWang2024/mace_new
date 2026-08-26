"""Contract tests for the density-suite workflow."""
from pathlib import Path

import pytest
import torch

from confidence_head.workflows import plot_density_suite as workflow
from confidence_head.workflows.plot_density_suite import (
    DensitySuiteError,
    density_plot_stems,
    publish_density_suite_from_predictions,
)


def _identity() -> dict[str, str]:
    return {
        "run_id": "a" * 64,
        "experiment_id": "b" * 64,
        "cache_id": "c" * 64,
        "binning_id": "d" * 64,
    }


def _predictions(branch: str) -> dict[str, object]:
    samples = 2 if branch == "force" else 1
    return {
        "schema_version": 1,
        "formula_version": "mace_confidence_head_test_evaluation_v1",
        "split": "test",
        "identity": _identity(),
        "enabled_branches": (branch,),
        "structure_ids": ("sample#0",),
        "structure_offsets": torch.tensor([0, 2], dtype=torch.int64),
        "force_target_mode": "atom_mean" if branch == "force" else None,
        branch: {
            "logits": torch.tensor([[2.0, 1.0]] * samples),
            "labels": torch.zeros(samples, dtype=torch.int64),
            "errors": torch.full((samples,), 0.1, dtype=torch.float64),
            "expected_errors": torch.full(
                (samples,), 0.2, dtype=torch.float64
            ),
        },
    }


def _loaded(tmp_path: Path) -> dict[str, tuple[None, dict[str, object], Path]]:
    result = {}
    for key in density_plot_stems():
        manifest = tmp_path / "sources" / key / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}", encoding="utf-8")
        branch = "force" if key == "force" else "energy"
        result[key] = (None, _predictions(branch), manifest)
    return result


def test_density_plot_stems_cover_force_and_all_energy_orders() -> None:
    assert density_plot_stems() == {
        "force": "force_expected_error_vs_actual_error",
        **{
            f"energy_order{order}": f"energy_order{order}_expected_error_vs_actual_error"
            for order in range(1, 9)
        },
    }


@pytest.mark.parametrize("selection", ["full", "force", "energy"])
def test_publish_density_suite_from_predictions_accepts_complete_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
) -> None:
    loaded = _loaded(tmp_path)
    if selection == "force":
        loaded = {"force": loaded["force"]}
    elif selection == "energy":
        loaded = {
            key: value for key, value in loaded.items() if key != "force"
        }
    calls = []

    def fake_publish(**kwargs):
        calls.append(kwargs)
        return kwargs["plot_root"] / kwargs["dataset"] / "plot_manifest.json"

    monkeypatch.setattr(workflow, "_publish_suite", fake_publish)

    manifest = publish_density_suite_from_predictions(
        dataset="mad_r2scan",
        plot_root=tmp_path / "plots",
        loaded=loaded,
        repo_root=tmp_path,
    )

    assert manifest == tmp_path / "plots" / "mad_r2scan" / "plot_manifest.json"
    assert len(calls) == 1
    assert set(calls[0]["loaded"]) == set(loaded)


def test_publish_density_suite_from_predictions_rejects_partial_energy_group(
    tmp_path: Path,
) -> None:
    loaded = _loaded(tmp_path)
    loaded.pop("force")
    loaded.pop("energy_order8")

    with pytest.raises(DensitySuiteError, match="key set differs"):
        publish_density_suite_from_predictions(
            dataset="mad_r2scan",
            plot_root=tmp_path / "plots",
            loaded=loaded,
            repo_root=tmp_path,
        )
