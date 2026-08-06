"""End-to-end tests for the exact nine-run publication analysis suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from confidence_head.plot_analysis import PlotConflictError
from confidence_head.workflows.plot_analysis_suite import run_plot_analysis_suite

from test_plot_workflows import _completed_matrix


def test_suite_commits_hash_bound_manifest_last_and_reuses_it(
    tmp_path: Path,
) -> None:
    force, energy = _completed_matrix(tmp_path)

    manifest_path = run_plot_analysis_suite(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    before = {
        path: path.stat().st_mtime_ns
        for path in [manifest_path, *(Path(name) for name in [])]
    }

    assert manifest_path == (
        force.run.output_root
        / force.run.name_prefix
        / "comparisons"
        / "plot_manifest.json"
    )
    assert payload["schema_version"] == 1
    assert payload["formula_version"] == "mace_confidence_publication_plots_v1"
    assert len(payload["sources"]) == 9
    assert payload["sources"][0]["role"] == "force"
    assert [source["cumulant_order"] for source in payload["sources"][1:]] == list(
        range(1, 9)
    )
    assert len(payload["artifact_sha256"]) == 60
    assert all(len(digest) == 64 for digest in payload["artifact_sha256"].values())
    assert payload["cache_id"] == payload["sources"][0]["identity"]["cache_id"]
    assert {
        source["identity"]["cache_id"] for source in payload["sources"]
    } == {payload["cache_id"]}

    assert run_plot_analysis_suite(tmp_path) == manifest_path
    assert before == {path: path.stat().st_mtime_ns for path in before}
    assert list(energy) == list(range(1, 9))


def test_suite_detects_tampered_plot_against_existing_manifest(
    tmp_path: Path,
) -> None:
    force, _ = _completed_matrix(tmp_path)
    manifest_path = run_plot_analysis_suite(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = next(
        name
        for name in payload["artifact_sha256"]
        if name.endswith("test_force_argmax_bin_boxplot.png")
    )
    publication_root = force.run.output_root / force.run.name_prefix
    plot_path = publication_root / relative
    plot_path.write_bytes(plot_path.read_bytes() + b"tamper")
    evidence = plot_path.read_bytes()

    with pytest.raises(PlotConflictError, match="manifest|sha256|differs"):
        run_plot_analysis_suite(tmp_path)
    assert plot_path.read_bytes() == evidence


def test_suite_leaves_stage_outputs_without_manifest_when_final_commit_fails(
    tmp_path: Path, monkeypatch
) -> None:
    force, _ = _completed_matrix(tmp_path)
    import confidence_head.workflows.plot_analysis_suite as module

    def fail_manifest(*args, **kwargs):
        raise RuntimeError("injected final manifest failure")

    monkeypatch.setattr(module, "atomic_json_dump", fail_manifest)

    with pytest.raises(RuntimeError, match="injected"):
        run_plot_analysis_suite(tmp_path)

    comparison_root = (
        force.run.output_root / force.run.name_prefix / "comparisons"
    )
    assert (
        comparison_root
        / "energy_correlations"
        / "linear_energy_correlations.csv"
    ).is_file()
    assert (
        comparison_root
        / "argmax_bin_boxplots"
        / "combined_energy_argmax_bin_boxplots.pdf"
    ).is_file()
    assert not (comparison_root / "plot_manifest.json").exists()
