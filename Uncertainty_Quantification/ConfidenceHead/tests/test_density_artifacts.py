"""Tests for deterministic density-plot publication artifacts."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from confidence_head.density_artifacts import (
    PlotConflictError,
    build_audit_payload,
    density_csv_bytes,
    points_csv_bytes,
    publish_artifact_set,
    write_plot_manifest,
)
from confidence_head.density_plot import (
    DensityGrid,
    DensityMetrics,
    DensityPlotSettings,
    ErrorPairs,
)
from confidence_head.identity import sha256_file


@pytest.fixture
def pair_fixture() -> ErrorPairs:
    return ErrorPairs(
        expected=np.array([0.1, 0.2]),
        actual=np.array([0.05, 0.4]),
        expected_log10=np.log10([0.1, 0.2]),
        actual_log10=np.log10([0.05, 0.4]),
        sample_ids=["structure-0", "structure-1"],
        atom_indices=[3, 8],
        unit="eV/Angstrom",
        task="force",
        order=None,
        audit={
            "total_count": 3,
            "valid_count": 2,
            "excluded_nonfinite": 1,
            "excluded_nonpositive": 0,
        },
    )


@pytest.fixture
def grid_fixture() -> DensityGrid:
    return DensityGrid(
        density=np.array([[0.1, 0.2], [0.3, 0.4]]),
        x_centers=np.array([-2.0, -1.0]),
        y_centers=np.array([-3.0, -2.0]),
        contour_levels=(0.15, 0.35),
    )


def test_points_csv_has_stable_columns_and_all_valid_rows(pair_fixture: ErrorPairs) -> None:
    payload = points_csv_bytes(pair_fixture)
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    assert list(rows[0]) == [
        "dataset_index", "structure_id", "atom_index", "expected_error",
        "actual_error", "log10_expected_error", "log10_actual_error",
    ]
    assert [row["dataset_index"] for row in rows] == ["0", "1"]
    assert [row["structure_id"] for row in rows] == ["structure-0", "structure-1"]
    assert [row["atom_index"] for row in rows] == ["3", "8"]
    assert float(rows[1]["expected_error"]) == pytest.approx(0.2)


def test_density_csv_has_one_row_per_cell_and_contour_membership(
    grid_fixture: DensityGrid,
) -> None:
    rows = list(csv.DictReader(io.StringIO(density_csv_bytes(grid_fixture).decode("utf-8"))))
    assert list(rows[0]) == [
        "x_center", "y_center", "density", "contour_level",
        "grid_x_index", "grid_y_index",
    ]
    assert len(rows) == 4
    assert {(row["grid_x_index"], row["grid_y_index"]) for row in rows} == {
        ("0", "0"), ("0", "1"), ("1", "0"), ("1", "1"),
    }
    assert rows[-1]["contour_level"] == "0.35"


def test_audit_payload_contains_source_counts_metrics_settings_and_code_identity(
    pair_fixture: ErrorPairs,
    grid_fixture: DensityGrid,
) -> None:
    payload = build_audit_payload(
        dataset="matpes_test",
        source_manifest_sha256="a" * 64,
        pairs=pair_fixture,
        density=grid_fixture,
        metrics=DensityMetrics(2, 0.5, 0.6),
        settings=DensityPlotSettings(),
        code={"commit": "deadbeef", "dirty": False, "diff_sha256": None},
    )
    assert payload["source_manifest_sha256"] == "a" * 64
    assert payload["filtering"] == pair_fixture.audit
    assert payload["metrics"]["spearman_rho"] == pytest.approx(0.5)
    assert payload["unit"] == "eV/Angstrom"
    assert payload["plot_settings"]["grid_size"] == 160
    assert payload["code_identity"]["commit"] == "deadbeef"
    assert payload["log10_range"]["expected"] == pytest.approx([-1.0, -0.6989700043360187])


def test_publication_is_idempotent_and_rejects_partial_or_different_outputs(
    tmp_path: Path,
) -> None:
    files = {"plot.png": b"png-bytes", "plot.csv": b"a,b\n1,2\n"}
    published = publish_artifact_set(tmp_path, files)
    assert {path.name for path in published} == set(files)
    mtimes = {path.name: path.stat().st_mtime_ns for path in published}
    assert publish_artifact_set(tmp_path, files) == published
    assert {path.name: path.stat().st_mtime_ns for path in published} == mtimes

    with pytest.raises(PlotConflictError):
        publish_artifact_set(tmp_path, {**files, "plot.csv": b"changed"})
    assert (tmp_path / "plot.csv").read_bytes() == files["plot.csv"]

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "plot.png").write_bytes(files["plot.png"])
    with pytest.raises(PlotConflictError):
        publish_artifact_set(partial, files)
    assert not (partial / "plot.csv").exists()


def test_manifest_hashes_every_artifact_except_itself(tmp_path: Path) -> None:
    publish_artifact_set(tmp_path, {"a.png": b"png", "b.json": b"{}"})
    manifest = write_plot_manifest(
        tmp_path,
        dataset="mad_test",
        source_manifests={"force": "f" * 64},
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["artifacts"] == {
        "a.png": sha256_file(tmp_path / "a.png"),
        "b.json": sha256_file(tmp_path / "b.json"),
    }
    assert "plot_manifest.json" not in payload["artifacts"]
    with pytest.raises(PlotConflictError):
        write_plot_manifest(tmp_path, dataset="other", source_manifests={})
