"""Orchestrate the exact nine-run CPU publication analysis and final manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..artifacts import atomic_json_dump
from ..identity import sha256_file
from ..plot_analysis import PlotConflictError
from .evaluate import load_evaluation_inputs, run_evaluate
from .plot_argmax_bin_boxplots import run_plot_argmax_bin_boxplots
from .plot_combined_argmax_bin_boxplots import (
    run_plot_combined_argmax_bin_boxplots,
)
from .plot_energy_correlations import run_plot_energy_correlations
from .production_matrix import discover_production_matrix


PLOT_MANIFEST_SCHEMA_VERSION = 1
PLOT_MANIFEST_FORMULA_VERSION = "mace_confidence_publication_plots_v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise PlotConflictError(f"plot manifest is unreadable: {error}") from error
    if type(value) is not dict:
        raise PlotConflictError("plot manifest must contain a JSON object")
    return value


def _publication_payload(force_config, energy_configs) -> dict[str, Any]:
    publication_root = (
        force_config.run.output_root / force_config.run.name_prefix
    )
    ordered = [
        ("force", None, force_config),
        *[
            ("energy", order, energy_configs[order])
            for order in range(1, 9)
        ],
    ]
    sources: list[dict[str, Any]] = []
    artifacts: dict[str, Path] = {}
    cache_id: str | None = None
    for role, order, config in ordered:
        inputs = load_evaluation_inputs(config)
        if cache_id is None:
            cache_id = inputs.identity["cache_id"]
        elif inputs.identity["cache_id"] != cache_id:
            raise PlotConflictError("publication source cache IDs differ")
        branch = "force" if role == "force" else "energy"
        sources.append(
            {
                "role": role,
                "cumulant_order": order,
                "identity": dict(inputs.identity),
                "run_root": inputs.run_root.relative_to(publication_root).as_posix(),
            }
        )
        paths = (
            inputs.run_dir / "test_predictions.pt",
            inputs.run_dir / "test_metrics.json",
            inputs.run_dir / "evaluation_manifest.json",
            inputs.run_root
            / "plots"
            / "argmax_bin_boxplots"
            / f"test_{branch}_argmax_bin_statistics.csv",
            inputs.run_root
            / "plots"
            / "argmax_bin_boxplots"
            / f"test_{branch}_argmax_bin_boxplot.png",
            inputs.run_root
            / "plots"
            / "argmax_bin_boxplots"
            / f"test_{branch}_argmax_bin_boxplot.pdf",
        )
        for path in paths:
            artifacts[path.relative_to(publication_root).as_posix()] = path

    comparison_root = publication_root / "comparisons"
    shared = (
        comparison_root
        / "energy_correlations"
        / "linear_energy_correlations.csv",
        comparison_root
        / "energy_correlations"
        / "linear_order_correlations_no_ci.png",
        comparison_root
        / "energy_correlations"
        / "linear_order_correlations_no_ci.pdf",
        comparison_root
        / "energy_correlations"
        / "comparison_metadata.json",
        comparison_root
        / "argmax_bin_boxplots"
        / "combined_force_argmax_bin_boxplots.pdf",
        comparison_root
        / "argmax_bin_boxplots"
        / "combined_energy_argmax_bin_boxplots.pdf",
    )
    for path in shared:
        artifacts[path.relative_to(publication_root).as_posix()] = path
    if len(artifacts) != 60:
        raise PlotConflictError("publication artifact inventory must contain 60 files")
    missing = [
        name
        for name, path in artifacts.items()
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise PlotConflictError(
            f"publication artifacts are missing or empty: {sorted(missing)}"
        )
    return {
        "schema_version": PLOT_MANIFEST_SCHEMA_VERSION,
        "formula_version": PLOT_MANIFEST_FORMULA_VERSION,
        "cache_id": cache_id,
        "sources": sources,
        "artifact_sha256": {
            name: sha256_file(path)
            for name, path in sorted(artifacts.items())
        },
    }


def run_plot_analysis_suite(config_dir: Path) -> Path:
    """Run the complete CPU test-analysis matrix and commit its manifest last."""
    force_config, energy_configs = discover_production_matrix(config_dir)
    ordered = [
        force_config,
        *(energy_configs[order] for order in range(1, 9)),
    ]
    for config in ordered:
        run_evaluate(config)
    for config in ordered:
        run_plot_argmax_bin_boxplots(config)
    run_plot_energy_correlations(energy_configs)
    run_plot_combined_argmax_bin_boxplots(force_config, energy_configs)

    publication_root = (
        force_config.run.output_root / force_config.run.name_prefix
    )
    manifest_path = publication_root / "comparisons" / "plot_manifest.json"
    expected = _publication_payload(force_config, energy_configs)
    if manifest_path.exists():
        if _read_json(manifest_path) != expected:
            raise PlotConflictError(
                "existing plot manifest differs from publication artifacts"
            )
        return manifest_path

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(manifest_path, expected)
    if _read_json(manifest_path) != expected:
        raise PlotConflictError("new plot manifest failed verification")
    return manifest_path
