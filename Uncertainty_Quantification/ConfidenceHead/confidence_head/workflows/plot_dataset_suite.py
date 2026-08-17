"""Publish stable per-dataset plots from test or external evaluations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

from ..artifacts import atomic_json_dump, load_torch_artifact
from ..evaluation_artifacts import validate_prediction_payload
from ..external_config import ExternalInferenceConfig
from ..identity import sha256_file
from ..plot_analysis import (
    PlotConflictError,
    argmax_bin_statistics,
    build_argmax_bin_boxplot_figure,
    render_argmax_bin_boxplot,
    shared_boxplot_scale,
    write_statistics_csv,
)
from .evaluate import load_evaluation_inputs
from .evaluate_external import _paths, resolve_head_config, run_evaluate_external
from .plot_analysis_suite import run_plot_analysis_suite
from .plot_energy_correlations import _csv_bytes, _render_correlation_plot, _row
from .production_matrix import discover_production_matrix


matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def dataset_plot_stems() -> dict[str, str]:
    return {
        "force": "force_atom_mean_argmax_bin_boxplot",
        **{
            f"energy_order{order}": f"energy_order{order}_argmax_bin_boxplot"
            for order in range(1, 9)
        },
    }


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise PlotConflictError(f"existing plot differs: {destination}")
        return
    shutil.copy2(source, destination)


def _manifest(root: Path, dataset_name: str, sources: dict[str, Path]) -> Path:
    manifest_path = root / "plot_manifest.json"
    artifacts = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path != manifest_path
    }
    payload = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "sources": {name: sha256_file(path) for name, path in sorted(sources.items())},
        "artifact_sha256": artifacts,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != payload:
            raise PlotConflictError("existing dataset plot manifest differs")
        return manifest_path
    atomic_json_dump(manifest_path, payload)
    return manifest_path


def run_plot_test_dataset_suite(
    config_dir: Path, plot_root: Path, dataset_name: str = "matpes_test"
) -> Path:
    run_plot_analysis_suite(config_dir)
    force, energy = discover_production_matrix(config_dir)
    root = Path(plot_root) / dataset_name
    stems = dataset_plot_stems()
    sources: dict[str, Path] = {}
    ordered = [("force", force), *[(f"energy_order{o}", energy[o]) for o in range(1, 9)]]
    for key, config in ordered:
        inputs = load_evaluation_inputs(config)
        branch = "force" if key == "force" else "energy"
        source_dir = inputs.run_root / "plots" / "argmax_bin_boxplots"
        for suffix in ("png", "pdf"):
            source = source_dir / f"test_{branch}_argmax_bin_boxplot.{suffix}"
            _copy(source, root / f"{stems[key]}.{suffix}")
        source_csv = source_dir / f"test_{branch}_argmax_bin_statistics.csv"
        _copy(source_csv, root / f"{stems[key]}_statistics.csv")
        sources[key] = inputs.run_dir / "test_predictions.pt"
    comparison = force.run.output_root / force.run.name_prefix / "comparisons"
    correlation = comparison / "energy_correlations"
    _copy(correlation / "linear_order_correlations_no_ci.png", root / "energy_orders_comparison.png")
    _copy(correlation / "linear_order_correlations_no_ci.pdf", root / "energy_orders_comparison.pdf")
    _copy(correlation / "linear_energy_correlations.csv", root / "energy_orders_comparison.csv")
    combined = comparison / "argmax_bin_boxplots"
    _copy(combined / "combined_force_argmax_bin_boxplots.pdf", root / "force_summary.pdf")
    _copy(combined / "combined_energy_argmax_bin_boxplots.pdf", root / "energy_summary.pdf")
    return _manifest(root, dataset_name, sources)


def run_plot_dataset_suite(config: ExternalInferenceConfig) -> Path:
    root = config.plot_root / config.dataset.name
    root.mkdir(parents=True, exist_ok=True)
    stems = dataset_plot_stems()
    loaded: dict[str, tuple[Any, dict[str, Any], Path]] = {}
    for key in stems:
        manifest = run_evaluate_external(config, key)
        training = resolve_head_config(config, key)
        inputs = load_evaluation_inputs(training)
        paths = _paths(config, key)
        predictions = validate_prediction_payload(
            load_torch_artifact(paths.predictions), expected_identity=inputs.identity
        )
        branch = "force" if key == "force" else "energy"
        rows = argmax_bin_statistics(
            predictions[branch]["logits"],
            predictions[branch]["errors"],
            inputs.binning.branches[branch],
        )
        stem = root / stems[key]
        write_statistics_csv(root / f"{stems[key]}_statistics.csv", rows)
        render_argmax_bin_boxplot(
            stem, branch, rows, f"{config.dataset.name} / {key}"
        )
        loaded[key] = (inputs, predictions, manifest)

    energy_rows = [
        _row(order, loaded[f"energy_order{order}"][0], loaded[f"energy_order{order}"][1])
        for order in range(1, 9)
    ]
    (root / "energy_orders_comparison.csv").write_bytes(_csv_bytes(energy_rows))
    _render_correlation_plot(
        root / "energy_orders_comparison.png",
        root / "energy_orders_comparison.pdf",
        energy_rows,
    )
    force_inputs, force_predictions, _ = loaded["force"]
    force_rows = argmax_bin_statistics(
        force_predictions["force"]["logits"],
        force_predictions["force"]["errors"],
        force_inputs.binning.branches["force"],
    )
    with PdfPages(root / "force_summary.pdf") as pages:
        figure, _ = build_argmax_bin_boxplot_figure("force", force_rows, config.dataset.name)
        pages.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    rows_by_order = {
        order: argmax_bin_statistics(
            loaded[f"energy_order{order}"][1]["energy"]["logits"],
            loaded[f"energy_order{order}"][1]["energy"]["errors"],
            loaded[f"energy_order{order}"][0].binning.branches["energy"],
        )
        for order in range(1, 9)
    }
    scale = shared_boxplot_scale(list(rows_by_order.values()))
    with PdfPages(root / "energy_summary.pdf") as pages:
        for order in range(1, 9):
            figure, _ = build_argmax_bin_boxplot_figure(
                "energy", rows_by_order[order], f"{config.dataset.name} / order {order}", scale=scale
            )
            pages.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    return _manifest(root, config.dataset.name, {key: value[2] for key, value in loaded.items()})
