"""Build combined Force-only and Energy order 1-8 argmax-bin PDFs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

from ..artifacts import load_torch_artifact
from ..config import ConfidenceHeadConfig
from ..evaluation_artifacts import (
    EvaluationPaths,
    validate_or_reuse_evaluation,
    validate_prediction_payload,
)
from ..plot_analysis import (
    PlotConflictError,
    argmax_bin_statistics,
    build_argmax_bin_boxplot_figure,
    shared_boxplot_scale,
)
from .evaluate import evaluation_input_hashes, load_evaluation_inputs
from .plot_argmax_bin_boxplots import run_plot_argmax_bin_boxplots
from .plot_energy_correlations import _load_energy_evaluations
from .production_matrix import validate_energy_orders


matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _load_force(force_config: ConfidenceHeadConfig):
    if (
        not isinstance(force_config, ConfidenceHeadConfig)
        or not force_config.force_enabled
        or force_config.energy_enabled
    ):
        raise PlotConflictError("combined Force PDF requires a force-only config")
    inputs = load_evaluation_inputs(force_config)
    paths = EvaluationPaths.from_run_root(inputs.run_root)
    if not validate_or_reuse_evaluation(
        paths, inputs.identity, evaluation_input_hashes(inputs)
    ):
        raise PlotConflictError("force evaluation is missing")
    predictions = validate_prediction_payload(
        load_torch_artifact(paths.predictions),
        expected_identity=inputs.identity,
    )
    if predictions["enabled_branches"] != ("force",):
        raise PlotConflictError("force prediction branch differs")
    return inputs, predictions


def _temporary_pdf(destination: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".pdf",
        delete=False,
    ) as handle:
        return Path(handle.name)


def run_plot_combined_argmax_bin_boxplots(
    force_config: ConfidenceHeadConfig,
    energy_configs,
) -> tuple[Path, Path]:
    """Generate or strictly reuse the one-page Force and eight-page Energy PDFs."""
    configs = validate_energy_orders(energy_configs)
    force_inputs, force_predictions = _load_force(force_config)
    energy_loaded = _load_energy_evaluations(configs)
    if (
        force_config.run.output_root != configs[1].run.output_root
        or force_config.run.name_prefix != configs[1].run.name_prefix
        or force_inputs.identity["cache_id"]
        != energy_loaded[1][0].identity["cache_id"]
    ):
        raise PlotConflictError("force and energy comparison roots/cache differ")

    run_plot_argmax_bin_boxplots(force_config)
    for order in range(1, 9):
        run_plot_argmax_bin_boxplots(configs[order])

    force_rows = argmax_bin_statistics(
        force_predictions["force"]["logits"],
        force_predictions["force"]["errors"],
        force_inputs.binning.branches["force"],
    )
    energy_rows = {
        order: argmax_bin_statistics(
            energy_loaded[order][1]["energy"]["logits"],
            energy_loaded[order][1]["energy"]["errors"],
            energy_loaded[order][0].binning.branches["energy"],
        )
        for order in range(1, 9)
    }
    output_dir = (
        force_config.run.output_root
        / force_config.run.name_prefix
        / "comparisons"
        / "argmax_bin_boxplots"
    )
    force_pdf = output_dir / "combined_force_argmax_bin_boxplots.pdf"
    energy_pdf = output_dir / "combined_energy_argmax_bin_boxplots.pdf"
    evidence = (force_pdf.exists(), energy_pdf.exists())
    if any(evidence):
        if not all(evidence):
            raise PlotConflictError(
                "combined argmax-bin PDFs are partial; preserve evidence and investigate"
            )
        if force_pdf.stat().st_size == 0 or energy_pdf.stat().st_size == 0:
            raise PlotConflictError("existing combined argmax-bin PDF is empty")
        return force_pdf, energy_pdf

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_force = _temporary_pdf(force_pdf)
    temporary_energy = _temporary_pdf(energy_pdf)
    try:
        with PdfPages(temporary_force) as pages:
            figure, _ = build_argmax_bin_boxplot_figure(
                "force",
                force_rows,
                f"{force_inputs.run_root.name} / {force_inputs.binning.algorithm}",
            )
            pages.savefig(figure, bbox_inches="tight")
            plt.close(figure)

        energy_scale = shared_boxplot_scale(
            [energy_rows[order] for order in range(1, 9)]
        )
        with PdfPages(temporary_energy) as pages:
            for order in range(1, 9):
                inputs = energy_loaded[order][0]
                figure, _ = build_argmax_bin_boxplot_figure(
                    "energy",
                    energy_rows[order],
                    (
                        f"Cumulant order {order} / {inputs.run_root.name} "
                        f"/ {inputs.binning.algorithm}"
                    ),
                    scale=energy_scale,
                )
                pages.savefig(figure, bbox_inches="tight")
                plt.close(figure)

        for temporary, destination in (
            (temporary_force, force_pdf),
            (temporary_energy, energy_pdf),
        ):
            if temporary.stat().st_size == 0:
                raise PlotConflictError("new combined PDF is empty")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
    finally:
        temporary_force.unlink(missing_ok=True)
        temporary_energy.unlink(missing_ok=True)
    return force_pdf, energy_pdf
