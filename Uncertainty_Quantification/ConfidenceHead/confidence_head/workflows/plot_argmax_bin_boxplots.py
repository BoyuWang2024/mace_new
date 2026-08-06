"""Generate one run's argmax-bin statistics and boxplot artifacts."""

from __future__ import annotations

from pathlib import Path

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
    render_argmax_bin_boxplot,
    statistics_csv_bytes,
    write_statistics_csv,
)
from .evaluate import evaluation_input_hashes, load_evaluation_inputs


def run_plot_argmax_bin_boxplots(config: ConfidenceHeadConfig) -> Path:
    """Generate or strictly reuse one enabled branch's CSV, PNG, and PDF."""
    inputs = load_evaluation_inputs(config)
    evaluation_paths = EvaluationPaths.from_run_root(inputs.run_root)
    if not validate_or_reuse_evaluation(
        evaluation_paths,
        inputs.identity,
        evaluation_input_hashes(inputs),
    ):
        raise PlotConflictError("complete test evaluation is required before plotting")
    predictions = validate_prediction_payload(
        load_torch_artifact(evaluation_paths.predictions),
        expected_identity=inputs.identity,
    )
    enabled = predictions["enabled_branches"]
    if len(enabled) != 1:
        raise PlotConflictError(
            "per-run publication plot requires exactly one enabled branch"
        )
    branch = enabled[0]
    rows = argmax_bin_statistics(
        predictions[branch]["logits"],
        predictions[branch]["errors"],
        inputs.binning.branches[branch],
    )
    output_dir = inputs.run_root / "plots" / "argmax_bin_boxplots"
    stem = output_dir / f"test_{branch}_argmax_bin_boxplot"
    csv_path = output_dir / f"test_{branch}_argmax_bin_statistics.csv"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    outputs = (csv_path, png_path, pdf_path)
    evidence = tuple(path.exists() for path in outputs)
    if any(evidence):
        if not all(evidence):
            raise PlotConflictError(
                "argmax-bin plot outputs are partial; preserve evidence and investigate"
            )
        if csv_path.read_bytes() != statistics_csv_bytes(rows):
            raise PlotConflictError("existing argmax-bin statistics differ")
        if png_path.stat().st_size == 0 or pdf_path.stat().st_size == 0:
            raise PlotConflictError("existing argmax-bin plot is empty")
        return csv_path

    output_dir.mkdir(parents=True, exist_ok=True)
    write_statistics_csv(csv_path, rows)
    render_argmax_bin_boxplot(
        stem,
        branch,
        rows,
        f"{inputs.run_root.name} / {inputs.binning.algorithm}",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        raise PlotConflictError("new argmax-bin outputs failed verification")
    return csv_path
