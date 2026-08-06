"""Compare Energy-only cumulant orders using argmax-bin representatives."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import torch

from ..artifacts import atomic_json_dump, load_torch_artifact
from ..evaluation_artifacts import (
    EvaluationPaths,
    validate_or_reuse_evaluation,
    validate_prediction_payload,
)
from ..identity import sha256_file
from ..metrics import pearson_correlation, spearman_correlation
from ..plot_analysis import PlotConflictError
from .evaluate import (
    EvaluationInputs,
    evaluation_input_hashes,
    load_evaluation_inputs,
)
from .production_matrix import validate_energy_orders


matplotlib.use("Agg")
from matplotlib import pyplot as plt


CORRELATION_FORMULA_VERSION = "mace_energy_order_correlation_v1"
CORRELATION_COLUMNS = (
    "cumulant_order",
    "sample_count",
    "pearson_argmax_representative_vs_observed",
    "spearman_argmax_representative_vs_observed",
)


def _load_energy_evaluations(
    energy_configs,
) -> dict[int, tuple[EvaluationInputs, dict[str, Any], Path]]:
    configs = validate_energy_orders(energy_configs)
    loaded: dict[int, tuple[EvaluationInputs, dict[str, Any], Path]] = {}
    reference_ids: tuple[str, ...] | None = None
    reference_offsets: torch.Tensor | None = None
    reference_errors: torch.Tensor | None = None
    cache_id: str | None = None
    for order in range(1, 9):
        inputs = load_evaluation_inputs(configs[order])
        paths = EvaluationPaths.from_run_root(inputs.run_root)
        if not validate_or_reuse_evaluation(
            paths, inputs.identity, evaluation_input_hashes(inputs)
        ):
            raise PlotConflictError(
                f"energy order {order} evaluation is missing"
            )
        predictions = validate_prediction_payload(
            load_torch_artifact(paths.predictions),
            expected_identity=inputs.identity,
        )
        if predictions["enabled_branches"] != ("energy",):
            raise PlotConflictError(
                f"energy order {order} prediction branch differs"
            )
        ids = predictions["structure_ids"]
        offsets = predictions["structure_offsets"]
        errors = predictions["energy"]["errors"]
        if reference_ids is None:
            reference_ids = ids
            reference_offsets = offsets
            reference_errors = errors
            cache_id = inputs.identity["cache_id"]
        elif (
            ids != reference_ids
            or not torch.equal(offsets, reference_offsets)
            or not torch.equal(errors, reference_errors)
            or inputs.identity["cache_id"] != cache_id
        ):
            raise PlotConflictError(
                f"energy order {order} test structure_ids/offsets/errors/cache differ"
            )
        loaded[order] = (inputs, predictions, paths.predictions)
    return loaded


def _row(
    order: int,
    inputs: EvaluationInputs,
    predictions: dict[str, Any],
) -> dict[str, int | float]:
    branch = predictions["energy"]
    representatives = inputs.binning.branches["energy"].representatives
    argmax_errors = representatives[branch["logits"].argmax(dim=-1)]
    try:
        pearson = pearson_correlation(argmax_errors, branch["errors"])
        spearman = spearman_correlation(argmax_errors, branch["errors"])
    except ValueError as error:
        raise PlotConflictError(
            f"energy order {order} correlation is undefined: {error}"
        ) from error
    return {
        "cumulant_order": order,
        "sample_count": int(branch["errors"].numel()),
        "pearson_argmax_representative_vs_observed": pearson,
        "spearman_argmax_representative_vs_observed": spearman,
    }


def _csv_bytes(rows: list[dict[str, int | float]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=CORRELATION_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                name: (
                    format(float(value), ".17g")
                    if isinstance(value, float)
                    else value
                )
                for name, value in row.items()
            }
        )
    return handle.getvalue().encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _render_correlation_plot(
    png_path: Path,
    pdf_path: Path,
    rows: list[dict[str, int | float]],
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    orders = [int(row["cumulant_order"]) for row in rows]
    axis.plot(
        orders,
        [float(row["pearson_argmax_representative_vs_observed"]) for row in rows],
        marker="o",
        linewidth=2,
        label="Pearson",
        color="#4C78A8",
    )
    axis.plot(
        orders,
        [float(row["spearman_argmax_representative_vs_observed"]) for row in rows],
        marker="s",
        linewidth=2,
        label="Tie-aware Spearman",
        color="#F58518",
    )
    axis.set_xticks(range(1, 9))
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlabel("Energy cumulant order")
    axis.set_ylabel("Correlation with observed per-atom Energy error")
    axis.set_title(
        "Argmax-bin representative correlation by cumulant order (no CI)"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    temporary_png: Path | None = None
    temporary_pdf: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=png_path.parent, suffix=".png", delete=False
        ) as handle:
            temporary_png = Path(handle.name)
        with tempfile.NamedTemporaryFile(
            dir=pdf_path.parent, suffix=".pdf", delete=False
        ) as handle:
            temporary_pdf = Path(handle.name)
        figure.savefig(temporary_png, dpi=300, format="png")
        figure.savefig(temporary_pdf, format="pdf")
        for temporary, destination in (
            (temporary_png, png_path),
            (temporary_pdf, pdf_path),
        ):
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
    finally:
        plt.close(figure)
        if temporary_png is not None:
            temporary_png.unlink(missing_ok=True)
        if temporary_pdf is not None:
            temporary_pdf.unlink(missing_ok=True)


def run_plot_energy_correlations(energy_configs) -> Path:
    """Generate or strictly reuse order 1-8 Energy correlation artifacts."""
    configs = validate_energy_orders(energy_configs)
    loaded = _load_energy_evaluations(configs)
    rows = [
        _row(order, loaded[order][0], loaded[order][1])
        for order in range(1, 9)
    ]
    output_dir = (
        configs[1].run.output_root
        / configs[1].run.name_prefix
        / "comparisons"
        / "energy_correlations"
    )
    csv_path = output_dir / "linear_energy_correlations.csv"
    png_path = output_dir / "linear_order_correlations_no_ci.png"
    pdf_path = output_dir / "linear_order_correlations_no_ci.pdf"
    metadata_path = output_dir / "comparison_metadata.json"
    metadata = {
        "schema_version": 1,
        "formula_version": CORRELATION_FORMULA_VERSION,
        "definition": "argmax_bin_representative_vs_observed_per_atom_energy_error",
        "no_bootstrap_ci": True,
        "cache_id": loaded[1][0].identity["cache_id"],
        "structure_count": len(loaded[1][1]["structure_ids"]),
        "sources": [
            {
                "cumulant_order": order,
                "identity": dict(loaded[order][0].identity),
                "prediction_sha256": sha256_file(loaded[order][2]),
            }
            for order in range(1, 9)
        ],
    }
    outputs = (csv_path, png_path, pdf_path, metadata_path)
    evidence = tuple(path.exists() for path in outputs)
    if any(evidence):
        if not all(evidence):
            raise PlotConflictError(
                "energy correlation outputs are partial; preserve evidence and investigate"
            )
        if csv_path.read_bytes() != _csv_bytes(rows):
            raise PlotConflictError("existing energy correlation CSV differs")
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise PlotConflictError(
                f"existing energy correlation metadata is unreadable: {error}"
            ) from error
        if existing_metadata != metadata:
            raise PlotConflictError("existing energy correlation metadata differs")
        if png_path.stat().st_size == 0 or pdf_path.stat().st_size == 0:
            raise PlotConflictError("existing energy correlation plot is empty")
        return csv_path

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(csv_path, _csv_bytes(rows))
    atomic_json_dump(metadata_path, metadata)
    _render_correlation_plot(png_path, pdf_path, rows)
    if not all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        raise PlotConflictError("new energy correlation outputs failed verification")
    return csv_path
