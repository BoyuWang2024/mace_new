"""Publish continuous density plots from existing evaluation artifacts only."""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
from typing import Any

import matplotlib
import numpy as np

from ..artifacts import load_torch_artifact
from ..density_artifacts import (
    build_audit_payload,
    canonical_json_bytes,
    density_csv_bytes,
    points_csv_bytes,
    publish_artifact_set,
    write_plot_manifest,
)
from ..density_plot import (
    DensityMetrics,
    DensityPlotSettings,
    build_error_pairs,
    compute_density_grid,
    compute_log_metrics,
    render_density_plot,
)
from ..evaluation_artifacts import validate_prediction_payload
from ..external_config import ExternalInferenceConfig
from ..identity import code_identity, sha256_file
from .evaluate import load_evaluation_inputs
from .evaluate_external import _paths, resolve_head_config
from .production_matrix import discover_production_matrix


matplotlib.use("Agg")
from matplotlib import pyplot as plt


class DensitySuiteError(RuntimeError):
    """Existing evaluation inputs cannot produce a complete density suite."""


def density_plot_stems() -> dict[str, str]:
    return {
        "force": "force_expected_error_vs_actual_error",
        **{
            f"energy_order{order}": f"energy_order{order}_expected_error_vs_actual_error"
            for order in range(1, 9)
        },
    }


def _require_outputs(paths: Any) -> None:
    missing = [str(path) for path in paths.outputs if not path.is_file()]
    if missing:
        raise DensitySuiteError(
            "density plotting requires existing evaluation artifacts; missing: "
            + ", ".join(missing)
        )


def _atom_metadata(predictions: dict[str, Any]) -> tuple[list[str], list[int]]:
    ids = list(predictions["structure_ids"])
    offsets = predictions["structure_offsets"].tolist()
    sample_ids: list[str] = []
    atom_indices: list[int] = []
    for index, structure_id in enumerate(ids):
        count = int(offsets[index + 1] - offsets[index])
        sample_ids.extend([str(structure_id)] * count)
        atom_indices.extend(range(count))
    return sample_ids, atom_indices


def _pairs(branch: str, order: int | None, predictions: dict[str, Any]):
    values = predictions[branch]
    if branch == "force":
        sample_ids, atom_indices = _atom_metadata(predictions)
        unit = "eV/Angstrom"
    else:
        sample_ids = [str(value) for value in predictions["structure_ids"]]
        atom_indices = None
        unit = "eV/atom"
    return build_error_pairs(
        expected=values["expected_errors"],
        actual=values["errors"],
        sample_ids=sample_ids,
        atom_indices=atom_indices,
        unit=unit,
        task=branch,
        order=order,
    )


def _publish_group(
    root: Path,
    *,
    stem: str,
    dataset: str,
    manifest: Path,
    pairs,
    settings: DensityPlotSettings,
    repo_root: Path,
) -> DensityMetrics:
    metrics = compute_log_metrics(pairs.expected, pairs.actual)
    density = compute_density_grid(
        pairs.expected,
        pairs.actual,
        grid_size=settings.grid_size,
        sigma=settings.gaussian_sigma,
        contour_masses=settings.contour_masses,
        log_margin=settings.log_margin,
    )
    audit = build_audit_payload(
        dataset=dataset,
        source_manifest_sha256=sha256_file(manifest),
        pairs=pairs,
        density=density,
        metrics=metrics,
        settings=settings,
        code=asdict(code_identity(repo_root)),
        generated_at=datetime.fromtimestamp(manifest.stat().st_mtime, timezone.utc).isoformat(),
    )
    with tempfile.TemporaryDirectory(prefix="confidence-density-") as temporary:
        temporary_stem = Path(temporary) / stem
        png, pdf = render_density_plot(temporary_stem, pairs, density, metrics, settings)
        files = {
            f"{stem}.png": png.read_bytes(),
            f"{stem}.pdf": pdf.read_bytes(),
            f"{stem}_points.csv": points_csv_bytes(pairs),
            f"{stem}_density.csv": density_csv_bytes(density),
            f"{stem}.json": canonical_json_bytes(audit),
        }
    publish_artifact_set(root, files)
    return metrics


def _correlation_csv(metrics: dict[int, DensityMetrics]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("cumulant_order", "sample_count", "spearman_rho", "log10_pearson_r"))
    for order in range(1, 9):
        value = metrics[order]
        writer.writerow((order, value.valid_count, f"{value.spearman_rho:.16g}", f"{value.log10_pearson_r:.16g}"))
    return output.getvalue().encode("utf-8")


def _correlation_plot(metrics: dict[int, DensityMetrics], root: Path) -> tuple[bytes, bytes]:
    with tempfile.TemporaryDirectory(prefix="confidence-correlation-") as temporary:
        png = Path(temporary) / "summary.png"
        pdf = Path(temporary) / "summary.pdf"
        orders = list(range(1, 9))
        figure, axis = plt.subplots(figsize=(7, 5), dpi=300)
        axis.plot(orders, [metrics[o].spearman_rho for o in orders], "o-", label="Spearman rho")
        axis.plot(orders, [metrics[o].log10_pearson_r for o in orders], "s-", label="log10 Pearson r")
        axis.set_xlabel("Energy cumulant order")
        axis.set_ylabel("Correlation")
        axis.set_xticks(orders)
        axis.grid(True, color="0.85", linewidth=0.6)
        axis.legend()
        figure.tight_layout()
        figure.savefig(png, dpi=300)
        figure.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
        plt.close(figure)
        return png.read_bytes(), pdf.read_bytes()


def _publish_suite(
    *,
    dataset: str,
    plot_root: Path,
    loaded: dict[str, tuple[Any, dict[str, Any], Path]],
    repo_root: Path,
) -> Path:
    root = Path(plot_root) / dataset
    settings = DensityPlotSettings()
    energy_metrics: dict[int, DensityMetrics] = {}
    sources: dict[str, str] = {}
    for key, stem in density_plot_stems().items():
        _, predictions, manifest = loaded[key]
        order = None if key == "force" else int(key.removeprefix("energy_order"))
        branch = "force" if key == "force" else "energy"
        metrics = _publish_group(
            root,
            stem=stem,
            dataset=dataset,
            manifest=manifest,
            pairs=_pairs(branch, order, predictions),
            settings=settings,
            repo_root=repo_root,
        )
        sources[key] = sha256_file(manifest)
        if order is not None:
            energy_metrics[order] = metrics
    png, pdf = _correlation_plot(energy_metrics, root)
    publish_artifact_set(
        root,
        {
            "energy_orders_correlation.csv": _correlation_csv(energy_metrics),
            "energy_orders_correlation.png": png,
            "energy_orders_correlation.pdf": pdf,
        },
    )
    return write_plot_manifest(root, dataset=dataset, source_manifests=sources)


def run_external_density_suite(config: ExternalInferenceConfig, repo_root: Path) -> Path:
    loaded: dict[str, tuple[Any, dict[str, Any], Path]] = {}
    for key in density_plot_stems():
        training = resolve_head_config(config, key)
        inputs = load_evaluation_inputs(training)
        paths = _paths(config, key)
        _require_outputs(paths)
        manifest = paths.manifest
        predictions = validate_prediction_payload(
            load_torch_artifact(paths.predictions), expected_identity=inputs.identity
        )
        loaded[key] = (inputs, predictions, manifest)
    return _publish_suite(
        dataset=config.dataset.name,
        plot_root=config.plot_root / "density",
        loaded=loaded,
        repo_root=repo_root,
    )


def run_test_density_suite(
    config_dir: Path,
    plot_root: Path,
    repo_root: Path,
    dataset: str = "matpes_test",
) -> Path:
    force, energy = discover_production_matrix(config_dir)
    configs = {"force": force, **{f"energy_order{o}": energy[o] for o in range(1, 9)}}
    loaded: dict[str, tuple[Any, dict[str, Any], Path]] = {}
    for key, config in configs.items():
        inputs = load_evaluation_inputs(config)
        paths = type("Paths", (), {"outputs": (inputs.run_dir / "test_predictions.pt", inputs.run_dir / "test_metrics.json", inputs.run_dir / "evaluation_manifest.json")})
        _require_outputs(paths)
        manifest = inputs.run_dir / "evaluation_manifest.json"
        predictions = validate_prediction_payload(
            load_torch_artifact(inputs.run_dir / "test_predictions.pt"),
            expected_identity=inputs.identity,
        )
        loaded[key] = (inputs, predictions, manifest)
    return _publish_suite(
        dataset=dataset,
        plot_root=Path(plot_root) / "density",
        loaded=loaded,
        repo_root=repo_root,
    )
