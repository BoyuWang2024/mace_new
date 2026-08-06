"""Tests for production-matrix discovery and cross-run publication plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest
import torch

from confidence_head.artifacts import load_torch_artifact
from confidence_head.cache import CacheWriter
from confidence_head.config import load_config
from confidence_head.evaluation_artifacts import (
    EvaluationPaths,
    commit_evaluation,
)
from confidence_head.identity import sha256_file
from confidence_head.metrics import branch_metrics, expected_errors, pearson_correlation
from confidence_head.workflows.check_training import run_check_training
from confidence_head.workflows.evaluate import (
    evaluation_input_hashes,
    load_evaluation_inputs,
    run_evaluate,
)
from confidence_head.workflows.fit_bins import run_fit_bins
from confidence_head.workflows.plot_argmax_bin_boxplots import (
    run_plot_argmax_bin_boxplots,
)
from confidence_head.workflows.plot_combined_argmax_bin_boxplots import (
    run_plot_combined_argmax_bin_boxplots,
)
from confidence_head.workflows.plot_energy_correlations import (
    run_plot_energy_correlations,
)
from confidence_head.workflows.production_matrix import (
    ProductionMatrixError,
    discover_production_matrix,
)
from confidence_head.workflows.train import run_train

from conftest import update_yaml, write_valid_config
from test_evaluate_workflow import _batch


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _write_exact_matrix_configs(tmp_path: Path) -> Path:
    base = write_valid_config(tmp_path, profile="production")
    force = tmp_path / "mace_matpes_full_force_only.yaml"
    force.write_bytes(base.read_bytes())
    update_yaml(force, _matrix_updates(order=1, branch="force"))
    for order in range(1, 9):
        energy = tmp_path / f"mace_matpes_full_energy_only_order{order}.yaml"
        energy.write_bytes(base.read_bytes())
        update_yaml(energy, _matrix_updates(order=order, branch="energy"))
    return tmp_path


def test_discover_production_matrix_requires_exact_branch_and_orders(
    tmp_path: Path,
) -> None:
    force, energy = discover_production_matrix(
        _write_exact_matrix_configs(tmp_path)
    )

    assert force.force_enabled and not force.energy_enabled
    assert list(energy) == list(range(1, 9))
    assert all(not config.force_enabled and config.energy_enabled for config in energy.values())
    assert [energy[order].model.energy.cumulant_order for order in energy] == list(
        range(1, 9)
    )
    assert {config.binning.algorithm for config in [force, *energy.values()]} == {
        "fixed_linear_v1"
    }


def test_discover_production_matrix_rejects_missing_or_extra_matrix_files(
    tmp_path: Path,
) -> None:
    for source in CONFIG_DIR.glob("mace_matpes_full_*.yaml"):
        shutil.copy2(source, tmp_path / source.name)
    (tmp_path / "mace_matpes_full_energy_only_order8.yaml").unlink()
    with pytest.raises(ProductionMatrixError, match="matrix files"):
        discover_production_matrix(tmp_path)

    shutil.copy2(
        CONFIG_DIR / "mace_matpes_full_energy_only_order8.yaml",
        tmp_path / "mace_matpes_full_energy_only_order8.yaml",
    )
    shutil.copy2(
        CONFIG_DIR / "mace_matpes_full_energy_only_order8.yaml",
        tmp_path / "mace_matpes_full_energy_only_order9.yaml",
    )
    with pytest.raises(ProductionMatrixError, match="matrix files"):
        discover_production_matrix(tmp_path)


def _matrix_updates(*, order: int, branch: str) -> dict[str, object]:
    return {
        "binning.force.num_bins": 3,
        "binning.force.max_error": 10.0,
        "binning.energy.num_bins": 3,
        "binning.energy.max_error": 10.0,
        "model.force.hidden_dims": [8],
        "model.energy.cumulant_order": order,
        "model.energy.projection_dim": 512,
        "model.energy.hidden_dims": [8],
        "model.energy.adapter_dropout": 0.0,
        "model.energy.dropout": 0.0,
        "loss.force_coefficient": 1.0 if branch == "force" else 0.0,
        "loss.energy_coefficient": 1.0 if branch == "energy" else 0.0,
        "trainer.batch_size": 2,
        "trainer.max_epochs": 1,
        "trainer.early_stopping_patience": 1,
        "runtime.device": "cpu",
        "logging.wandb": False,
        "logging.wandb_mode": "disabled",
        "run.name_prefix": "matrix",
    }


def _completed_matrix(tmp_path: Path):
    base = write_valid_config(tmp_path, profile="production")
    force_path = tmp_path / "mace_matpes_full_force_only.yaml"
    force_path.write_bytes(base.read_bytes())
    update_yaml(force_path, _matrix_updates(order=1, branch="force"))
    energy_paths: dict[int, Path] = {}
    for order in range(1, 9):
        path = tmp_path / f"mace_matpes_full_energy_only_order{order}.yaml"
        path.write_bytes(base.read_bytes())
        update_yaml(path, _matrix_updates(order=order, branch="energy"))
        energy_paths[order] = path
    force = load_config(force_path)
    energy = {order: load_config(path) for order, path in energy_paths.items()}

    from confidence_head.workflows import fit_bins as fit_workflow

    cache_id = fit_workflow._cache_identity(force)
    cache_root = force.run.output_root / force.run.name_prefix / "cache" / cache_id
    writer = CacheWriter(cache_root, cache_id=cache_id, shard_max_atoms=2)
    for split, atom_counts, energy_prediction in (
        ("train", (1, 2), (0.5, 1.5)),
        ("validation", (2,), (1.0,)),
        ("test", (1, 2), (2.0, 9.0)),
    ):
        writer.append(
            _batch(
                split,
                atom_counts=atom_counts,
                energy_prediction=energy_prediction,
                force_prediction=(
                    torch.tensor(
                        [[1.0, 2.0, 3.0], [3.0, 0.0, 3.0], [0.0, 1.0, 2.0]]
                    )
                    if split == "test"
                    else None
                ),
            ),
            split=split,
        )
        writer.finalize_split(split)
    writer.finalize()

    for config in [force, *(energy[order] for order in range(1, 9))]:
        run_fit_bins(config)
        run_train(config)
        run_check_training(config)
        run_evaluate(config)
        if config.energy_enabled:
            inputs = load_evaluation_inputs(config)
            paths = EvaluationPaths.from_run_root(inputs.run_root)
            predictions = load_torch_artifact(paths.predictions)
            logits = torch.tensor(
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
                dtype=predictions["energy"]["logits"].dtype,
            )
            representatives = inputs.binning.branches["energy"].representatives
            predictions["energy"]["logits"] = logits
            predictions["energy"]["expected_errors"] = expected_errors(
                logits, representatives
            )
            metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))
            metrics["branches"]["energy"] = branch_metrics(
                logits,
                predictions["energy"]["labels"],
                predictions["energy"]["errors"],
                representatives,
            )
            input_hashes = evaluation_input_hashes(inputs)
            for path in paths.outputs:
                path.unlink()
            commit_evaluation(
                paths,
                predictions,
                metrics,
                input_hashes,
            )
        run_plot_argmax_bin_boxplots(config)
    return force, energy


def test_cross_run_workflows_generate_correlations_metadata_and_combined_pdfs(
    tmp_path: Path,
) -> None:
    force, energy = _completed_matrix(tmp_path)

    csv_path = run_plot_energy_correlations(energy)
    force_pdf, energy_pdf = run_plot_combined_argmax_bin_boxplots(force, energy)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    correlation_dir = csv_path.parent
    metadata = json.loads(
        (correlation_dir / "comparison_metadata.json").read_text(encoding="utf-8")
    )

    assert [int(row["cumulant_order"]) for row in rows] == list(range(1, 9))
    assert set(rows[0]) == {
        "cumulant_order",
        "sample_count",
        "pearson_argmax_representative_vs_observed",
        "spearman_argmax_representative_vs_observed",
    }
    first_inputs = load_evaluation_inputs(energy[1])
    first_predictions_path = first_inputs.run_dir / "test_predictions.pt"
    first_predictions = load_torch_artifact(first_predictions_path)
    first_branch = first_predictions["energy"]
    representatives = first_inputs.binning.branches["energy"].representatives
    argmax_error = representatives[first_branch["logits"].argmax(dim=-1)]
    assert float(
        rows[0]["pearson_argmax_representative_vs_observed"]
    ) == pytest.approx(
        pearson_correlation(argmax_error, first_branch["errors"])
    )
    assert metadata["no_bootstrap_ci"] is True
    assert metadata["sources"][0]["prediction_sha256"] == sha256_file(
        first_predictions_path
    )
    assert not any("ci" in name.lower() for name in rows[0])
    assert (
        correlation_dir / "linear_order_correlations_no_ci.png"
    ).stat().st_size > 1000
    assert (
        correlation_dir / "linear_order_correlations_no_ci.pdf"
    ).stat().st_size > 1000
    assert force_pdf.name == "combined_force_argmax_bin_boxplots.pdf"
    assert energy_pdf.name == "combined_energy_argmax_bin_boxplots.pdf"
    assert force_pdf.read_bytes().startswith(b"%PDF")
    assert energy_pdf.read_bytes().startswith(b"%PDF")

    PdfReader = pytest.importorskip("PyPDF2").PdfReader
    assert len(PdfReader(str(force_pdf)).pages) == 1
    assert len(PdfReader(str(energy_pdf)).pages) == 8

    before = {
        path: path.stat().st_mtime_ns
        for path in (
            csv_path,
            correlation_dir / "linear_order_correlations_no_ci.png",
            correlation_dir / "linear_order_correlations_no_ci.pdf",
            correlation_dir / "comparison_metadata.json",
            force_pdf,
            energy_pdf,
        )
    }
    assert run_plot_energy_correlations(energy) == csv_path
    assert run_plot_combined_argmax_bin_boxplots(force, energy) == (
        force_pdf,
        energy_pdf,
    )
    assert before == {path: path.stat().st_mtime_ns for path in before}


def test_energy_correlation_rejects_missing_order_without_outputs(
    tmp_path: Path,
) -> None:
    base = write_valid_config(tmp_path)
    config = load_config(base)

    with pytest.raises(ProductionMatrixError, match="orders"):
        run_plot_energy_correlations({1: config})
    comparisons = config.run.output_root / config.run.name_prefix / "comparisons"
    assert not comparisons.exists()
