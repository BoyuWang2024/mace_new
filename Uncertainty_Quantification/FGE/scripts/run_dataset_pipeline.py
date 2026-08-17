from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Uncertainty_Quantification.FGE.fge.config import load_config
from Uncertainty_Quantification.FGE.fge.dataset_evaluation import evaluate_dataset
from Uncertainty_Quantification.FGE.fge.dataset_prediction import predict_dataset
from Uncertainty_Quantification.FGE.fge.dataset_spec import DatasetSpec
from Uncertainty_Quantification.FGE.fge.derived_artifacts import DerivedLayout
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.plot_data import load_dataset_plot_run
from Uncertainty_Quantification.FGE.fge.plot_workflow import render_dataset_figures
from Uncertainty_Quantification.FGE.fge.wandb import create_wandb_logger


EXPERIMENT_CONFIGS = (
    "mace_fge_full_gpu_b64.yaml",
    "mace_fge_full_gpu_b64_lr1e-7_1e-6.yaml",
    "mace_fge_full_gpu_b64_lr1e-6_1e-5.yaml",
    "mace_fge_full_gpu_b64_lr1e-5_1e-4.yaml",
)


@dataclass(frozen=True)
class DatasetJob:
    label: str
    filename: str
    compute_stress: bool


DATASETS = (
    DatasetJob("matpes_test", "matpes_test.extxyz", True),
    DatasetJob("mad_test", "mad-test.xyz", False),
    DatasetJob("matpes_train", "matpes_train.extxyz", True),
)


@dataclass(frozen=True)
class PipelinePlan:
    config_dir: Path
    data_dir: Path
    outputs_root: Path
    figures_root: Path
    batch_size: int
    shard_size: int

    def __post_init__(self) -> None:
        for name in ("config_dir", "data_dir", "outputs_root", "figures_root"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in ("batch_size", "shard_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise HardFailure(f"{name} must be a positive integer")


def _json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HardFailure(f"{description} cannot be loaded") from exc
    if not isinstance(value, dict):
        raise HardFailure(f"{description} must be a mapping")
    return value


def _log_stage(
    config: Any,
    work_dir: Path,
    *,
    stage: str,
    status: str,
    experiment: str,
    dataset: str,
    warning_count: int,
    step: int,
    structure_count: int | None = None,
    atom_count: int | None = None,
    metrics: Mapping[str, float] | None = None,
) -> None:
    payload: dict[str, object] = {
        "stage": stage,
        "status": status,
        "experiment": experiment,
        "dataset": dataset,
        "warning_count": warning_count,
    }
    if structure_count is not None:
        payload["structure_count"] = structure_count
    if atom_count is not None:
        payload["atom_count"] = atom_count
    if metrics:
        payload.update(metrics)
    wandb_warnings: list[dict[str, str]] = []
    logger = create_wandb_logger(
        config.section("wandb"), Path(work_dir), warnings=wandb_warnings
    )
    logger.log(payload, step=step)
    logger.finish()


def _config(plan: PipelinePlan, filename: str):
    path = plan.config_dir / filename
    if not path.is_file():
        raise HardFailure(f"pipeline configuration is missing: {filename}")
    return load_config(path)


def _spec(plan: PipelinePlan, config: Any, dataset: DatasetJob) -> DatasetSpec:
    data = config.section("data")
    return DatasetSpec(
        label=dataset.label,
        path=plan.data_dir / dataset.filename,
        energy_key=data["energy_key"],
        forces_key=data["forces_key"],
        stress_key=data["stress_key"],
        head_name=data["head_name"],
        compute_stress=dataset.compute_stress,
        batch_size=plan.batch_size,
        shard_size=plan.shard_size,
    )


def predict_one(plan: PipelinePlan, filename: str, dataset: DatasetJob) -> Path:
    config = _config(plan, filename)
    layout = DerivedLayout(plan.outputs_root, config.project_name, dataset.label)
    try:
        manifest_path = predict_dataset(
            config, _spec(plan, config, dataset), plan.outputs_root
        )
        manifest = _json(manifest_path, "prediction manifest")
        shape = manifest.get("shape_symbols")
        if not isinstance(shape, dict):
            raise HardFailure("prediction manifest has no shape symbols")
        _log_stage(
            config,
            layout.root / "_work",
            stage="prediction",
            status="PASS",
            experiment=config.project_name,
            dataset=dataset.label,
            warning_count=0,
            structure_count=int(shape["S"]),
            atom_count=int(shape["A"]),
            step=0,
        )
        return manifest_path
    except Exception:
        _log_stage(
            config,
            layout.root / "_work",
            stage="prediction",
            status="ERROR",
            experiment=config.project_name,
            dataset=dataset.label,
            warning_count=0,
            step=0,
        )
        raise


def _rmse_metrics(audit: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    branches = audit.get("branches")
    if not isinstance(branches, Mapping):
        raise HardFailure("evaluation audit has no branches")
    for branch_name, prefix in (
        ("equal_weight", "equal"),
        ("validation_weighted", "weighted"),
    ):
        branch = branches.get(branch_name)
        if not isinstance(branch, Mapping) or not isinstance(
            branch.get("metrics"), Mapping
        ):
            raise HardFailure("evaluation audit branch metrics are missing")
        metrics = branch["metrics"]
        for metric_name, short_name in (
            ("energy_per_atom", "energy_rmse"),
            ("force_component", "force_rmse"),
            ("stress_component", "stress_rmse"),
        ):
            metric = metrics.get(metric_name)
            if metric is not None:
                result[f"{prefix}_{short_name}"] = float(metric["rmse"])
    return result


def evaluate_one(plan: PipelinePlan, filename: str, dataset: DatasetJob) -> Path:
    config = _config(plan, filename)
    layout = DerivedLayout(plan.outputs_root, config.project_name, dataset.label)
    try:
        audit_path = evaluate_dataset(config, layout)
        audit = _json(audit_path, "evaluation audit")
        warnings = audit.get("warnings")
        if not isinstance(warnings, list):
            raise HardFailure("evaluation audit warnings are invalid")
        equal = audit["branches"]["equal_weight"]["shape_symbols"]
        _log_stage(
            config,
            layout.root / "_work",
            stage="evaluation",
            status="PASS",
            experiment=config.project_name,
            dataset=dataset.label,
            warning_count=len(warnings),
            structure_count=int(equal["S"]),
            atom_count=int(equal["A"]),
            metrics=_rmse_metrics(audit),
            step=1,
        )
        return audit_path
    except Exception:
        _log_stage(
            config,
            layout.root / "_work",
            stage="evaluation",
            status="ERROR",
            experiment=config.project_name,
            dataset=dataset.label,
            warning_count=0,
            step=1,
        )
        raise


def plot_one(plan: PipelinePlan, dataset: DatasetJob) -> Path:
    configs = [_config(plan, filename) for filename in EXPERIMENT_CONFIGS]
    runs = {
        config.project_name: load_dataset_plot_run(
            DerivedLayout(plan.outputs_root, config.project_name, dataset.label).root
        )
        for config in configs
    }
    try:
        output = render_dataset_figures(
            runs, plan.figures_root, dataset=dataset.label
        )
        audit = _json(output / "plot_audit.json", "plot audit")
        for config in configs:
            layout = DerivedLayout(
                plan.outputs_root, config.project_name, dataset.label
            )
            _log_stage(
                config,
                layout.root / "_work",
                stage="plot",
                status="PASS",
                experiment=config.project_name,
                dataset=dataset.label,
                warning_count=0,
                metrics={
                    "logical_figure_count": float(
                        audit["logical_figure_count"]
                    )
                },
                step=2,
            )
        return output
    except Exception:
        for config in configs:
            layout = DerivedLayout(
                plan.outputs_root, config.project_name, dataset.label
            )
            _log_stage(
                config,
                layout.root / "_work",
                stage="plot",
                status="ERROR",
                experiment=config.project_name,
                dataset=dataset.label,
                warning_count=0,
                step=2,
            )
        raise


def run(plan: PipelinePlan) -> None:
    for dataset in DATASETS:
        for experiment in EXPERIMENT_CONFIGS:
            predict_one(plan, experiment, dataset)
    for dataset in DATASETS:
        for experiment in EXPERIMENT_CONFIGS:
            evaluate_one(plan, experiment, dataset)
    for dataset in DATASETS:
        plot_one(plan, dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sequential remote FGE dataset pipeline"
    )
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--figures-root", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--shard-size", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(
        PipelinePlan(
            config_dir=args.config_dir,
            data_dir=args.data_dir,
            outputs_root=args.outputs_root,
            figures_root=args.figures_root,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
