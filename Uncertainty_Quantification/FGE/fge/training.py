"""Fixed readout-only FGE training with one optimizer and one global EMA."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_ema import ExponentialMovingAverage

from mace.modules.loss import WeightedHuberEnergyForcesStressLoss
from mace.tools.train import evaluate, take_step

from .artifacts import ExperimentLayout, atomic_write_json
from .data import build_mace_loaders, load_extxyz
from .errors import HardFailure
from .manifests import build_training_manifest
from .metric_views import rmse_only
from .members import ReadoutGuard, commit_member_pair, freeze_readouts
from .preflight import _load_model, run_preflight
from .schedule import AsymmetricTriangularLR
from .wandb import create_wandb_logger


@dataclass
class _MaceRuntime:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    ema: ExponentialMovingAverage
    train_loader: Any
    validation_loader: Any
    loss_fn: nn.Module
    output_args: dict[str, bool]
    max_grad_norm: float
    device: torch.device
    guard: ReadoutGuard

    def take_step(self, batch: Any) -> float:
        loss, _ = take_step(
            self.model,
            self.loss_fn,
            batch,
            self.optimizer,
            self.ema,
            self.output_args,
            self.max_grad_norm,
            self.device,
        )
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise HardFailure("training loss contains NaN or Inf")
        return value

    def evaluate(self) -> dict[str, float]:
        loss, auxiliary = evaluate(
            self.model,
            self.loss_fn,
            self.validation_loader,
            self.output_args,
            self.device,
        )
        result = {
            "loss": float(loss),
            "energy_rmse": float(auxiliary["rmse_e_per_atom"]),
            "forces_rmse": float(auxiliary["rmse_f"]),
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise HardFailure("validation metrics contain NaN or Inf")
        return result


def _model_scalar(model: nn.Module, name: str) -> float:
    value = getattr(model, name, None)
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (int, float)):
        return float(value)
    raise HardFailure(f"base model has no usable {name}")


def _build_runtime(config: Any) -> tuple[_MaceRuntime, Path]:
    paths = config.section("paths")
    training = config.section("training")
    data_config = config.section("data")
    base_path = Path(paths["base_checkpoint"])
    device = torch.device(training["device"])
    model = _load_model(base_path, str(device)).to(device)
    first_parameter = next(model.parameters())
    torch.set_default_dtype(first_parameter.dtype)
    guard = freeze_readouts(model, training["expected_readout_parameter_count"])
    keys = {
        "energy": data_config["energy_key"],
        "forces": data_config["forces_key"],
        "stress": data_config["stress_key"],
        "head": data_config["head_name"],
    }
    train_configurations = load_extxyz(
        Path(paths["train_data"]), keys=keys, required={"energy", "forces", "stress"}, head_name=data_config["head_name"]
    )
    validation_configurations = load_extxyz(
        Path(paths["val_data"]), keys=keys, required={"energy", "forces", "stress"}, head_name=data_config["head_name"]
    )
    atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    cutoff = _model_scalar(model, "r_max")
    heads = list(getattr(model, "heads", [data_config["head_name"]]))
    train_loader = build_mace_loaders(
        train_configurations,
        atomic_numbers=atomic_numbers,
        cutoff=cutoff,
        batch_size=training["batch_size"],
        shuffle=True,
        heads=heads,
    )
    validation_loader = build_mace_loaders(
        validation_configurations,
        atomic_numbers=atomic_numbers,
        cutoff=cutoff,
        batch_size=training["validation_batch_size"],
        shuffle=False,
        heads=heads,
    )
    loss_fn = WeightedHuberEnergyForcesStressLoss(
        energy_weight=training["energy_weight"],
        forces_weight=training["forces_weight"],
        stress_weight=training["stress_weight"],
        huber_delta=training["huber_delta"],
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in guard.trainable_parameters],
        lr=config.section("fge")["lr_min"],
        weight_decay=training["weight_decay"],
        betas=(training["beta"], 0.999),
        amsgrad=training["amsgrad"],
    )
    ema = ExponentialMovingAverage(model.parameters(), decay=config.section("ema")["decay"])
    runtime = _MaceRuntime(
        model=model,
        optimizer=optimizer,
        ema=ema,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_fn=loss_fn,
        output_args={"energy": True, "forces": True, "virials": False, "stress": True, "dipoles": False},
        max_grad_norm=training["max_grad_norm"],
        device=device,
        guard=guard,
    )
    return runtime, base_path


def _quality_warnings(
    member_id: str,
    metrics: dict[str, float],
    base: dict[str, float],
    quality: Any,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for metric in ("energy_rmse", "forces_rmse"):
        ratio = metrics[metric] / base[metric] if base[metric] > 0 else float("inf")
        threshold = None
        code = None
        if ratio >= quality["collapsed_rmse_multiplier"]:
            threshold, code = quality["collapsed_rmse_multiplier"], "collapsed_rmse"
        elif ratio >= quality["weak_rmse_multiplier"]:
            threshold, code = quality["weak_rmse_multiplier"], "weak_rmse"
        if code:
            warnings.append(
                {
                    "code": code,
                    "member_id": member_id,
                    "metric": metric,
                    "value": metrics[metric],
                    "base": base[metric],
                    "ratio": ratio,
                    "threshold": threshold,
                }
            )
    return warnings


def train_fge(config: Any) -> Path:
    """Train all FGE cycles while preserving optimizer and EMA state globally."""
    run_preflight(config, "train")
    layout = ExperimentLayout(config.output_dir)
    runtime, base_path = _build_runtime(config)
    guard = getattr(runtime, "guard", None) or freeze_readouts(
        runtime.model, config.section("training")["expected_readout_parameter_count"]
    )
    training = config.section("training")
    fge = config.section("fge")
    steps_per_cycle = len(runtime.train_loader) * training["epochs_per_cycle"]
    schedule = AsymmetricTriangularLR(
        steps_per_cycle, fge["lr_min"], fge["lr_max"], fge["rise_fraction"]
    )
    warnings: list[dict[str, Any]] = []
    logger = create_wandb_logger(
        config.section("wandb"), layout.work_dir, warnings=warnings
    )
    base_metrics = runtime.evaluate()
    members: list[dict[str, Any]] = []
    global_step = 0
    try:
        for cycle in range(1, training["member_count"] + 1):
            for _epoch in range(training["epochs_per_cycle"]):
                for batch in runtime.train_loader:
                    lr = schedule.value(global_step % steps_per_cycle)
                    for group in runtime.optimizer.param_groups:
                        group["lr"] = lr
                    loss = runtime.take_step(batch)
                    guard.assert_trainable_finite()
                    logger.log({"loss": loss, "lr": lr, "cycle": cycle}, global_step)
                    global_step += 1
            guard.assert_frozen_unchanged()
            raw_metrics = runtime.evaluate()
            raw_model = deepcopy(runtime.model).to("cpu")
            with runtime.ema.average_parameters():
                guard.assert_frozen_unchanged()
                guard.assert_trainable_finite()
                ema_metrics = runtime.evaluate()
                ema_model = deepcopy(runtime.model).to("cpu")
            member_id = f"member_{cycle:02d}"
            warnings.extend(
                _quality_warnings(
                    member_id, raw_metrics, base_metrics, config.section("quality")
                )
            )
            member = commit_member_pair(
                layout,
                member_id,
                raw_model,
                ema_model,
                {"raw_metrics": raw_metrics, "ema_metrics": ema_metrics},
            )
            member["cycle"] = cycle
            members.append(member)
            atomic_write_json(
                layout.work_dir / "resume" / "committed_members.json",
                {"members": [{"member_id": item["member_id"], "cycle": item["cycle"]} for item in members]},
            )
    finally:
        logger.finish()
    manifest = build_training_manifest(
        root=layout.root,
        project_name=config.project_name,
        k_requested=training["member_count"],
        base_model_path=base_path,
        base_model_metrics=rmse_only(base_metrics),
        members=members,
        warnings=warnings,
    )
    atomic_write_json(layout.training_manifest, manifest)
    return layout.training_manifest
