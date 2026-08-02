"""Deterministic epoch trainer for cached continuous ConfidenceHead data."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .binning import BinningArtifact, labels_from_thresholds
from .cache import ContinuousBatch
from .checkpoint import EarlyStoppingState, TrainingState, save_best, save_last
from .config import ConfidenceHeadConfig
from .labels import energy_errors, force_errors
from .logging import TrainingLogger
from .losses import LossAccumulator, joint_loss


class ConfidenceTrainer:
    """Train enabled heads from complete cache batches at epoch boundaries.

    The workflow owns data factories and resume-directory decisions. This class
    owns only batch mathematics, strict best selection, durable event ordering,
    and rolling checkpoints.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: ConfidenceHeadConfig,
        binning: BinningArtifact,
        device: torch.device,
        logger: TrainingLogger,
        identity: Mapping[str, Any],
        best_path: Path,
        last_path: Path,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        if not isinstance(config, ConfidenceHeadConfig):
            raise TypeError("config must be a ConfidenceHeadConfig")
        if not isinstance(binning, BinningArtifact):
            raise TypeError("binning must be an immutable BinningArtifact")
        if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be a supported torch.device")
        if not isinstance(logger, TrainingLogger):
            raise TypeError("logger must be a TrainingLogger")
        if type(identity) is not dict or set(identity) != {
            "run_id",
            "experiment_id",
            "cache_id",
            "binning_id",
        }:
            raise ValueError("trainer identity schema differs")
        if any(not isinstance(value, str) or not value for value in identity.values()):
            raise ValueError("trainer identity values must be non-empty strings")

        model_force = getattr(model, "force_head", None) is not None
        model_energy = (
            getattr(model, "energy_adapter", None) is not None
            and getattr(model, "energy_head", None) is not None
        )
        if model_force != config.force_enabled or model_energy != config.energy_enabled:
            raise ValueError("model/config enabled-branch mismatch")
        enabled = {
            name
            for name, active in (
                ("force", config.force_enabled),
                ("energy", config.energy_enabled),
            )
            if active
        }
        if set(binning.branches) != enabled:
            raise ValueError("binning enabled-branch mismatch")
        if binning.algorithm != config.binning.algorithm:
            raise ValueError("binning algorithm mismatch")
        if (
            config.force_enabled
            and binning.force_target_mode != config.model.force.target_mode
        ):
            raise ValueError("binning force target-mode mismatch")
        if not config.force_enabled and binning.force_target_mode is not None:
            raise ValueError("disabled force binning must not have a target mode")
        artifact_identity = {
            "run_id": binning.run_id,
            "experiment_id": binning.experiment_id,
            "cache_id": binning.cache_id,
            "binning_id": binning.binning_id,
        }
        if artifact_identity != dict(identity):
            raise ValueError("binning/trainer identity mismatch")
        model_parameters = {id(parameter) for parameter in model.parameters()}
        optimizer_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if model_parameters != optimizer_parameters:
            raise ValueError("optimizer must own exactly the model parameters")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.binning = binning
        self.device = device
        self.logger = logger
        self.identity = dict(identity)
        self.best_path = Path(best_path)
        self.last_path = Path(last_path)

    def _move_batch(self, value: object) -> ContinuousBatch:
        if not isinstance(value, ContinuousBatch):
            raise ValueError("training batch must be a ContinuousBatch")
        tensors = {
            name: getattr(value, name)
            for name in (
                "structure_index",
                "atomic_numbers",
                "atom_offsets",
                "scalar_features",
                "force_prediction",
                "force_reference",
                "energy_prediction",
                "energy_reference",
            )
        }
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
            raise ValueError("batch data must contain tensors")
        floating = (
            tensors["scalar_features"],
            tensors["force_prediction"],
            tensors["force_reference"],
            tensors["energy_prediction"],
            tensors["energy_reference"],
        )
        if any(
            not tensor.is_floating_point()
            or not bool(torch.isfinite(tensor).all().item())
            for tensor in floating
        ):
            raise ValueError("batch floating data must be finite")
        offsets = tensors["atom_offsets"]
        if offsets.dtype is not torch.long or offsets.ndim != 1 or offsets.numel() < 2:
            raise ValueError("batch atom offsets are invalid")
        if int(offsets[0].item()) != 0 or not bool(
            torch.all(offsets[1:] > offsets[:-1]).item()
        ):
            raise ValueError("batch atom offsets are invalid")
        atoms = int(offsets[-1].item())
        structures = offsets.numel() - 1
        if (
            tensors["scalar_features"].ndim != 2
            or tensors["scalar_features"].shape[0] != atoms
        ):
            raise ValueError("batch feature shape differs from atom offsets")
        if tensors["force_prediction"].shape != (atoms, 3) or tensors[
            "force_reference"
        ].shape != (atoms, 3):
            raise ValueError("batch force shape differs from atom offsets")
        if tensors["energy_prediction"].shape != (structures,) or tensors[
            "energy_reference"
        ].shape != (structures,):
            raise ValueError("batch energy shape differs from atom offsets")
        return ContinuousBatch(
            structure_index=tensors["structure_index"].to(self.device),
            structure_id=value.structure_id,
            atomic_numbers=tensors["atomic_numbers"].to(self.device),
            atom_offsets=offsets.to(self.device),
            scalar_features=tensors["scalar_features"].to(self.device),
            force_prediction=tensors["force_prediction"].to(self.device),
            force_reference=tensors["force_reference"].to(self.device),
            energy_prediction=tensors["energy_prediction"].to(self.device),
            energy_reference=tensors["energy_reference"].to(self.device),
        )

    def _objective(self, batch: ContinuousBatch):
        outputs = self.model(batch.scalar_features, batch.atom_offsets)
        expected = {
            name
            for name, enabled in (
                ("force", self.config.force_enabled),
                ("energy", self.config.energy_enabled),
            )
            if enabled
        }
        if set(outputs) != expected:
            raise ValueError("model output enabled-branch mismatch")
        force_labels = None
        if self.config.force_enabled:
            errors = force_errors(
                batch.force_prediction,
                batch.force_reference,
                self.config.model.force.target_mode,
            )
            force_labels = (
                labels_from_thresholds(
                    errors.reshape(-1), self.binning.branches["force"].thresholds
                )
                .reshape(errors.shape)
                .to(self.device)
            )
        energy_labels = None
        if self.config.energy_enabled:
            counts = batch.atom_offsets[1:] - batch.atom_offsets[:-1]
            errors = energy_errors(
                batch.energy_prediction, batch.energy_reference, counts
            )
            energy_labels = labels_from_thresholds(
                errors, self.binning.branches["energy"].thresholds
            ).to(self.device)
        losses = joint_loss(
            outputs.get("force"),
            force_labels,
            outputs.get("energy"),
            energy_labels,
            force_coefficient=self.config.loss.force_coefficient,
            energy_coefficient=self.config.loss.energy_coefficient,
        )
        force_samples = 0
        if force_labels is not None:
            force_samples = int(force_labels.numel())
        energy_samples = 0 if energy_labels is None else int(energy_labels.numel())
        return losses, force_samples, energy_samples

    def _run_epoch(
        self, batches: Iterable[ContinuousBatch], *, training: bool
    ) -> dict[str, float | int]:
        self.model.train(training)
        accumulator = LossAccumulator()
        sample_counts = {"force": 0, "energy": 0}
        seen = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for raw_batch in batches:
                seen += 1
                batch = self._move_batch(raw_batch)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                losses, force_samples, energy_samples = self._objective(batch)
                if losses.total.ndim != 0 or not bool(
                    torch.isfinite(losses.total).item()
                ):
                    raise FloatingPointError("epoch total loss must be a finite scalar")
                if training:
                    losses.total.backward()
                    gradients = [
                        parameter.grad
                        for parameter in self.model.parameters()
                        if parameter.grad is not None
                    ]
                    if not gradients or any(
                        not bool(torch.isfinite(gradient).all().item())
                        for gradient in gradients
                    ):
                        raise FloatingPointError("training gradients must be finite")
                    self.optimizer.step()
                    if any(
                        (parameter.is_floating_point() or parameter.is_complex())
                        and not bool(torch.isfinite(parameter).all().item())
                        for parameter in self.model.parameters()
                    ):
                        raise FloatingPointError(
                            "optimizer produced non-finite parameters"
                        )
                if losses.force is not None:
                    accumulator.update("force", losses.force, force_samples)
                    sample_counts["force"] += force_samples
                if losses.energy is not None:
                    accumulator.update("energy", losses.energy, energy_samples)
                    sample_counts["energy"] += energy_samples
        if seen == 0:
            raise ValueError("epoch batch iterable is empty")
        force_loss = accumulator.mean("force") if self.config.force_enabled else 0.0
        energy_loss = accumulator.mean("energy") if self.config.energy_enabled else 0.0
        total = accumulator.total(
            self.config.loss.force_coefficient,
            self.config.loss.energy_coefficient,
        )
        if not all(math.isfinite(value) for value in (force_loss, energy_loss, total)):
            raise FloatingPointError("epoch metrics must be finite")
        return {
            "force_loss": force_loss,
            "energy_loss": energy_loss,
            "total_loss": total,
            "force_samples": sample_counts["force"],
            "energy_samples": sample_counts["energy"],
        }

    def train_epoch(self, batches: Iterable[ContinuousBatch]) -> dict[str, float | int]:
        """Train on every complete batch and return true-sample-weighted metrics."""
        return self._run_epoch(batches, training=True)

    def validate_epoch(
        self, batches: Iterable[ContinuousBatch]
    ) -> dict[str, float | int]:
        """Evaluate every batch without gradients or parameter updates."""
        return self._run_epoch(batches, training=False)

    def fit(
        self,
        train_batches_for_epoch: Callable[[int], Iterable[ContinuousBatch]],
        validation_batches: Callable[[], Iterable[ContinuousBatch]],
        *,
        state: TrainingState | None = None,
        _stop_after_completed_epochs: int | None = None,
        _stop_exception: type[BaseException] = RuntimeError,
    ) -> TrainingState:
        """Fit from the first uncommitted epoch and checkpoint each boundary."""
        current = TrainingState.initial() if state is None else state
        if not isinstance(current, TrainingState):
            raise TypeError("state must be a TrainingState")
        if current.completed:
            raise ValueError("training state is already completed")
        if current.next_epoch > self.config.trainer.max_epochs:
            raise ValueError("next_epoch exceeds configured max_epochs")
        expected_last_event = current.next_epoch - 1 if current.next_epoch else None
        if self.logger.local.last_epoch != expected_last_event:
            raise ValueError("event log does not match the checkpoint boundary")
        if _stop_after_completed_epochs is not None:
            if (
                type(_stop_after_completed_epochs) is not int
                or _stop_after_completed_epochs < 1
            ):
                raise ValueError("controlled stop boundary must be a positive integer")
            if not isinstance(_stop_exception, type) or not issubclass(
                _stop_exception, BaseException
            ):
                raise TypeError("controlled stop exception must be an exception type")

        for epoch in range(current.next_epoch, self.config.trainer.max_epochs):
            train_metrics = self.train_epoch(train_batches_for_epoch(epoch))
            validation_metrics = self.validate_epoch(validation_batches())
            early, improved = current.early_stopping.observe(
                epoch, float(validation_metrics["total_loss"])
            )
            completed = early.should_stop(
                self.config.trainer.early_stopping_patience
            ) or (epoch + 1 >= self.config.trainer.max_epochs)
            current = TrainingState(
                epoch + 1,
                early.best_epoch,
                early.best_validation_loss,
                early.bad_epochs,
                completed,
            )
            if improved:
                save_best(
                    self.best_path,
                    model=self.model,
                    identity=self.identity,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                )
            event = {
                "epoch": epoch,
                "train": dict(train_metrics),
                "validation": dict(validation_metrics),
                "improved": improved,
                "best_epoch": current.best_epoch,
                "best_validation_loss": current.best_validation_loss,
                "bad_epochs": current.bad_epochs,
                "completed": current.completed,
            }
            self.logger.append_epoch(event)
            save_last(
                self.last_path,
                model=self.model,
                optimizer=self.optimizer,
                identity=self.identity,
                validation_metrics=validation_metrics,
                state=current,
            )
            if (
                _stop_after_completed_epochs is not None
                and current.next_epoch >= _stop_after_completed_epochs
            ):
                raise _stop_exception()
            if completed:
                return current
        raise RuntimeError("fit reached no epoch boundary")
