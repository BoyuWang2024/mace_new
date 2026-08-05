"""Readout-only parameter guards and committed member serialization."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .artifacts import ExperimentLayout, atomic_torch_save
from .errors import HardFailure


def _is_readout(name: str) -> bool:
    return name.startswith("readouts.") or name.startswith("model.readouts.")


def _fingerprint(parameters: list[tuple[str, nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        tensor = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ReadoutGuard:
    model: nn.Module
    frozen_parameters: list[tuple[str, nn.Parameter]]
    trainable_parameters: list[tuple[str, nn.Parameter]]
    frozen_fingerprint: str

    def assert_frozen_unchanged(self) -> None:
        if _fingerprint(self.frozen_parameters) != self.frozen_fingerprint:
            raise HardFailure("frozen backbone drift detected")

    def assert_trainable_finite(self) -> None:
        for name, parameter in self.trainable_parameters:
            if not torch.isfinite(parameter).all().item():
                raise HardFailure(f"trainable parameter {name} contains NaN or Inf")


def freeze_readouts(model: nn.Module, expected_count: int = 2192) -> ReadoutGuard:
    """Freeze every parameter outside the exact MACE readout namespaces."""
    if not isinstance(model, nn.Module):
        raise HardFailure("model must be a torch module")
    frozen: list[tuple[str, nn.Parameter]] = []
    trainable: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_readout(name))
        (trainable if parameter.requires_grad else frozen).append((name, parameter))
    count = sum(parameter.numel() for _, parameter in trainable)
    if type(expected_count) is not int or count != expected_count:
        raise HardFailure(
            f"readout trainable parameter count {count} does not match {expected_count}"
        )
    guard = ReadoutGuard(model, frozen, trainable, _fingerprint(frozen))
    guard.assert_trainable_finite()
    return guard


def commit_member_pair(
    layout: ExperimentLayout,
    member_id: str,
    raw_model: nn.Module,
    ema_model: nn.Module,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit one aligned raw/EMA member pair to fixed formal paths."""
    if not re.fullmatch(r"member_[0-9]{2}", member_id):
        raise HardFailure("member_id must use member_NN format")
    for label, model in (("raw", raw_model), ("ema", ema_model)):
        if not isinstance(model, nn.Module):
            raise HardFailure(f"{label} member must be a torch module")
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all().item():
                raise HardFailure(f"{label} member contains NaN or Inf")
    raw_path = layout.training_dir / "members" / "raw" / f"{member_id}.model"
    ema_path = layout.training_dir / "members" / "ema" / f"{member_id}.model"
    atomic_torch_save(raw_path, raw_model)
    atomic_torch_save(ema_path, ema_model)
    result = {
        "member_id": member_id,
        "raw_path": raw_path,
        "ema_path": ema_path,
        "frozen_backbone_verified": True,
        **deepcopy(dict(metrics)),
    }
    return result
