"""MACE readout-only parameter policy."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import HardFailure


@dataclass(frozen=True)
class ReadoutAudit:
    trainable_parameter_count: int
    trainable_names: tuple[str, ...]
    frozen_parameter_count: int


def apply_mace_readout_policy(model: object, *, expected_parameter_count: int = 2192) -> ReadoutAudit:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise HardFailure("MACE model does not expose named_parameters")
    trainable_names: list[str] = []
    trainable_count = 0
    frozen_count = 0
    for name, parameter in named_parameters():
        selected = name == "readouts" or name.startswith("readouts.")
        parameter.requires_grad_(selected)
        count = int(parameter.numel())
        if selected:
            trainable_names.append(name)
            trainable_count += count
        else:
            frozen_count += count
    if not trainable_names:
        raise HardFailure("MACE model has no readouts parameters")
    if trainable_count != expected_parameter_count:
        raise HardFailure(
            f"MACE readout parameter count drift: expected {expected_parameter_count}, got {trainable_count}"
        )
    return ReadoutAudit(trainable_count, tuple(trainable_names), frozen_count)
