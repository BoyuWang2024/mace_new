"""Discovery and stable metadata for MACE readout parameters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ReadoutLayout:
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    parameters: tuple[torch.nn.Parameter, ...]
    size: int

    def metadata(self) -> dict[str, object]:
        """Return the JSON-compatible portion of the parameter layout."""
        return {
            "names": list(self.names),
            "shapes": [list(shape) for shape in self.shapes],
            "size": self.size,
        }


def discover_readout_layout(
    model: nn.Module, expected_size: int | None = None
) -> ReadoutLayout:
    """Select trainable tensors under ``readouts`` in named-parameter order."""
    selected = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("readouts.")
    )
    names = tuple(name for name, _ in selected)
    parameters = tuple(parameter for _, parameter in selected)
    shapes = tuple(tuple(parameter.shape) for parameter in parameters)
    size = sum(parameter.numel() for parameter in parameters)
    if expected_size is not None and size != expected_size:
        raise ValueError(
            f"readout size mismatch: expected {expected_size}, actual {size}"
        )
    return ReadoutLayout(
        names=names,
        shapes=shapes,
        parameters=parameters,
        size=size,
    )
