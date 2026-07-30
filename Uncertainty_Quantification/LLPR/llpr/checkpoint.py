"""Validated loading of the fixed MACE checkpoint used by LLPR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mace.modules.models import ScaleShiftMACE

from .artifacts import sha256_file
from .config import PathIdentity
from .readout import discover_readout_layout


EXPECTED_READOUT_SIZE = 2192


@dataclass(frozen=True)
class CheckpointIdentity:
    sha256: str
    model_class: str
    heads: tuple[str, ...]
    selected_head: str
    r_max: float
    atomic_numbers: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    identity: CheckpointIdentity


def load_checkpoint(
    source: PathIdentity,
    device: torch.device,
    selected_head: str = "default",
) -> LoadedCheckpoint:
    """Load and validate the formal ScaleShiftMACE checkpoint."""
    if selected_head != "default":
        raise ValueError("selected_head must be 'default'")

    actual_sha256 = sha256_file(source.path)
    if (
        source.expected_sha256 is not None
        and actual_sha256.lower() != source.expected_sha256.lower()
    ):
        raise ValueError(
            "checkpoint SHA256 mismatch: "
            f"expected {source.expected_sha256}, actual {actual_sha256}"
        )

    model = torch.load(source.path, map_location=device, weights_only=False)
    if not isinstance(model, ScaleShiftMACE):
        raise ValueError("checkpoint must contain a real ScaleShiftMACE model")
    model.to(device)
    model.eval()

    model_class = model.__class__.__name__
    heads = tuple(str(head) for head in model.heads)
    if "default" not in heads:
        raise ValueError("checkpoint heads must include 'default'")
    r_max = float(model.r_max)
    if r_max != 6.0:
        raise ValueError(f"checkpoint r_max must be 6.0, got {r_max}")
    discover_readout_layout(model, expected_size=EXPECTED_READOUT_SIZE)
    atomic_numbers = tuple(int(number) for number in model.atomic_numbers.tolist())
    dtype = next(model.parameters()).dtype

    return LoadedCheckpoint(
        model=model,
        identity=CheckpointIdentity(
            sha256=actual_sha256,
            model_class=model_class,
            heads=heads,
            selected_head=selected_head,
            r_max=r_max,
            atomic_numbers=atomic_numbers,
            dtype=dtype,
        ),
    )
