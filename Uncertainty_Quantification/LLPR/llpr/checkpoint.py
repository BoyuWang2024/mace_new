"""Validated loading of the fixed MACE checkpoint used by LLPR."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .artifacts import sha256_file
from .config import PathIdentity


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
    if not isinstance(model, torch.nn.Module):
        raise ValueError("checkpoint must contain a torch.nn.Module")
    model.to(device)
    model.eval()

    model_class = model.__class__.__name__
    if model_class != "ScaleShiftMACE":
        raise ValueError(
            f"checkpoint model class must be ScaleShiftMACE, got {model_class}"
        )
    heads = tuple(str(head) for head in model.heads)
    if selected_head not in heads:
        raise ValueError(
            f"checkpoint head {selected_head!r} is not available in {heads!r}"
        )
    r_max = float(model.r_max)
    if r_max != 6.0:
        raise ValueError(f"checkpoint r_max must be 6.0, got {r_max}")
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
