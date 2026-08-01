"""Trusted loading and immutable identity for a frozen MACE backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import CheckpointConfig
from .errors import DataContractError
from .identity import sha256_file


EXPECTED_FEATURE_MODULES = (("products.0", 512), ("products.1", 128))


@dataclass(frozen=True)
class BackboneIdentity:
    sha256: str
    model_class: str
    heads: tuple[str, ...]
    selected_head: str
    r_max: float
    atomic_numbers: tuple[int, ...]
    dtype: torch.dtype
    feature_modules: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class LoadedBackbone:
    model: torch.nn.Module
    identity: BackboneIdentity


def _floating_dtype(model: torch.nn.Module) -> torch.dtype:
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.is_floating_point():
            return tensor.dtype
    raise DataContractError("ScaleShiftMACE checkpoint has no floating-point tensors")


def _trusted_load(path: str, *, map_location: str) -> torch.nn.Module:
    from mace.tools.scripts_utils import extract_load

    return extract_load(path, map_location=map_location)


def _scale_shift_mace_type() -> type[torch.nn.Module]:
    from mace.modules import ScaleShiftMACE

    return ScaleShiftMACE


def _default_head(model: torch.nn.Module) -> tuple[tuple[str, ...], str]:
    raw_heads = getattr(model, "heads", None)
    if (
        not isinstance(raw_heads, (list, tuple))
        or not raw_heads
        or any(not isinstance(head, str) or not head for head in raw_heads)
    ):
        raise DataContractError("ScaleShiftMACE checkpoint must define non-empty heads")
    heads = tuple(str(head) for head in raw_heads)
    defaults = tuple(head for head in heads if head.lower() == "default")
    if len(defaults) != 1:
        raise DataContractError(
            "ScaleShiftMACE checkpoint must contain exactly one Default head"
        )
    return heads, defaults[0]


def load_frozen_backbone(
    config: CheckpointConfig, *, device: str | torch.device = "cpu"
) -> LoadedBackbone:
    """Verify and load a local ScaleShiftMACE checkpoint as a frozen model."""
    path = config.path.resolve()
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != config.expected_sha256.lower():
        raise DataContractError(
            f"checkpoint SHA-256 mismatch: expected {config.expected_sha256}, "
            f"got {actual_sha256}"
        )
    feature_modules = tuple(
        (module.name, module.expected_dim) for module in config.feature_modules
    )
    if feature_modules != EXPECTED_FEATURE_MODULES:
        raise DataContractError(
            "checkpoint feature modules must be products.0:512 then products.1:128"
        )
    model = _trusted_load(str(path), map_location=str(device))
    if not isinstance(model, _scale_shift_mace_type()):
        raise DataContractError(
            "checkpoint must contain a mace.modules.ScaleShiftMACE object"
        )
    heads, selected_head = _default_head(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    identity = BackboneIdentity(
        sha256=actual_sha256,
        model_class=model.__class__.__name__,
        heads=heads,
        selected_head=selected_head,
        r_max=float(torch.as_tensor(model.r_max).detach().cpu().item()),
        atomic_numbers=tuple(
            int(number)
            for number in torch.as_tensor(model.atomic_numbers).detach().cpu().tolist()
        ),
        dtype=_floating_dtype(model),
        feature_modules=feature_modules,
    )
    return LoadedBackbone(model=model, identity=identity)
