"""One-forward feature capture from the fixed MACE product modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .data import StructureBatch
from .errors import FeatureSchemaError


EXPECTED_FEATURE_MODULES = (("products.0", 512), ("products.1", 128))


@dataclass
class ContinuousBatch:
    indices: torch.Tensor
    structure_ids: tuple[str, ...]
    num_atoms: torch.Tensor
    atomic_numbers: torch.Tensor
    atom_offsets: torch.Tensor
    features: torch.Tensor
    reference_energy: torch.Tensor
    reference_forces: torch.Tensor


class FeatureCapture:
    """Capture exactly one output from each configured MACE product module."""

    def __init__(
        self, model: nn.Module, feature_modules: Sequence[tuple[str, int]]
    ) -> None:
        self.model = model
        self.feature_modules = tuple(
            (str(name), int(width)) for name, width in feature_modules
        )
        if self.feature_modules != EXPECTED_FEATURE_MODULES:
            raise FeatureSchemaError(
                "feature modules must be products.0:512 then products.1:128"
            )
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captured: dict[str, list[object]] = {
            name: [] for name, _ in self.feature_modules
        }

    @property
    def hooks_registered(self) -> bool:
        return bool(self._handles)

    def _hook(self, name: str):
        def capture(
            _module: nn.Module, _inputs: tuple[object, ...], output: object
        ) -> None:
            self._captured[name].append(output)

        return capture

    def __enter__(self) -> FeatureCapture:
        if self.hooks_registered:
            raise FeatureSchemaError("feature hooks are already registered")
        try:
            for name, _ in self.feature_modules:
                module = self.model.get_submodule(name)
                self._handles.append(
                    module.register_forward_hook(self._hook(name))
                )
        except (AttributeError, KeyError) as error:
            self.__exit__(None, None, None)
            raise FeatureSchemaError(f"missing feature module {name}") from error
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for outputs in self._captured.values():
            outputs.clear()

    def take(self, *, expected_atoms: int) -> torch.Tensor:
        """Validate, detach, concatenate, and clear one forward's outputs."""
        captured = {
            name: tuple(outputs) for name, outputs in self._captured.items()
        }
        for outputs in self._captured.values():
            outputs.clear()

        tensors: list[torch.Tensor] = []
        for name, width in self.feature_modules:
            outputs = captured[name]
            if len(outputs) != 1:
                raise FeatureSchemaError(
                    f"feature module {name} triggered {len(outputs)} times; "
                    "expected once"
                )
            output = outputs[0]
            if not isinstance(output, torch.Tensor):
                raise FeatureSchemaError(
                    f"feature module {name} did not return a tensor"
                )
            if output.ndim != 2 or output.shape[0] != expected_atoms:
                raise FeatureSchemaError(
                    f"feature module {name} must return [{expected_atoms}, {width}]"
                )
            if output.shape[1] != width:
                raise FeatureSchemaError(
                    f"feature module {name} expected width {width}, "
                    f"got {output.shape[1]}"
                )
            detached = output.detach()
            if not torch.isfinite(detached).all():
                raise FeatureSchemaError(f"feature module {name} must be finite")
            tensors.append(detached)

        dtype = tensors[0].dtype
        device = tensors[0].device
        if any(
            tensor.dtype != dtype or tensor.device != device
            for tensor in tensors[1:]
        ):
            raise FeatureSchemaError(
                "captured feature tensors must share dtype and device"
            )
        return torch.cat(tensors, dim=-1)


def to_continuous_batch(
    batch: StructureBatch, features: torch.Tensor
) -> ContinuousBatch:
    """Replace a MACE graph batch with detached continuous atom features."""
    expected_atoms = int(batch.atom_offsets[-1].item())
    if features.ndim != 2 or features.shape != (expected_atoms, 640):
        raise FeatureSchemaError(
            f"continuous features must have shape ({expected_atoms}, 640)"
        )
    if (
        features.dtype != batch.reference_forces.dtype
        or features.device != batch.reference_forces.device
        or features.dtype != batch.reference_energy.dtype
        or features.device != batch.reference_energy.device
    ):
        raise FeatureSchemaError(
            "continuous features must match reference dtype and device"
        )
    detached = features.detach()
    if not torch.isfinite(detached).all():
        raise FeatureSchemaError("continuous features must be finite")
    return ContinuousBatch(
        indices=batch.indices,
        structure_ids=batch.structure_ids,
        num_atoms=batch.num_atoms,
        atomic_numbers=batch.atomic_numbers,
        atom_offsets=batch.atom_offsets,
        features=detached,
        reference_energy=batch.reference_energy,
        reference_forces=batch.reference_forces,
    )
