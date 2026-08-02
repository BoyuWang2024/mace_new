"""Composition of the enabled force and energy confidence branches."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .adapters import LocalToGlobalCumulantAdapter
from .config import ConfidenceHeadConfig
from .heads import ComponentConfidenceHead, ConfidenceHead


class MultiBranchConfidenceModel(nn.Module):
    """Own exactly the trainable modules for branches enabled by the loss."""

    _OPTIONAL_MODULES = frozenset({"force_head", "energy_adapter", "energy_head"})

    def __init__(
        self,
        *,
        force_head: nn.Module | None,
        energy_adapter: LocalToGlobalCumulantAdapter | None,
        energy_head: ConfidenceHead | None,
    ) -> None:
        super().__init__()
        if force_head is None and (energy_adapter is None or energy_head is None):
            raise ValueError("at least one confidence branch must be enabled")
        if (energy_adapter is None) != (energy_head is None):
            raise ValueError("energy adapter and head must be enabled together")
        if force_head is not None:
            self.force_head = force_head
        if energy_adapter is not None:
            self.energy_adapter = energy_adapter
            self.energy_head = energy_head

    def __getattr__(self, name: str) -> Any:
        if name in self._OPTIONAL_MODULES:
            modules = object.__getattribute__(self, "_modules")
            return modules.get(name)
        return super().__getattr__(name)

    @classmethod
    def from_config(
        cls, config: ConfidenceHeadConfig, feature_dim: int = 640
    ) -> MultiBranchConfidenceModel:
        """Build only modules whose configured loss coefficient is positive."""
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise ValueError("feature_dim must be a positive integer")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if not config.force_enabled and not config.energy_enabled:
            raise ValueError("at least one confidence branch must be enabled")

        force_head: nn.Module | None = None
        if config.force_enabled:
            force_type = (
                ConfidenceHead
                if config.model.force.target_mode == "atom_mean"
                else ComponentConfidenceHead
            )
            force_head = force_type(
                feature_dim,
                config.model.force.hidden_dims,
                config.model.force.dropout,
                config.binning.force.num_bins,
            )

        energy_adapter = None
        energy_head = None
        if config.energy_enabled:
            energy_adapter = LocalToGlobalCumulantAdapter(
                feature_dim,
                config.model.energy.cumulant_order,
                config.model.energy.projection_dim,
                config.model.energy.signed_root,
                config.model.energy.adapter_dropout,
            )
            energy_head = ConfidenceHead(
                config.model.energy.projection_dim,
                config.model.energy.hidden_dims,
                config.model.energy.dropout,
                config.binning.energy.num_bins,
            )
        return cls(
            force_head=force_head,
            energy_adapter=energy_adapter,
            energy_head=energy_head,
        )

    def forward(
        self, features: torch.Tensor, offsets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return logits for exactly the enabled branches."""
        outputs: dict[str, torch.Tensor] = {}
        if self.force_head is not None:
            outputs["force"] = self.force_head(features)
        if self.energy_adapter is not None:
            outputs["energy"] = self.energy_head(self.energy_adapter(features, offsets))
        return outputs
