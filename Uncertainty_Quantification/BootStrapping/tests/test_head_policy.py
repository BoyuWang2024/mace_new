from __future__ import annotations

import pytest
import torch

from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.head_policy import apply_mace_readout_policy


class TinyMACE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.interactions = torch.nn.Linear(3, 4)
        self.readouts = torch.nn.ModuleList([torch.nn.Linear(4, 8), torch.nn.Linear(8, 2)])


def test_only_mace_readouts_are_trainable() -> None:
    model = TinyMACE()
    expected = sum(parameter.numel() for parameter in model.readouts.parameters())
    audit = apply_mace_readout_policy(model, expected_parameter_count=expected)
    assert audit.trainable_parameter_count == expected
    assert all(name.startswith("readouts") for name in audit.trainable_names)
    assert not any(parameter.requires_grad for parameter in model.interactions.parameters())


def test_readout_parameter_count_drift_is_hard_failure() -> None:
    with pytest.raises(HardFailure, match="parameter count"):
        apply_mace_readout_policy(TinyMACE(), expected_parameter_count=2192)
