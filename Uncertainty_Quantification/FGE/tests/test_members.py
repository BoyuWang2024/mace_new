from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.artifacts import ExperimentLayout
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.members import (
    commit_member_pair,
    freeze_readouts,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2, bias=False)
        self.readouts = torch.nn.Linear(2, 2, bias=False)


def test_freeze_readouts_enforces_trainable_scope_and_count() -> None:
    model = TinyModel()
    guard = freeze_readouts(model, expected_count=4)
    assert model.readouts.weight.requires_grad
    assert not model.backbone.weight.requires_grad
    guard.assert_frozen_unchanged()


def test_frozen_backbone_change_is_hard_failure() -> None:
    model = TinyModel()
    guard = freeze_readouts(model, expected_count=4)
    with torch.no_grad():
        model.backbone.weight.add_(1.0)
    with pytest.raises(HardFailure, match="frozen backbone drift"):
        guard.assert_frozen_unchanged()


def test_nonfinite_trainable_parameter_is_hard_failure() -> None:
    model = TinyModel()
    guard = freeze_readouts(model, expected_count=4)
    with torch.no_grad():
        model.readouts.weight[0, 0] = float("nan")
    with pytest.raises(HardFailure, match="NaN or Inf"):
        guard.assert_trainable_finite()


def test_commit_member_pair_writes_raw_and_ema_atomically(tmp_path: Path) -> None:
    layout = ExperimentLayout(tmp_path / "run")
    result = commit_member_pair(
        layout,
        "member_01",
        TinyModel(),
        TinyModel(),
        {"raw_metrics": {"energy_rmse": 1.0}, "ema_metrics": {"energy_rmse": 1.1}},
    )
    assert (layout.training_dir / "members" / "raw" / "member_01.model").is_file()
    assert (layout.training_dir / "members" / "ema" / "member_01.model").is_file()
    assert result["member_id"] == "member_01"
    assert result["frozen_backbone_verified"] is True
