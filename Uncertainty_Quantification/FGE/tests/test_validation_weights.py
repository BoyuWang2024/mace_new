from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.artifacts import atomic_torch_save
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.validation import validate_result
from Uncertainty_Quantification.FGE.tests.test_validation import _make_result


def test_validator_detects_tampered_ensemble_weights(tmp_path: Path) -> None:
    config = _make_result(tmp_path)
    path = tmp_path / "evaluation" / "equal_weight" / "ensemble.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["energy_weights"] = torch.tensor([0.9, 0.1], dtype=torch.float64)
    atomic_torch_save(path, payload)
    with pytest.raises(HardFailure, match="energy_weights"):
        validate_result(config, tmp_path)
