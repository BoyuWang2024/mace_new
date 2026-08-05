from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.prediction import validate_prediction_payload


def test_nan_is_always_a_hard_failure() -> None:
    payload = {
        "schema_version": "fge.prediction.v1",
        "split": "test",
        "member_source": "raw",
        "member_ids": ["member_01", "member_02"],
        "observables": ["energy", "forces"],
        "energy_members": torch.tensor([[float("nan")], [1.0]], dtype=torch.float64),
        "forces_members": torch.zeros((2, 1, 3), dtype=torch.float64),
        "energy_reference": torch.zeros(1, dtype=torch.float64),
        "forces_reference": torch.zeros((1, 3), dtype=torch.float64),
        "n_atoms": torch.ones(1, dtype=torch.int64),
        "atom_to_structure": torch.zeros(1, dtype=torch.int64),
        "structure_ptr": torch.tensor([0, 1], dtype=torch.int64),
    }
    with pytest.raises(HardFailure, match="NaN or Inf"):
        validate_prediction_payload(payload)
