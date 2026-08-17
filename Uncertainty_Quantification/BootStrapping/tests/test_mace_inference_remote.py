from __future__ import annotations

import os
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric

from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import sha256_file
from Uncertainty_Quantification.BootStrapping.bootstrap.mace_inference import (
    load_member_model,
    predict_batch,
)


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is required for the remote integration test")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"{name} is not a file: {path}")
    return path


@pytest.mark.remote
def test_real_raw_best_model_predicts_finite_n20_energy_force_stress() -> None:
    model_path = _required_path("MACE_BOOTSTRAP_REMOTE_MODEL")
    dataset_path = _required_path("MACE_BOOTSTRAP_REMOTE_DATASET")
    model = load_member_model(
        model_path,
        expected_sha256=sha256_file(model_path),
        device="cpu",
        dtype=torch.float64,
    )

    atoms_list = ase.io.read(dataset_path, index="0:2", format="extxyz")
    assert len(atoms_list) == 2
    z_table = AtomicNumberTable(
        [int(value) for value in torch.as_tensor(model.atomic_numbers).tolist()]
    )
    cutoff = float(torch.as_tensor(model.r_max).item())
    heads = [str(head) for head in getattr(model, "heads", ["Default"])]
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        graphs = [
            AtomicData.from_config(
                config_from_atoms(atoms),
                z_table=z_table,
                cutoff=cutoff,
                heads=heads,
            )
            for atoms in atoms_list
        ]
    finally:
        torch.set_default_dtype(previous_dtype)
    batch = next(
        iter(
            torch_geometric.dataloader.DataLoader(
                dataset=graphs,
                batch_size=len(graphs),
                shuffle=False,
                drop_last=False,
            )
        )
    )

    result = predict_batch(
        model,
        batch,
        domains=("energy", "forces", "stress"),
    )

    total_atoms = sum(len(atoms) for atoms in atoms_list)
    assert result["energy"].shape == (2,)
    assert result["forces"].shape == (total_atoms, 3)
    assert result["stress"].shape == (2, 3, 3)
    assert all(np.isfinite(value).all() for value in result.values())
