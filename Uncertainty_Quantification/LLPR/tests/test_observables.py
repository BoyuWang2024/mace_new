from __future__ import annotations

from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.LLPR.llpr.checkpoint import load_checkpoint
from Uncertainty_Quantification.LLPR.llpr.config import PathIdentity
from Uncertainty_Quantification.LLPR.llpr.data import build_dataset, iter_samples
from Uncertainty_Quantification.LLPR.llpr.observables import (
    compute_structure_jacobians,
)
from Uncertainty_Quantification.LLPR.llpr.readout import discover_readout_layout


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CHECKPOINT_PATH = (
    _REPOSITORY_ROOT / "data/checkpoint/MACE-matpes-r2scan-omat-ft.model"
)
_DATASET_PATH = _REPOSITORY_ROOT / "data/dataset/matpes_n20.extxyz"


class AnalyticReadout(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([2.0, 3.0, 5.0]))


class AnalyticModel(torch.nn.Module):
    expected_total = 23.0

    def __init__(self) -> None:
        super().__init__()
        self.readouts = torch.nn.ModuleList([AnalyticReadout()])

    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        del data
        theta = self.readouts[0].theta
        energy = torch.tensor([1.0, 2.0, 3.0]) @ theta
        coefficients = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 2.0, 3.0],
                [-1.0, 0.5, 2.0],
                [4.0, -2.0, 1.0],
            ]
        )
        forces = (coefficients @ theta).reshape(2, 3)
        if not training:
            forces = forces.detach()
        return {
            "energy": energy.reshape(1),
            "forces": forces if compute_force else None,
        }


class AnalyticBatch:
    num_nodes = 2

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {}


def analytic_batch(num_atoms: int) -> AnalyticBatch:
    assert num_atoms == 2
    return AnalyticBatch()


def test_energy_is_per_atom_and_force_is_per_component() -> None:
    model = AnalyticModel()
    layout = discover_readout_layout(model)

    jac = compute_structure_jacobians(
        model=model,
        batch=analytic_batch(num_atoms=2),
        layout=layout,
        force_component_chunk_size=2,
        max_force_components=None,
    )

    assert jac.energy_per_atom == pytest.approx(model.expected_total / 2)
    assert jac.g_energy.shape == (layout.size,)
    assert jac.g_energy.tolist() == pytest.approx([0.5, 1.0, 1.5])
    assert jac.g_forces.shape == (6, layout.size)
    torch.testing.assert_close(
        jac.g_forces,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 2.0, 3.0],
                [-1.0, 0.5, 2.0],
                [4.0, -2.0, 1.0],
            ],
            dtype=torch.float64,
        ),
    )
    assert jac.force_indices.tolist() == list(range(6))
    assert jac.chunk_size == 2
    assert jac.g_energy.device.type == "cpu"
    assert jac.g_energy.dtype == torch.float64
    assert jac.g_forces.device.type == "cpu"
    assert jac.g_forces.dtype == torch.float64
    assert jac.forces.dtype == torch.float32
    torch.testing.assert_close(
        jac.forces, torch.tensor([[2.0, 3.0, 5.0], [23.0, 9.5, 7.0]])
    )


def test_max_force_components_selects_flat_prefix() -> None:
    model = AnalyticModel()
    layout = discover_readout_layout(model)

    jac = compute_structure_jacobians(
        model=model,
        batch=analytic_batch(num_atoms=2),
        layout=layout,
        force_component_chunk_size=8,
        max_force_components=4,
    )

    assert jac.force_indices.tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(
        jac.g_forces,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=torch.float64,
        ),
    )
    assert jac.forces.shape == (2, 3)


def test_real_first_structure_force_chunk_parity() -> None:
    loaded = load_checkpoint(PathIdentity(_CHECKPOINT_PATH), torch.device("cpu"))
    dataset = build_dataset(
        _DATASET_PATH,
        None,
        loaded.identity.atomic_numbers,
        loaded.identity.r_max,
    )
    sample = next(
        iter_samples(
            dataset,
            torch.device("cpu"),
            loaded.identity.dtype,
            max_structures=1,
        )
    )
    layout = discover_readout_layout(loaded.model)

    scalar = compute_structure_jacobians(
        model=loaded.model,
        batch=sample.batch,
        layout=layout,
        force_component_chunk_size=1,
        max_force_components=None,
    )
    chunked = compute_structure_jacobians(
        model=loaded.model,
        batch=sample.batch,
        layout=layout,
        force_component_chunk_size=8,
        max_force_components=None,
    )

    assert scalar.energy_per_atom == pytest.approx(
        chunked.energy_per_atom, rel=1.0e-10, abs=1.0e-12
    )
    torch.testing.assert_close(
        scalar.forces, chunked.forces, rtol=1.0e-10, atol=1.0e-12
    )
    torch.testing.assert_close(
        scalar.g_energy, chunked.g_energy, rtol=1.0e-10, atol=1.0e-12
    )
    torch.testing.assert_close(
        scalar.g_forces, chunked.g_forces, rtol=1.0e-10, atol=1.0e-12
    )
    assert scalar.force_indices.tolist() == list(range(3 * sample.num_atoms))
    assert torch.equal(scalar.force_indices, chunked.force_indices)
