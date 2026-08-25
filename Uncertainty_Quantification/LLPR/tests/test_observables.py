from __future__ import annotations

from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.LLPR.llpr import observables
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
        self.compute_force_calls: list[bool] = []

    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        del data
        self.compute_force_calls.append(compute_force)
        theta = self.readouts[0].theta
        energy = torch.tensor([1.0, 2.0, 3.0]) @ theta
        if not compute_force:
            return {"energy": energy.reshape(1)}
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
        return {"energy": energy.reshape(1), "forces": forces}


class MissingEnergyModel(AnalyticModel):
    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        del data, training
        self.compute_force_calls.append(compute_force)
        return {}


class NonFiniteEnergyModel(AnalyticModel):
    def __init__(self, energy_value: float) -> None:
        super().__init__()
        self.energy_value = energy_value

    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        del data, training
        self.compute_force_calls.append(compute_force)
        theta = self.readouts[0].theta
        energy = theta.sum() * 0.0 + torch.tensor(self.energy_value)
        return {"energy": energy.reshape(1)}


class AnalyticBatch:
    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {}


def analytic_batch(num_atoms: int) -> AnalyticBatch:
    return AnalyticBatch(num_atoms)


def test_energy_only_matches_full_jacobian_without_requesting_forces() -> None:
    model = AnalyticModel()
    layout = discover_readout_layout(model)
    batch = analytic_batch(num_atoms=2)

    energy_only = observables.compute_energy_jacobian(model, batch, layout)
    full = compute_structure_jacobians(
        model=model,
        batch=batch,
        layout=layout,
        force_component_chunk_size=2,
        max_force_components=None,
    )

    assert model.compute_force_calls == [False, True]
    assert energy_only.energy_total == pytest.approx(23.0)
    assert energy_only.energy_per_atom == pytest.approx(11.5)
    assert energy_only.g_energy.tolist() == pytest.approx([0.5, 1.0, 1.5])
    assert energy_only.energy_per_atom == pytest.approx(full.energy_per_atom)
    torch.testing.assert_close(energy_only.g_energy, full.g_energy)
    assert energy_only.g_energy.device.type == "cpu"
    assert energy_only.g_energy.dtype == torch.float64


@pytest.mark.parametrize("num_atoms", [0, -1])
def test_energy_only_rejects_non_positive_num_atoms(num_atoms: int) -> None:
    model = AnalyticModel()
    layout = discover_readout_layout(model)

    with pytest.raises(ValueError, match="num_atoms must be positive"):
        observables.compute_energy_jacobian(
            model, analytic_batch(num_atoms=num_atoms), layout
        )


def test_energy_only_requires_model_energy() -> None:
    model = MissingEnergyModel()
    layout = discover_readout_layout(model)

    with pytest.raises(ValueError, match="model did not return energy"):
        observables.compute_energy_jacobian(model, analytic_batch(2), layout)

    assert model.compute_force_calls == [False]


@pytest.mark.parametrize(
    "energy_value", [float("nan"), float("inf"), float("-inf")]
)
def test_energy_only_rejects_non_finite_energy(energy_value: float) -> None:
    model = NonFiniteEnergyModel(energy_value)
    layout = discover_readout_layout(model)

    with pytest.raises(ValueError, match="model energy must be finite"):
        observables.compute_energy_jacobian(model, analytic_batch(2), layout)


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
