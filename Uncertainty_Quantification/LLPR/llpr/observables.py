"""Per-structure MACE predictions and readout Jacobians."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .readout import ReadoutLayout


@dataclass
class StructureJacobians:
    energy_per_atom: float
    forces: Tensor
    g_energy: Tensor
    g_forces: Tensor
    force_indices: Tensor
    chunk_size: int


def flatten_grads(
    gradients: tuple[Tensor | None, ...],
    parameters: tuple[nn.Parameter, ...],
) -> Tensor:
    """Flatten parameter gradients in readout-layout order on CPU as float64."""
    pieces = [
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ]
    if not pieces:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(pieces).detach().to(device="cpu", dtype=torch.float64)


def compute_structure_jacobians(
    model: nn.Module,
    batch: object,
    layout: ReadoutLayout,
    force_component_chunk_size: int,
    max_force_components: int | None,
) -> StructureJacobians:
    """Compute energy-per-atom and independent force-component Jacobians."""
    if (
        isinstance(force_component_chunk_size, bool)
        or not isinstance(force_component_chunk_size, int)
        or force_component_chunk_size <= 0
    ):
        raise ValueError("force_component_chunk_size must be a positive integer")
    if (
        max_force_components is not None
        and (
            isinstance(max_force_components, bool)
            or not isinstance(max_force_components, int)
            or max_force_components <= 0
        )
    ):
        raise ValueError("max_force_components must be positive or None")

    output = model(batch.to_dict(), training=True, compute_force=True)
    forces = output["forces"]
    if forces is None:
        raise ValueError("model did not return forces")

    num_atoms = int(batch.num_nodes)
    energy_total = output["energy"].reshape(-1).sum()
    energy_per_atom = energy_total / num_atoms
    g_energy = flatten_grads(
        torch.autograd.grad(
            energy_per_atom,
            layout.parameters,
            allow_unused=True,
            retain_graph=True,
        ),
        layout.parameters,
    )

    flat_forces = forces.reshape(-1)
    component_count = flat_forces.numel()
    if max_force_components is not None:
        component_count = min(component_count, max_force_components)
    force_indices = torch.arange(component_count, dtype=torch.long)

    gradient_chunks = []
    for start in range(0, component_count, force_component_chunk_size):
        rows = []
        for index in force_indices[start : start + force_component_chunk_size].tolist():
            rows.append(
                flatten_grads(
                    torch.autograd.grad(
                        flat_forces[index],
                        layout.parameters,
                        allow_unused=True,
                        retain_graph=True,
                    ),
                    layout.parameters,
                )
            )
        gradient_chunks.append(torch.stack(rows))
    g_forces = torch.cat(gradient_chunks, dim=0)

    return StructureJacobians(
        energy_per_atom=float(energy_per_atom.detach().item()),
        forces=forces.detach(),
        g_energy=g_energy,
        g_forces=g_forces,
        force_indices=force_indices,
        chunk_size=force_component_chunk_size,
    )
