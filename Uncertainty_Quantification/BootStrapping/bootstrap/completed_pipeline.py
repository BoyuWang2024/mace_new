"""Resumable inference and analysis publication for completed MACE ensembles."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import ase.io
import numpy as np
import torch
from ase.stress import voigt_6_to_full_3x3_stress
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric

from .artifacts import atomic_write_json, atomic_write_npz, sha256_file
from .completed_run import audit_completed_run
from .dataset_inference import DatasetChunk, DatasetPlan, inspect_dataset
from .errors import HardFailure
from .inference_config import CompletedInferenceConfig, InferenceDatasetConfig
from .inference_store import InferenceStore
from .mace_inference import load_member_model, predict_batch


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _read_chunk(plan: DatasetPlan, chunk: DatasetChunk) -> list:
    start, stop = chunk.structure_range
    try:
        values = ase.io.read(plan.source, index=f"{start}:{stop}")
    except Exception as error:
        raise HardFailure(f"could not read dataset chunk {chunk.index}: {error}") from error
    if not isinstance(values, list) or len(values) != stop - start:
        raise HardFailure(f"dataset chunk {chunk.index} structure count drift")
    return values


def _graphs(model: torch.nn.Module, atoms_list: list, dtype: torch.dtype) -> list:
    atomic_numbers = [int(value) for value in torch.as_tensor(model.atomic_numbers).tolist()]
    table = AtomicNumberTable(atomic_numbers)
    cutoff = float(torch.as_tensor(model.r_max).item())
    heads = [str(value) for value in getattr(model, "heads", ["Default"])]
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        return [
            AtomicData.from_config(
                config_from_atoms(atoms), z_table=table, cutoff=cutoff, heads=heads
            )
            for atoms in atoms_list
        ]
    finally:
        torch.set_default_dtype(previous)


def _predict_chunk(
    model: torch.nn.Module,
    atoms_list: list,
    *,
    domains: tuple[str, ...],
    batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    loader = torch_geometric.dataloader.DataLoader(
        dataset=_graphs(model, atoms_list, dtype),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    parts: dict[str, list[np.ndarray]] = {"energy": [], "forces": []}
    if "stress" in domains:
        parts["stress"] = []
    for batch in loader:
        result = predict_batch(model, batch.to(torch.device(device)), domains=domains)
        for name, array in result.items():
            parts[name].append(array)
    return {name: np.concatenate(values, axis=0) for name, values in parts.items()}


def _target_arrays(plan: DatasetPlan) -> dict[str, np.ndarray]:
    ids: list[str] = []
    counts: list[int] = []
    energy: list[float] = []
    forces: list[np.ndarray] = []
    stress: list[np.ndarray] = []
    for chunk in plan.chunks:
        for index, atoms in enumerate(_read_chunk(plan, chunk), start=chunk.structure_range[0]):
            results = atoms.calc.results
            ids.append(str(atoms.info.get("structure_id", index)))
            counts.append(len(atoms))
            energy.append(float(results["energy"]))
            forces.append(np.asarray(results["forces"], dtype=float))
            if "stress" in plan.domains:
                value = np.asarray(results["stress"], dtype=float)
                stress.append(voigt_6_to_full_3x3_stress(value) if value.shape == (6,) else value)
    num_atoms = np.asarray(counts, dtype=np.int64)
    arrays: dict[str, np.ndarray] = {
        "structure_ids": np.asarray(ids),
        "num_atoms": num_atoms,
        "atom_offsets": np.concatenate(([0], np.cumsum(num_atoms))),
        "energy": np.asarray(energy, dtype=float),
        "forces": np.concatenate(forces, axis=0),
    }
    if stress:
        arrays["stress"] = np.stack(stress, axis=0)
    return arrays


def _gmd(values: np.ndarray) -> np.ndarray:
    pairs = [
        np.abs(values[left] - values[right])
        for left in range(values.shape[0])
        for right in range(left + 1, values.shape[0])
    ]
    return np.mean(np.stack(pairs, axis=0), axis=0)


def summarize_members(
    member_arrays: Iterable[dict[str, np.ndarray]], num_atoms: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    members = tuple(member_arrays)
    if len(members) < 2:
        raise HardFailure("completed inference requires at least two members")
    fields = set(members[0])
    if any(set(member) != fields for member in members):
        raise HardFailure("completed inference member fields differ")
    ensemble: dict[str, np.ndarray] = {}
    uncertainty: dict[str, np.ndarray] = {}
    for field in sorted(fields):
        values = np.stack([member[field] for member in members], axis=0)
        ensemble[field] = np.mean(values, axis=0)
        if field == "stress":
            values = 0.5 * (values + np.swapaxes(values, -1, -2))
            values = values[..., [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        stem = "force" if field == "forces" else field
        uncertainty[f"{stem}_std"] = np.std(values, axis=0, ddof=1)
        uncertainty[f"{stem}_gmd"] = _gmd(values)
        if field == "energy":
            per_atom = values / np.asarray(num_atoms, dtype=float)
            ensemble["energy_per_atom"] = np.mean(per_atom, axis=0)
            uncertainty["energy_per_atom_std"] = np.std(per_atom, axis=0, ddof=1)
            uncertainty["energy_per_atom_gmd"] = _gmd(per_atom)
    return ensemble, uncertainty


def _load_member(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _publish_analysis(
    destination: Path, store: InferenceStore, plan: DatasetPlan, member_count: int
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    targets = _target_arrays(plan)
    members = [
        _load_member(store.final_root / "members" / f"member_{index:03d}.npz")
        for index in range(member_count)
    ]
    ensemble, uncertainty = summarize_members(members, targets["num_atoms"])
    paths = (
        atomic_write_npz(destination / "targets.npz", **targets),
        atomic_write_npz(destination / "ensemble.npz", **ensemble),
        atomic_write_npz(destination / "uncertainty.npz", **uncertainty),
    )
    atomic_write_json(
        destination / "manifest.json",
        {
            "schema": "mace.bootstrap.completed-analysis/v1",
            "dataset_sha256": plan.source_sha256,
            "domains": list(plan.domains),
            "member_count": member_count,
            "artifacts": {path.name: sha256_file(path) for path in paths},
        },
    )
    return destination


def run_prediction_dataset(
    config: CompletedInferenceConfig,
    dataset: InferenceDatasetConfig,
    *,
    output_root: str | Path,
) -> Path:
    if dataset.kind != "predict":
        raise HardFailure(f"dataset {dataset.name} is not configured for prediction")
    completed = audit_completed_run(config.source.run, expected_members=config.source.member_count)
    plan = inspect_dataset(
        dataset.source,
        domains=dataset.domains,
        max_structures_per_chunk=config.prediction.max_structures_per_chunk,
        max_atoms_per_chunk=config.prediction.max_atoms_per_chunk,
    )
    root = Path(output_root).expanduser().resolve() / dataset.name
    store = InferenceStore(
        root / "inference",
        request={
            "dataset": dataset.name,
            "dataset_sha256": plan.source_sha256,
            "domains": list(dataset.domains),
            "member_count": completed.member_count,
            "run_manifest_sha256": completed.run_manifest_sha256,
            "precision": config.prediction.precision,
            "chunks": [
                {
                    "index": chunk.index,
                    "structure_range": list(chunk.structure_range),
                    "atom_count": chunk.atom_count,
                    "identity_sha256": chunk.identity_sha256,
                }
                for chunk in plan.chunks
            ],
        },
    )
    dtype = _dtype(config.prediction.precision)
    for member in completed.members:
        model = load_member_model(
            member.path,
            expected_sha256=member.sha256,
            device=config.prediction.device,
            dtype=dtype,
        )
        for chunk in plan.chunks:
            structures = chunk.structure_range[1] - chunk.structure_range[0]
            expected = {"energy": (structures,), "forces": (chunk.atom_count, 3)}
            if "stress" in dataset.domains:
                expected["stress"] = (structures, 3, 3)
            if store.reusable_chunk(member=member.index, chunk=chunk.index, expected=expected):
                continue
            arrays = _predict_chunk(
                model,
                _read_chunk(plan, chunk),
                domains=dataset.domains,
                batch_size=config.prediction.batch_size,
                device=config.prediction.device,
                dtype=dtype,
            )
            if {name: value.shape for name, value in arrays.items()} != expected:
                raise HardFailure(f"member {member.index} chunk {chunk.index} output shape drift")
            store.write_chunk(member=member.index, chunk=chunk.index, arrays=arrays)
        store.merge_member(member.index)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not store.final_root.exists():
        store.publish(member_count=completed.member_count)
    return _publish_analysis(root / "analysis", store, plan, completed.member_count)


__all__ = ["run_prediction_dataset", "summarize_members"]
