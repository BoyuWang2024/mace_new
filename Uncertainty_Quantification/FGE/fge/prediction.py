"""Immutable raw-member inference and canonical prediction serialization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .artifacts import ExperimentLayout, atomic_torch_save, atomic_write_json
from .data import build_mace_loaders, load_extxyz
from .errors import HardFailure
from .manifests import build_prediction_manifest
from .preflight import _load_model, run_preflight


_BASE_KEYS = {
    "schema_version",
    "split",
    "member_source",
    "member_ids",
    "observables",
    "energy_members",
    "forces_members",
    "energy_reference",
    "forces_reference",
    "n_atoms",
    "atom_to_structure",
    "structure_ptr",
}


@dataclass(frozen=True)
class PredictionShape:
    members: int
    structures: int
    atoms: int
    has_stress: bool


def _tensor(
    payload: Mapping[str, Any], name: str, shape: tuple[int, ...], dtype: torch.dtype
) -> Tensor:
    value = payload.get(name)
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise HardFailure(f"canonical field {name} has invalid shape")
    if value.dtype != dtype:
        expected = "float64" if dtype == torch.float64 else "int64"
        raise HardFailure(f"canonical field {name} must use {expected}")
    if value.is_floating_point() and not torch.isfinite(value).all().item():
        raise HardFailure(f"canonical field {name} contains NaN or Inf")
    if value.device.type != "cpu":
        raise HardFailure(f"canonical field {name} must be stored on CPU")
    return value


def validate_prediction_payload(payload: Mapping[str, Any]) -> PredictionShape:
    """Validate exact keys, dtype, member order, and E/F structure alignment."""
    if not isinstance(payload, Mapping):
        raise HardFailure("canonical prediction must be a mapping")
    observables = payload.get("observables")
    has_stress = observables == ["energy", "forces", "stress"]
    if observables not in (["energy", "forces"], ["energy", "forces", "stress"]):
        raise HardFailure("canonical observables are invalid")
    expected_keys = _BASE_KEYS | ({"stress_members", "stress_reference"} if has_stress else set())
    if set(payload) != expected_keys:
        raise HardFailure("canonical prediction keys do not match the fixed schema")
    if (
        payload.get("schema_version") != "fge.prediction.v1"
        or payload.get("split") != "test"
        or payload.get("member_source") != "raw"
    ):
        raise HardFailure("canonical prediction metadata is invalid")
    member_ids = payload.get("member_ids")
    if not isinstance(member_ids, list) or len(member_ids) < 2:
        raise HardFailure("canonical member list must contain K >= 2")
    expected_ids = [f"member_{index:02d}" for index in range(1, len(member_ids) + 1)]
    if member_ids != expected_ids:
        raise HardFailure("canonical member order must be contiguous")
    energy_reference = payload.get("energy_reference")
    forces_reference = payload.get("forces_reference")
    if not isinstance(energy_reference, Tensor) or energy_reference.ndim != 1:
        raise HardFailure("canonical energy reference has invalid shape")
    if not isinstance(forces_reference, Tensor) or forces_reference.ndim != 2:
        raise HardFailure("canonical force reference has invalid shape")
    members, structures, atoms = len(member_ids), energy_reference.shape[0], forces_reference.shape[0]
    if structures < 1 or atoms < 1:
        raise HardFailure("canonical prediction cannot be empty")
    _tensor(payload, "energy_members", (members, structures), torch.float64)
    _tensor(payload, "forces_members", (members, atoms, 3), torch.float64)
    _tensor(payload, "energy_reference", (structures,), torch.float64)
    _tensor(payload, "forces_reference", (atoms, 3), torch.float64)
    n_atoms = _tensor(payload, "n_atoms", (structures,), torch.int64)
    mapping = _tensor(payload, "atom_to_structure", (atoms,), torch.int64)
    pointer = _tensor(payload, "structure_ptr", (structures + 1,), torch.int64)
    if (n_atoms <= 0).any().item() or int(n_atoms.sum().item()) != atoms:
        raise HardFailure("canonical atom counts are inconsistent")
    expected_mapping = torch.repeat_interleave(torch.arange(structures), n_atoms)
    expected_pointer = torch.cat((torch.zeros(1, dtype=torch.int64), n_atoms.cumsum(0)))
    if not torch.equal(mapping, expected_mapping) or not torch.equal(pointer, expected_pointer):
        raise HardFailure("canonical atom mapping is inconsistent")
    if has_stress:
        _tensor(payload, "stress_members", (members, structures, 3, 3), torch.float64)
        _tensor(payload, "stress_reference", (structures, 3, 3), torch.float64)
    return PredictionShape(members, structures, atoms, has_stress)


def _model_value(model: nn.Module, name: str) -> float:
    value = getattr(model, name, None)
    if isinstance(value, Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (int, float)):
        return float(value)
    raise HardFailure(f"member model has no usable {name}")


def _infer_one(
    model: nn.Module, loader: Any, device: torch.device, compute_stress: bool
) -> dict[str, Tensor]:
    energies: list[Tensor] = []
    forces: list[Tensor] = []
    references_e: list[Tensor] = []
    references_f: list[Tensor] = []
    counts: list[Tensor] = []
    stresses: list[Tensor] = []
    references_s: list[Tensor] = []
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for batch in loader:
        batch = batch.to(device)
        with torch.enable_grad():
            output = model(
                batch.to_dict(),
                training=False,
                compute_force=True,
                compute_virials=False,
                compute_stress=compute_stress,
            )
        energies.append(output["energy"].detach().cpu().to(torch.float64))
        forces.append(output["forces"].detach().cpu().to(torch.float64))
        references_e.append(batch.energy.detach().cpu().to(torch.float64))
        references_f.append(batch.forces.detach().cpu().to(torch.float64))
        counts.append((batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().to(torch.int64))
        if compute_stress:
            stresses.append(output["stress"].detach().cpu().to(torch.float64))
            references_s.append(batch.stress.detach().cpu().to(torch.float64))
    result = {
        "energy": torch.cat(energies),
        "forces": torch.cat(forces),
        "energy_reference": torch.cat(references_e),
        "forces_reference": torch.cat(references_f),
        "n_atoms": torch.cat(counts),
    }
    if compute_stress:
        result.update({"stress": torch.cat(stresses), "stress_reference": torch.cat(references_s)})
    return result


def generate_prediction_payload(
    config: Any,
    manifest: Mapping[str, Any],
    configurations: Sequence[Any],
    *,
    compute_stress: bool,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run committed raw members for an explicit ordered configuration sequence."""
    data_config = config.section("data")
    prediction_config = config.section("prediction")
    training_config = config.section("training")
    device = torch.device(training_config["device"])
    results: list[dict[str, Tensor]] = []
    member_ids: list[str] = []
    for member in manifest["members"]:
        member_id = member.get("member_id")
        if not isinstance(member_id, str) or not re.fullmatch(r"member_[0-9]{2}", member_id):
            raise HardFailure("training manifest contains an invalid member ID")
        member_ids.append(member_id)
        model_path = Path(config.output_dir) / member["raw"]["path"]
        model = _load_model(model_path, str(device))
        atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
        loader = build_mace_loaders(
            configurations,
            atomic_numbers=atomic_numbers,
            cutoff=_model_value(model, "r_max"),
            batch_size=(
                prediction_config["batch_size"] if batch_size is None else batch_size
            ),
            shuffle=False,
            heads=list(getattr(model, "heads", [data_config["head_name"]])),
        )
        results.append(_infer_one(model, loader, device, compute_stress))
    if not results:
        raise HardFailure("training manifest contains no members")
    first = results[0]
    reference_names = ["energy_reference", "forces_reference", "n_atoms"]
    if compute_stress:
        reference_names.append("stress_reference")
    for result in results[1:]:
        for name in reference_names:
            if not torch.equal(result[name], first[name]):
                raise HardFailure("member prediction references are not aligned")
    n_atoms = first["n_atoms"]
    structure_ptr = torch.cat((torch.zeros(1, dtype=torch.int64), n_atoms.cumsum(0)))
    atom_to_structure = torch.repeat_interleave(torch.arange(n_atoms.numel()), n_atoms)
    payload: dict[str, Any] = {
        "schema_version": "fge.prediction.v1",
        "split": "test",
        "member_source": "raw",
        "member_ids": member_ids,
        "observables": ["energy", "forces"],
        "energy_members": torch.stack([result["energy"] for result in results]),
        "forces_members": torch.stack([result["forces"] for result in results]),
        "energy_reference": first["energy_reference"],
        "forces_reference": first["forces_reference"],
        "n_atoms": n_atoms,
        "atom_to_structure": atom_to_structure,
        "structure_ptr": structure_ptr,
    }
    if compute_stress:
        payload["observables"].append("stress")
        payload["stress_members"] = torch.stack([result["stress"] for result in results])
        payload["stress_reference"] = first["stress_reference"]
    return payload


def _generate_prediction_payload(config: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    paths = config.section("paths")
    data_config = config.section("data")
    prediction_config = config.section("prediction")
    compute_stress = bool(prediction_config["compute_stress"])
    keys = {
        "energy": data_config["energy_key"],
        "forces": data_config["forces_key"],
        "stress": data_config["stress_key"],
        "head": data_config["head_name"],
    }
    required = {"energy", "forces", "stress"} if compute_stress else {"energy", "forces"}
    configurations = load_extxyz(
        Path(paths["test_data"]),
        keys=keys,
        required=required,
        head_name=data_config["head_name"],
    )
    return generate_prediction_payload(
        config,
        manifest,
        configurations,
        compute_stress=compute_stress,
    )

def predict_members(config: Any) -> Path:
    """Run only raw committed members and write one canonical prediction artifact."""
    run_preflight(config, "predict")
    layout = ExperimentLayout(config.output_dir)
    try:
        manifest = json.loads(layout.training_manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HardFailure("training manifest cannot be loaded") from exc
    payload = _generate_prediction_payload(config, manifest)
    shape = validate_prediction_payload(payload)
    atomic_torch_save(layout.prediction_tensor, payload)
    prediction_manifest = build_prediction_manifest(
        root=layout.root,
        prediction_path=layout.prediction_tensor,
        member_count=shape.members,
        structure_count=shape.structures,
        atom_count=shape.atoms,
        observables=payload["observables"],
    )
    atomic_write_json(layout.prediction_manifest, prediction_manifest)
    return layout.prediction_manifest
