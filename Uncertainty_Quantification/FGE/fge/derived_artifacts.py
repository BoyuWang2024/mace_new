"""Source-neutral layouts and resumable artifacts for derived FGE inference."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .artifacts import atomic_torch_save, atomic_write_json, sha256_file
from .errors import HardFailure
from .prediction import validate_prediction_payload


_LABEL_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_SIGNATURE_KEYS = {
    "schema_version",
    "dataset",
    "observables",
    "member_ids",
    "shard_index",
    "structure_start",
    "structure_stop",
    "atom_start",
    "atom_stop",
    "batch_size",
}


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or _LABEL_PATTERN.fullmatch(value) is None:
        raise HardFailure(f"{name} must be a source-neutral logical label")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HardFailure(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class DerivedLayout:
    outputs_root: Path
    experiment: str
    dataset: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs_root", Path(self.outputs_root))
        _label(self.experiment, "experiment")
        _label(self.dataset, "dataset")

    @property
    def root(self) -> Path:
        return self.outputs_root / "inference" / self.experiment / self.dataset

    @property
    def prediction_dir(self) -> Path:
        return self.root / "prediction"

    @property
    def prediction_manifest(self) -> Path:
        return self.prediction_dir / "manifest.json"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    def prediction_shard(self, index: int) -> Path:
        index = _integer(index, "shard index")
        return self.prediction_dir / "shards" / f"shard_{index:06d}.pt"

    def prediction_shard_manifest(self, index: int) -> Path:
        index = _integer(index, "shard index")
        return self.prediction_dir / "shards" / f"shard_{index:06d}.json"


def build_prediction_shard_signature(
    *,
    dataset: str,
    observables: Sequence[str],
    member_ids: Sequence[str],
    shard_index: int,
    structure_start: int,
    structure_stop: int,
    atom_start: int,
    atom_stop: int,
    batch_size: int,
) -> dict[str, object]:
    dataset = _label(dataset, "dataset")
    observable_values = list(observables)
    if observable_values not in (["energy", "forces"], ["energy", "forces", "stress"]):
        raise HardFailure("prediction shard observables are invalid")
    member_values = list(member_ids)
    expected_ids = [f"member_{index:02d}" for index in range(1, len(member_values) + 1)]
    if len(member_values) < 2 or member_values != expected_ids:
        raise HardFailure("prediction shard member IDs must be contiguous")
    shard_index = _integer(shard_index, "shard_index")
    structure_start = _integer(structure_start, "structure_start")
    structure_stop = _integer(structure_stop, "structure_stop")
    atom_start = _integer(atom_start, "atom_start")
    atom_stop = _integer(atom_stop, "atom_stop")
    batch_size = _integer(batch_size, "batch_size", minimum=1)
    if structure_stop <= structure_start or atom_stop <= atom_start:
        raise HardFailure("prediction shard ranges must be nonempty and increasing")
    return {
        "schema_version": "fge.derived-prediction-signature.v1",
        "dataset": dataset,
        "observables": observable_values,
        "member_ids": member_values,
        "shard_index": shard_index,
        "structure_start": structure_start,
        "structure_stop": structure_stop,
        "atom_start": atom_start,
        "atom_stop": atom_stop,
        "batch_size": batch_size,
    }


def _validated_signature(value: Mapping[str, Any]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _SIGNATURE_KEYS:
        raise HardFailure("prediction shard signature has invalid fields")
    return build_prediction_shard_signature(
        dataset=value["dataset"],
        observables=value["observables"],
        member_ids=value["member_ids"],
        shard_index=value["shard_index"],
        structure_start=value["structure_start"],
        structure_stop=value["structure_stop"],
        atom_start=value["atom_start"],
        atom_stop=value["atom_stop"],
        batch_size=value["batch_size"],
    )


def _validate_alignment(payload: Mapping[str, Any], signature: Mapping[str, Any]) -> None:
    shape = validate_prediction_payload(payload)
    if payload["observables"] != signature["observables"]:
        raise HardFailure("prediction payload and shard signature observables are not aligned")
    if payload["member_ids"] != signature["member_ids"]:
        raise HardFailure("prediction payload and shard signature members are not aligned")
    expected_structures = signature["structure_stop"] - signature["structure_start"]
    expected_atoms = signature["atom_stop"] - signature["atom_start"]
    if shape.structures != expected_structures or shape.atoms != expected_atoms:
        raise HardFailure("prediction payload and shard signature ranges are not aligned")


def write_prediction_shard(
    layout: DerivedLayout,
    index: int,
    payload: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> Path:
    tensor_path = layout.prediction_shard(index)
    manifest_path = layout.prediction_shard_manifest(index)
    if tensor_path.exists() or manifest_path.exists():
        raise HardFailure(f"refusing to overwrite prediction shard {index}")
    validated = _validated_signature(signature)
    if validated["shard_index"] != index:
        raise HardFailure("prediction shard index does not match its signature")
    _validate_alignment(payload, validated)
    atomic_torch_save(tensor_path, dict(payload))
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "fge.derived-prediction-shard.v1",
            "signature": validated,
            "artifact": {"path": tensor_path.name, "sha256": sha256_file(tensor_path)},
        },
    )
    return tensor_path


def verify_prediction_shard_if_present(
    layout: DerivedLayout, index: int, expected_signature: Mapping[str, Any]
) -> bool:
    tensor_path = layout.prediction_shard(index)
    manifest_path = layout.prediction_shard_manifest(index)
    if not tensor_path.exists() and not manifest_path.exists():
        return False
    if not tensor_path.is_file() or not manifest_path.is_file():
        raise HardFailure(f"prediction shard {index} is partially present")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HardFailure(f"prediction shard {index} manifest cannot be loaded") from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "signature",
        "artifact",
    }:
        raise HardFailure(f"prediction shard {index} manifest is invalid")
    if manifest.get("schema_version") != "fge.derived-prediction-shard.v1":
        raise HardFailure(f"prediction shard {index} manifest schema is invalid")
    actual_signature = _validated_signature(manifest.get("signature"))
    expected = _validated_signature(expected_signature)
    if actual_signature != expected:
        raise HardFailure(f"prediction shard {index} signature mismatch")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("path") != tensor_path.name:
        raise HardFailure(f"prediction shard {index} manifest artifact is invalid")
    if artifact.get("sha256") != sha256_file(tensor_path):
        raise HardFailure(f"prediction shard {index} SHA-256 mismatch")
    try:
        payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise HardFailure(f"prediction shard {index} cannot be loaded") from exc
    if not isinstance(payload, Mapping):
        raise HardFailure(f"prediction shard {index} payload is not a mapping")
    _validate_alignment(payload, actual_signature)
    return True
