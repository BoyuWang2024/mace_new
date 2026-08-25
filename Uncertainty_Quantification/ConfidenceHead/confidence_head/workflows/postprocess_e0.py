"""Post-inference E0 correction workflow for external ConfidenceHead results."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import ase.io
import numpy as np
import torch

from ..artifacts import atomic_json_dump, atomic_torch_save, load_torch_artifact
from ..binning import labels_from_thresholds
from ..cache import CacheManifest, iter_cache_batches
from ..data import load_dataset
from ..e0_corrections import composition_matrix, energy_errors_per_atom
from ..evaluation_artifacts import (
    EVALUATION_FORMULA_VERSION,
    EVALUATION_SCHEMA_VERSION,
    validate_prediction_payload,
)
from ..identity import sha256_file, stable_id
from ..metrics import branch_metrics, validate_metric_payload


E0_FORMULA_VERSION = "confidence_head_e0_postprocess_v1"
_METHODS = frozenset({"e0_replace", "e0_reestimate"})
_INPUT_HASH_NAMES = frozenset(
    {
        "source_predictions.pt",
        "source_manifest.json",
        "shared_inputs.pt",
        "correction.pt",
    }
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class E0WorkflowError(RuntimeError):
    """E0 workflow inputs, identities, or committed artifacts conflict."""


@dataclass(frozen=True)
class EnergyInputs:
    """Aligned raw MACE total energies and composition for one split."""

    structure_ids: tuple[str, ...]
    structure_indices: np.ndarray
    num_atoms: np.ndarray
    composition: np.ndarray
    raw_total: np.ndarray
    reference_total: np.ndarray


@dataclass(frozen=True)
class TestMetadata:
    """Test-only metadata not persisted in the feature cache."""

    structure_ids: tuple[str, ...]
    source_indices: np.ndarray
    formulas: tuple[str, ...]
    atomization_total: np.ndarray


@dataclass(frozen=True)
class DerivedEvaluationPaths:
    """Canonical artifacts for one method and one energy order."""

    root: Path
    predictions: Path
    metrics: Path
    manifest: Path

    @classmethod
    def from_root(cls, root: Path) -> "DerivedEvaluationPaths":
        directory = Path(root)
        return cls(
            root=directory,
            predictions=directory / "predictions.pt",
            metrics=directory / "metrics.json",
            manifest=directory / "manifest.json",
        )

    @property
    def outputs(self) -> tuple[Path, Path, Path]:
        return (self.predictions, self.metrics, self.manifest)


def _numpy(tensor: torch.Tensor, *, dtype: np.dtype[Any]) -> np.ndarray:
    if not isinstance(tensor, torch.Tensor):
        raise E0WorkflowError("cache value must be a tensor")
    return np.asarray(tensor.detach().cpu().numpy(), dtype=dtype)


def collect_energy_inputs(
    cache: CacheManifest,
    *,
    split: str,
    batch_size: int,
    supported_atomic_numbers: Sequence[int],
) -> EnergyInputs:
    """Collect a committed cache split without changing its stored tensors."""
    if split not in cache.splits:
        raise E0WorkflowError(f"cache split is missing: {split}")
    structure_ids: list[str] = []
    indices: list[np.ndarray] = []
    atom_counts: list[np.ndarray] = []
    atomic_numbers: list[np.ndarray] = []
    raw_total: list[np.ndarray] = []
    reference_total: list[np.ndarray] = []
    for batch in iter_cache_batches(cache, split, batch_size):
        ids = tuple(batch.structure_id)
        offsets = _numpy(batch.atom_offsets, dtype=np.int64)
        if offsets.shape != (len(ids) + 1,):
            raise E0WorkflowError("cache atom offsets differ from structure IDs")
        structure_ids.extend(ids)
        indices.append(_numpy(batch.structure_index, dtype=np.int64))
        atom_counts.append(np.diff(offsets))
        numbers = _numpy(batch.atomic_numbers, dtype=np.int64)
        for index in range(len(ids)):
            atomic_numbers.append(numbers[offsets[index] : offsets[index + 1]])
        raw_total.append(_numpy(batch.energy_prediction, dtype=np.float64))
        reference_total.append(_numpy(batch.energy_reference, dtype=np.float64))
    if not structure_ids:
        raise E0WorkflowError("cache split contains no structures")
    joined_indices = np.concatenate(indices)
    expected_indices = np.arange(len(structure_ids), dtype=np.int64)
    if not np.array_equal(joined_indices, expected_indices):
        raise E0WorkflowError("cache structure indices differ from stable order")
    if len(set(structure_ids)) != len(structure_ids):
        raise E0WorkflowError("cache structure IDs must be unique")
    counts = np.concatenate(atom_counts).astype(np.int64, copy=False)
    raw = np.concatenate(raw_total).astype(np.float64, copy=False)
    reference = np.concatenate(reference_total).astype(np.float64, copy=False)
    if not (counts.shape == raw.shape == reference.shape == joined_indices.shape):
        raise E0WorkflowError("cache energy vectors differ from structure count")
    if np.any(counts <= 0) or not np.isfinite(raw).all() or not np.isfinite(reference).all():
        raise E0WorkflowError("cache energy inputs are invalid")
    return EnergyInputs(
        structure_ids=tuple(structure_ids),
        structure_indices=joined_indices,
        num_atoms=counts,
        composition=composition_matrix(
            atomic_numbers, supported_atomic_numbers
        ),
        raw_total=raw,
        reference_total=reference,
    )


def _source_indices(path: Path | None, count: int) -> np.ndarray:
    if path is None:
        return np.arange(count, dtype=np.int64)
    source = Path(path)
    try:
        with source.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ("compatible_index", "source_index"):
                raise E0WorkflowError("source-index columns differ")
            rows = list(reader)
    except E0WorkflowError:
        raise
    except Exception as error:
        raise E0WorkflowError(f"could not read source-index file: {error}") from error
    if len(rows) != count:
        raise E0WorkflowError("source-index row count differs")
    try:
        compatible = np.asarray(
            [int(row["compatible_index"]) for row in rows], dtype=np.int64
        )
        original = np.asarray(
            [int(row["source_index"]) for row in rows], dtype=np.int64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise E0WorkflowError("source-index values are invalid") from error
    if not np.array_equal(compatible, np.arange(count, dtype=np.int64)):
        raise E0WorkflowError("source-index compatible order differs")
    if np.any(original < 0) or len(set(original.tolist())) != count:
        raise E0WorkflowError("source-index originals are invalid")
    return original


def load_test_metadata(
    *,
    path: Path,
    expected_sha256: str,
    supported_atomic_numbers: Sequence[int],
    expected_structure_ids: Sequence[str],
    atomization_energy_key: str,
    source_index_path: Path | None,
) -> TestMetadata:
    """Load test atomization values and prove extxyz/cache alignment."""
    handle = load_dataset(
        path, expected_sha256, supported_atomic_numbers
    )
    expected_ids = tuple(expected_structure_ids)
    if handle.structure_ids != expected_ids:
        raise E0WorkflowError("test structure IDs differ from cache")
    try:
        structures = ase.io.read(handle.path, index=":")
    except Exception as error:
        raise E0WorkflowError(f"could not read test metadata: {error}") from error
    if not isinstance(structures, list):
        structures = [structures]
    if len(structures) != len(expected_ids):
        raise E0WorkflowError("test metadata structure count differs")
    formulas: list[str] = []
    atomization: list[float] = []
    for index, atoms in enumerate(structures):
        if atomization_energy_key not in atoms.info:
            raise E0WorkflowError(
                f"test structure {index} is missing {atomization_energy_key}"
            )
        value = np.asarray(atoms.info[atomization_energy_key], dtype=np.float64)
        if value.shape != () or not np.isfinite(value).all():
            raise E0WorkflowError(
                f"test structure {index} atomization energy is invalid"
            )
        formulas.append(atoms.get_chemical_formula())
        atomization.append(float(value))
    return TestMetadata(
        structure_ids=expected_ids,
        source_indices=_source_indices(source_index_path, len(expected_ids)),
        formulas=tuple(formulas),
        atomization_total=np.asarray(atomization, dtype=np.float64),
    )


def extract_model_e0(
    model: Any,
    *,
    atomic_numbers: Sequence[int],
    selected_head: str,
) -> np.ndarray:
    """Extract one checkpoint E0 row in checkpoint element order."""
    heads = tuple(getattr(model, "heads", ()))
    if not heads or selected_head not in heads or len(set(heads)) != len(heads):
        raise E0WorkflowError("checkpoint head identity is invalid")
    block = getattr(model, "atomic_energies_fn", None)
    values = getattr(block, "atomic_energies", None)
    if not isinstance(values, torch.Tensor):
        raise E0WorkflowError("checkpoint atomic energies are missing")
    expected_shape = (len(heads), len(tuple(atomic_numbers)))
    matrix = torch.atleast_2d(values.detach().cpu())
    if tuple(matrix.shape) != expected_shape:
        raise E0WorkflowError(
            f"checkpoint E0 shape differs: {tuple(matrix.shape)} != {expected_shape}"
        )
    row = matrix[heads.index(selected_head)].to(torch.float64)
    if not bool(torch.isfinite(row).all().item()):
        raise E0WorkflowError("checkpoint E0 values must be finite")
    return np.asarray(row.numpy(), dtype=np.float64)


def derive_energy_predictions(
    source: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, str],
    corrected_total: Sequence[float],
    reference_total: Sequence[float],
    num_atoms: Sequence[int],
    thresholds: torch.Tensor,
) -> dict[str, Any]:
    """Preserve frozen head outputs and replace energy errors and labels."""
    validated = validate_prediction_payload(
        dict(source), expected_identity=expected_identity
    )
    if tuple(validated["enabled_branches"]) != ("energy",):
        raise E0WorkflowError("derived energy source must be energy-only")
    if len(validated["structure_ids"]) != len(tuple(corrected_total)):
        raise E0WorkflowError("corrected energy count differs from predictions")
    try:
        errors = torch.from_numpy(
            energy_errors_per_atom(
                reference_total=reference_total,
                prediction_total=corrected_total,
                num_atoms=num_atoms,
            )
        )
        labels = labels_from_thresholds(errors, thresholds)
    except (ValueError, TypeError) as error:
        raise E0WorkflowError(f"could not derive energy labels: {error}") from error
    energy = dict(validated["energy"])
    energy["errors"] = errors
    energy["labels"] = labels
    derived = dict(validated)
    derived["energy"] = energy
    return validate_prediction_payload(
        derived, expected_identity=expected_identity
    )


def energy_metrics_payload(
    predictions: Mapping[str, Any], representatives: torch.Tensor
) -> dict[str, Any]:
    """Build the standard publication metric payload for derived energy."""
    identity = dict(predictions["identity"])
    validated = validate_prediction_payload(
        dict(predictions), expected_identity=identity
    )
    if tuple(validated["enabled_branches"]) != ("energy",):
        raise E0WorkflowError("energy metrics require an energy-only prediction")
    branch = validated["energy"]
    try:
        metrics = branch_metrics(
            branch["logits"],
            branch["labels"],
            branch["errors"],
            representatives,
        )
    except ValueError as error:
        raise E0WorkflowError(f"could not compute energy metrics: {error}") from error
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "formula_version": EVALUATION_FORMULA_VERSION,
        "split": "test",
        "identity": identity,
        "branches": {"energy": metrics},
    }


def _read_json(path: Path, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise E0WorkflowError(f"could not read {where}: {error}") from error
    if type(value) is not dict:
        raise E0WorkflowError(f"{where} must be a mapping")
    return value


def _payload_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.device.type == right.device.type == "cpu"
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_payload_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_payload_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _hashes(value: Mapping[str, str]) -> dict[str, str]:
    if set(value) != set(_INPUT_HASH_NAMES):
        raise E0WorkflowError("derived evaluation input hash names differ")
    result = dict(sorted(value.items()))
    if any(type(digest) is not str or _HEX64.fullmatch(digest) is None for digest in result.values()):
        raise E0WorkflowError("derived evaluation input hashes are invalid")
    return result


def _validate_metrics(
    value: Mapping[str, Any], expected_identity: Mapping[str, str]
) -> dict[str, Any]:
    expected_keys = {
        "schema_version", "formula_version", "split", "identity", "branches"
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise E0WorkflowError("derived metrics keys differ")
    if (
        value["schema_version"] != EVALUATION_SCHEMA_VERSION
        or value["formula_version"] != EVALUATION_FORMULA_VERSION
        or value["split"] != "test"
        or value["identity"] != dict(expected_identity)
    ):
        raise E0WorkflowError("derived metrics identity differs")
    branches = value["branches"]
    if type(branches) is not dict or set(branches) != {"energy"}:
        raise E0WorkflowError("derived metrics branches differ")
    try:
        validated = validate_metric_payload(branches["energy"])
    except ValueError as error:
        raise E0WorkflowError(f"derived energy metrics are invalid: {error}") from error
    result = dict(value)
    result["branches"] = {"energy": validated}
    return result


def _manifest_base(
    *,
    method: str,
    head_key: str,
    identity: Mapping[str, str],
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if method not in _METHODS:
        raise E0WorkflowError("unsupported E0 method")
    if head_key not in {f"energy_order{order}" for order in range(1, 9)}:
        raise E0WorkflowError("unsupported energy head key")
    inputs = _hashes(input_hashes)
    source_identity = dict(identity)
    derived_id = stable_id(
        {
            "formula_version": E0_FORMULA_VERSION,
            "method": method,
            "head_key": head_key,
            "source_identity": source_identity,
            "input_sha256": inputs,
        }
    )
    return {
        "schema_version": 1,
        "formula_version": E0_FORMULA_VERSION,
        "derived_id": derived_id,
        "method": method,
        "head_key": head_key,
        "source_identity": source_identity,
        "input_sha256": inputs,
    }


def publish_derived_evaluation(
    paths: DerivedEvaluationPaths,
    *,
    predictions: Mapping[str, Any],
    metrics: Mapping[str, Any],
    method: str,
    head_key: str,
    input_hashes: Mapping[str, str],
) -> Path:
    """Atomically publish or byte-validate one derived energy evaluation."""
    identity = dict(predictions["identity"])
    validated_predictions = validate_prediction_payload(
        dict(predictions), expected_identity=identity
    )
    validated_metrics = _validate_metrics(dict(metrics), identity)
    base = _manifest_base(
        method=method,
        head_key=head_key,
        identity=identity,
        input_hashes=input_hashes,
    )
    evidence = tuple(path.exists() for path in paths.outputs)
    if any(evidence):
        if not all(evidence):
            raise E0WorkflowError("derived evaluation evidence is partial")
        manifest = _read_json(paths.manifest, "derived evaluation manifest")
        if {key: manifest.get(key) for key in base} != base:
            raise E0WorkflowError("derived evaluation identity differs")
        outputs = manifest.get("output_sha256")
        expected_outputs = {
            paths.predictions.name: sha256_file(paths.predictions),
            paths.metrics.name: sha256_file(paths.metrics),
        }
        if outputs != expected_outputs:
            raise E0WorkflowError("derived evaluation output hashes differ")
        persisted_predictions = validate_prediction_payload(
            load_torch_artifact(paths.predictions), expected_identity=identity
        )
        _validate_metrics(_read_json(paths.metrics, "derived metrics"), identity)
        if not _payload_equal(persisted_predictions, validated_predictions):
            raise E0WorkflowError("derived prediction content differs")
        if _read_json(paths.metrics, "derived metrics") != validated_metrics:
            raise E0WorkflowError("derived metric content differs")
        return paths.manifest

    paths.root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        atomic_torch_save(paths.predictions, validated_predictions)
        written.append(paths.predictions)
        atomic_json_dump(paths.metrics, validated_metrics)
        written.append(paths.metrics)
        manifest = dict(base)
        manifest["output_sha256"] = {
            paths.predictions.name: sha256_file(paths.predictions),
            paths.metrics.name: sha256_file(paths.metrics),
        }
        atomic_json_dump(paths.manifest, manifest)
        written.append(paths.manifest)
    except Exception:
        for path in reversed(written):
            path.unlink(missing_ok=True)
        raise
    return publish_derived_evaluation(
        paths,
        predictions=validated_predictions,
        metrics=validated_metrics,
        method=method,
        head_key=head_key,
        input_hashes=input_hashes,
    )


__all__ = [
    "DerivedEvaluationPaths",
    "E0WorkflowError",
    "EnergyInputs",
    "TestMetadata",
    "collect_energy_inputs",
    "derive_energy_predictions",
    "energy_metrics_payload",
    "extract_model_e0",
    "load_test_metadata",
    "publish_derived_evaluation",
]
