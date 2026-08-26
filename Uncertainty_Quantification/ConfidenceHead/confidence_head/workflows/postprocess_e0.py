"""Post-inference E0 correction workflow for external ConfidenceHead results."""

from __future__ import annotations

import csv
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import ase.io
import numpy as np
import torch

from ..artifacts import atomic_json_dump, atomic_torch_save, load_torch_artifact
from ..backbone import load_frozen_backbone
from ..binning import labels_from_thresholds
from ..cache import CacheManifest, iter_cache_batches, load_complete_cache
from ..data import load_dataset
from ..e0_corrections import (
    apply_e0_reestimate,
    apply_e0_replace,
    composition_matrix,
    energy_errors_per_atom,
    fit_e0_reestimate,
)
from ..evaluation_artifacts import (
    EVALUATION_FORMULA_VERSION,
    EVALUATION_SCHEMA_VERSION,
    validate_or_reuse_evaluation,
    validate_prediction_payload,
)
from ..external_config import E0PostprocessConfig, ExternalInferenceConfig
from ..identity import sha256_file, stable_id
from ..metrics import branch_metrics, validate_metric_payload
from .build_cache import REPOSITORY_ROOT
from .build_external_cache import (
    external_cache_id,
    external_cache_root,
    run_build_external_cache,
)
from .evaluate import evaluation_input_hashes, load_evaluation_inputs
from .evaluate_external import (
    _paths as external_evaluation_paths,
    resolve_head_config,
    run_evaluate_external,
)


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
_SHARED_INPUT_NAMES = frozenset(
    {
        "e0_config.yaml",
        "validation_config.yaml",
        "test_config.yaml",
        "validation_cache_manifest.json",
        "test_cache_manifest.json",
        "checkpoint.model",
    }
)


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
class ValidationIsolation:
    """Validation calibration rows after removing test content overlap."""

    inputs: EnergyInputs
    excluded_structure_ids: tuple[str, ...]
    excluded_structure_indices: np.ndarray


@dataclass(frozen=True)
class TestMetadata:
    """Test-only metadata not persisted in the feature cache."""

    structure_ids: tuple[str, ...]
    source_indices: np.ndarray
    formulas: tuple[str, ...]
    atomization_total: np.ndarray


@dataclass(frozen=True)
class EnergyEvaluationSource:
    """One immutable source Energy-only evaluation and frozen bins."""

    head_key: str
    predictions: Mapping[str, Any]
    predictions_path: Path
    manifest_path: Path
    thresholds: torch.Tensor
    representatives: torch.Tensor


@dataclass(frozen=True)
class E0MethodResult:
    """One reconstructable correction artifact and corrected test totals."""

    method: str
    corrected_total: np.ndarray
    artifact: dict[str, Any]


@dataclass(frozen=True)
class E0PostprocessOutputs:
    """Published manifests for one complete two-method E0 experiment."""

    shared_manifest: Path
    method_manifests: dict[str, Path]
    plot_manifests: dict[str, Path]


@dataclass(frozen=True)
class _TestEvaluationSources:
    force_context: Any
    force_predictions: dict[str, Any]
    force_predictions_path: Path
    force_manifest_path: Path
    energy_sources: dict[int, EnergyEvaluationSource]


@dataclass(frozen=True)
class _PreparedE0Sources:
    validation: EnergyInputs
    test: EnergyInputs
    metadata: TestMetadata
    supported_atomic_numbers: tuple[int, ...]
    model_e0: np.ndarray
    input_files: dict[str, Path]
    evaluations: _TestEvaluationSources
    excluded_validation_structure_ids: tuple[str, ...]
    excluded_validation_structure_indices: np.ndarray


@dataclass(frozen=True)
class SharedArtifactPaths:
    """Canonical shared raw-energy inputs and force reference artifacts."""

    root: Path
    inputs: Path
    force_manifest: Path
    manifest: Path

    @classmethod
    def from_output_root(cls, output_root: Path) -> "SharedArtifactPaths":
        root = Path(output_root) / "shared"
        return cls(
            root=root,
            inputs=root / "shared_inputs.pt",
            force_manifest=root / "force" / "manifest.json",
            manifest=root / "manifest.json",
        )

    @property
    def outputs(self) -> tuple[Path, Path, Path]:
        return (self.inputs, self.force_manifest, self.manifest)


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


def _sample_content_id(sample_id: str) -> str:
    if not isinstance(sample_id, str):
        raise E0WorkflowError("sample structure ID must be a string")
    content_id, marker, occurrence = sample_id.rpartition("#")
    if marker != "#" or not content_id or not occurrence.isdigit():
        raise E0WorkflowError("sample structure ID format differs")
    return content_id


def isolate_validation_inputs(
    validation: EnergyInputs, test: EnergyInputs
) -> ValidationIsolation:
    """Remove validation structures whose content also appears in test."""
    _energy_inputs_payload(validation, "validation before isolation")
    _energy_inputs_payload(test, "test before isolation")
    test_content = {
        _sample_content_id(sample_id) for sample_id in test.structure_ids
    }
    keep = np.asarray(
        [
            _sample_content_id(sample_id) not in test_content
            for sample_id in validation.structure_ids
        ],
        dtype=bool,
    )
    excluded = ~keep
    if not bool(np.any(excluded)):
        return ValidationIsolation(
            inputs=validation,
            excluded_structure_ids=(),
            excluded_structure_indices=np.empty(0, dtype=np.int64),
        )
    if not bool(np.any(keep)):
        raise E0WorkflowError(
            "validation/test overlap removes every calibration structure"
        )
    isolated = EnergyInputs(
        structure_ids=tuple(
            sample_id
            for sample_id, retained in zip(validation.structure_ids, keep)
            if retained
        ),
        structure_indices=np.arange(int(np.sum(keep)), dtype=np.int64),
        num_atoms=np.asarray(validation.num_atoms, dtype=np.int64)[keep].copy(),
        composition=np.asarray(validation.composition, dtype=np.float64)[
            keep
        ].copy(),
        raw_total=np.asarray(validation.raw_total, dtype=np.float64)[keep].copy(),
        reference_total=np.asarray(
            validation.reference_total, dtype=np.float64
        )[keep].copy(),
    )
    _energy_inputs_payload(isolated, "validation after isolation")
    return ValidationIsolation(
        inputs=isolated,
        excluded_structure_ids=tuple(
            sample_id
            for sample_id, removed in zip(validation.structure_ids, excluded)
            if removed
        ),
        excluded_structure_indices=np.asarray(
            validation.structure_indices, dtype=np.int64
        )[excluded].copy(),
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


def _tensor(values: Sequence[float] | np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float64).copy())


def _aligned_inputs(
    validation: EnergyInputs,
    test: EnergyInputs,
    metadata: TestMetadata,
    supported_atomic_numbers: Sequence[int],
) -> tuple[int, ...]:
    supported = tuple(int(number) for number in supported_atomic_numbers)
    if not supported or len(set(supported)) != len(supported):
        raise E0WorkflowError("supported atomic-number order is invalid")
    if validation.composition.shape[1] != len(supported):
        raise E0WorkflowError("validation composition columns differ")
    if test.composition.shape[1] != len(supported):
        raise E0WorkflowError("test composition columns differ")
    if test.structure_ids != metadata.structure_ids:
        raise E0WorkflowError("test metadata structure IDs differ")
    validation_content = {
        _sample_content_id(sample_id) for sample_id in validation.structure_ids
    }
    test_content = {
        _sample_content_id(sample_id) for sample_id in test.structure_ids
    }
    if validation_content & test_content:
        raise E0WorkflowError("validation and test content overlap")
    return supported


def compute_e0_corrections(
    *,
    validation: EnergyInputs,
    test: EnergyInputs,
    metadata: TestMetadata,
    model_e0: Sequence[float] | np.ndarray,
    supported_atomic_numbers: Sequence[int],
    excluded_validation_structure_ids: Sequence[str] = (),
    excluded_validation_structure_indices: Sequence[int] = (),
) -> dict[str, E0MethodResult]:
    """Compute both corrections while fitting reestimate on validation only."""
    supported = _aligned_inputs(validation, test, metadata, supported_atomic_numbers)
    checkpoint_e0 = np.asarray(model_e0, dtype=np.float64)
    excluded_ids = tuple(excluded_validation_structure_ids)
    excluded_indices = np.asarray(
        excluded_validation_structure_indices, dtype=np.int64
    )
    if (
        excluded_indices.ndim != 1
        or excluded_indices.shape != (len(excluded_ids),)
        or any(not isinstance(sample_id, str) for sample_id in excluded_ids)
        or len(set(excluded_ids)) != len(excluded_ids)
        or np.any(excluded_indices < 0)
        or len(set(excluded_indices.tolist())) != len(excluded_ids)
    ):
        raise E0WorkflowError("excluded validation structures are invalid")
    try:
        replacement = apply_e0_replace(
            raw_total=test.raw_total,
            reference_total=test.reference_total,
            atomization_total=metadata.atomization_total,
            composition=test.composition,
            model_e0=checkpoint_e0,
        )
        fit = fit_e0_reestimate(
            composition=validation.composition,
            reference_total=validation.reference_total,
            raw_total=validation.raw_total,
        )
        reestimated = apply_e0_reestimate(
            raw_total=test.raw_total,
            composition=test.composition,
            delta_e0=fit.delta_e0,
        )
    except ValueError as error:
        raise E0WorkflowError(f"could not compute E0 corrections: {error}") from error
    common = {
        "schema_version": 1,
        "formula_version": E0_FORMULA_VERSION,
        "supported_atomic_numbers": torch.tensor(supported, dtype=torch.int64),
        "model_e0": _tensor(checkpoint_e0),
    }
    replacement_artifact = {
        **common,
        "method": "e0_replace",
        "field_sources": {
            "mad_baseline": "test.reference_energy-test.atomization_energy",
            "model_baseline": "test.composition@checkpoint.model_e0",
        },
        "model_baseline": _tensor(replacement.model_baseline),
        "mad_baseline": _tensor(replacement.mad_baseline),
        "correction": _tensor(replacement.correction),
        "corrected_total": _tensor(replacement.corrected_total),
    }
    reestimate_artifact = {
        **common,
        "method": "e0_reestimate",
        "fit_split": "validation",
        "fit_weighting": "unweighted_total_energy_ols",
        "validation_structure_count": len(validation.structure_ids),
        "excluded_validation_structure_ids": excluded_ids,
        "excluded_validation_structure_indices": torch.from_numpy(
            excluded_indices.copy()
        ),
        "delta_e0": _tensor(fit.delta_e0),
        "new_e0": _tensor(checkpoint_e0 + fit.delta_e0),
        "rank": fit.rank,
        "singular_values": _tensor(fit.singular_values),
        "condition_number": fit.condition_number,
        "residual_sum_squares": fit.residual_sum_squares,
        "residual_rmse": fit.residual_rmse,
        "residual_max_abs": fit.residual_max_abs,
        "corrected_validation_total": _tensor(fit.corrected_validation_total),
        "correction": _tensor(reestimated.correction),
        "corrected_total": _tensor(reestimated.corrected_total),
    }
    return {
        "e0_replace": E0MethodResult(
            "e0_replace", replacement.corrected_total, replacement_artifact
        ),
        "e0_reestimate": E0MethodResult(
            "e0_reestimate", reestimated.corrected_total, reestimate_artifact
        ),
    }


def _jsonable_artifact(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_artifact(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_artifact(item) for item in value]
    return value


def _energy_inputs_payload(value: EnergyInputs, where: str) -> dict[str, Any]:
    count = len(value.structure_ids)
    indices = np.asarray(value.structure_indices, dtype=np.int64)
    num_atoms = np.asarray(value.num_atoms, dtype=np.int64)
    composition = np.asarray(value.composition, dtype=np.float64)
    raw = np.asarray(value.raw_total, dtype=np.float64)
    reference = np.asarray(value.reference_total, dtype=np.float64)
    if count == 0 or len(set(value.structure_ids)) != count:
        raise E0WorkflowError(f"{where} structure IDs are invalid")
    if indices.shape != (count,) or not np.array_equal(
        indices, np.arange(count, dtype=np.int64)
    ):
        raise E0WorkflowError(f"{where} structure indices differ")
    if (
        num_atoms.shape != (count,)
        or composition.ndim != 2
        or composition.shape[0] != count
        or raw.shape != (count,)
        or reference.shape != (count,)
    ):
        raise E0WorkflowError(f"{where} shared input shapes differ")
    if (
        np.any(num_atoms <= 0)
        or np.any(composition < 0.0)
        or not np.equal(composition, np.floor(composition)).all()
        or not np.array_equal(composition.sum(axis=1), num_atoms.astype(np.float64))
        or not np.isfinite(composition).all()
        or not np.isfinite(raw).all()
        or not np.isfinite(reference).all()
    ):
        raise E0WorkflowError(f"{where} shared input values are invalid")
    return {
        "structure_ids": tuple(value.structure_ids),
        "structure_indices": torch.from_numpy(indices.copy()),
        "num_atoms": torch.from_numpy(num_atoms.copy()),
        "composition": _tensor(composition),
        "raw_total": _tensor(raw),
        "reference_total": _tensor(reference),
    }


def _shared_payload(
    *,
    validation: EnergyInputs,
    test: EnergyInputs,
    metadata: TestMetadata,
    supported_atomic_numbers: Sequence[int],
    model_e0: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    supported = _aligned_inputs(validation, test, metadata, supported_atomic_numbers)
    checkpoint_e0 = np.asarray(model_e0, dtype=np.float64)
    if checkpoint_e0.shape != (len(supported),) or not np.isfinite(checkpoint_e0).all():
        raise E0WorkflowError("shared model E0 values are invalid")
    count = len(test.structure_ids)
    source_indices = np.asarray(metadata.source_indices, dtype=np.int64)
    atomization = np.asarray(metadata.atomization_total, dtype=np.float64)
    if (
        source_indices.shape != (count,)
        or np.any(source_indices < 0)
        or len(set(source_indices.tolist())) != count
        or len(metadata.formulas) != count
        or any(not isinstance(formula, str) or not formula for formula in metadata.formulas)
        or atomization.shape != (count,)
        or not np.isfinite(atomization).all()
    ):
        raise E0WorkflowError("shared test metadata is invalid")
    test_payload = _energy_inputs_payload(test, "test")
    test_payload.update(
        {
            "source_indices": torch.from_numpy(source_indices.copy()),
            "formulas": tuple(metadata.formulas),
            "atomization_total": _tensor(atomization),
        }
    )
    return {
        "schema_version": 1,
        "formula_version": E0_FORMULA_VERSION,
        "supported_atomic_numbers": torch.tensor(supported, dtype=torch.int64),
        "model_e0": _tensor(checkpoint_e0),
        "validation": _energy_inputs_payload(validation, "validation"),
        "test": test_payload,
    }


def _relative_path(path: Path, start: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), start=Path(start).resolve())).as_posix()


def publish_shared_inputs(
    *,
    output_root: Path,
    validation: EnergyInputs,
    test: EnergyInputs,
    metadata: TestMetadata,
    supported_atomic_numbers: Sequence[int],
    model_e0: Sequence[float] | np.ndarray,
    input_files: Mapping[str, Path],
    force_predictions_path: Path,
    force_manifest_path: Path,
) -> SharedArtifactPaths:
    """Publish or validate one shared raw-input artifact and force reference."""
    if set(input_files) != set(_SHARED_INPUT_NAMES):
        raise E0WorkflowError("shared input file names differ")
    bound_inputs = {name: Path(path) for name, path in input_files.items()}
    missing = [name for name, path in bound_inputs.items() if not path.is_file()]
    force_predictions = Path(force_predictions_path)
    force_manifest = Path(force_manifest_path)
    if missing or not force_predictions.is_file() or not force_manifest.is_file():
        raise E0WorkflowError("shared source files are missing")
    paths = SharedArtifactPaths.from_output_root(output_root)
    payload = _shared_payload(
        validation=validation,
        test=test,
        metadata=metadata,
        supported_atomic_numbers=supported_atomic_numbers,
        model_e0=model_e0,
    )
    force_reference = {
        "schema_version": 1,
        "formula_version": E0_FORMULA_VERSION,
        "source_paths": {
            "predictions.pt": _relative_path(
                force_predictions, paths.force_manifest.parent
            ),
            "manifest.json": _relative_path(
                force_manifest, paths.force_manifest.parent
            ),
        },
        "source_sha256": {
            "predictions.pt": sha256_file(force_predictions),
            "manifest.json": sha256_file(force_manifest),
        },
    }
    base = {
        "schema_version": 1,
        "formula_version": E0_FORMULA_VERSION,
        "shared_id": stable_id(_jsonable_artifact(payload)),
        "validation_structure_count": len(validation.structure_ids),
        "test_structure_count": len(test.structure_ids),
        "input_paths": {
            name: _relative_path(path, paths.root)
            for name, path in sorted(bound_inputs.items())
        },
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(bound_inputs.items())
        },
        "force_source": force_reference,
    }
    evidence = tuple(path.exists() for path in paths.outputs)
    if any(evidence):
        if not all(evidence):
            raise E0WorkflowError("shared artifact evidence is partial")
        manifest = _read_json(paths.manifest, "shared artifact manifest")
        if {key: manifest.get(key) for key in base} != base:
            raise E0WorkflowError("shared artifact identity differs")
        output_hashes = manifest.get("output_sha256")
        expected_names = {
            paths.inputs.relative_to(paths.root).as_posix(),
            paths.force_manifest.relative_to(paths.root).as_posix(),
        }
        actual_names = {
            path.relative_to(paths.root).as_posix()
            for path in paths.root.rglob("*")
            if path.is_file() and path != paths.manifest
        }
        if type(output_hashes) is not dict or set(output_hashes) != expected_names:
            raise E0WorkflowError("shared artifact output hashes differ")
        if actual_names != expected_names:
            raise E0WorkflowError("shared artifact output file set differs")
        for relative, digest in output_hashes.items():
            if type(digest) is not str or _HEX64.fullmatch(digest) is None:
                raise E0WorkflowError("shared artifact output hashes differ")
            if sha256_file(paths.root / relative) != digest:
                raise E0WorkflowError("shared artifact output hashes differ")
        if not _payload_equal(load_torch_artifact(paths.inputs), payload):
            raise E0WorkflowError("shared input artifact content differs")
        if _read_json(paths.force_manifest, "shared force manifest") != force_reference:
            raise E0WorkflowError("shared force reference differs")
        return paths
    if paths.root.exists() and any(paths.root.iterdir()):
        raise E0WorkflowError("shared artifact directory contains unknown evidence")
    paths.force_manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(paths.inputs, payload)
    if not _payload_equal(load_torch_artifact(paths.inputs), payload):
        raise E0WorkflowError("new shared input artifact content differs")
    atomic_json_dump(paths.force_manifest, force_reference)
    output_hashes = {
        paths.inputs.relative_to(paths.root).as_posix(): sha256_file(paths.inputs),
        paths.force_manifest.relative_to(paths.root).as_posix(): sha256_file(
            paths.force_manifest
        ),
    }
    atomic_json_dump(paths.manifest, {**base, "output_sha256": output_hashes})
    return publish_shared_inputs(
        output_root=output_root,
        validation=validation,
        test=test,
        metadata=metadata,
        supported_atomic_numbers=supported_atomic_numbers,
        model_e0=model_e0,
        input_files=input_files,
        force_predictions_path=force_predictions,
        force_manifest_path=force_manifest,
    )


def _csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise E0WorkflowError(f"existing method artifact differs: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise E0WorkflowError(f"temporary method artifact already exists: {temporary}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _source_hashes(
    sources: Mapping[int, EnergyEvaluationSource],
) -> dict[str, dict[str, str]]:
    if set(sources) != set(range(1, 9)):
        raise E0WorkflowError("energy sources must contain exact orders 1 through 8")
    hashes: dict[str, dict[str, str]] = {}
    for order in range(1, 9):
        source = sources[order]
        expected_key = f"energy_order{order}"
        if source.head_key != expected_key:
            raise E0WorkflowError(f"energy source head differs at order {order}")
        if not source.predictions_path.is_file() or not source.manifest_path.is_file():
            raise E0WorkflowError(f"energy source files are missing at order {order}")
        identity = dict(source.predictions["identity"])
        validated = validate_prediction_payload(
            dict(source.predictions), expected_identity=identity
        )
        persisted = validate_prediction_payload(
            load_torch_artifact(source.predictions_path),
            expected_identity=identity,
        )
        if not _payload_equal(validated, persisted):
            raise E0WorkflowError(
                f"energy source prediction content differs at order {order}"
            )
        hashes[expected_key] = {
            "predictions_sha256": sha256_file(source.predictions_path),
            "manifest_sha256": sha256_file(source.manifest_path),
        }
    return hashes


def _method_base(
    *,
    result: E0MethodResult,
    test_inputs: EnergyInputs,
    metadata: TestMetadata,
    sources: Mapping[int, EnergyEvaluationSource],
    shared_inputs_path: Path,
    shared_manifest_path: Path,
) -> dict[str, Any]:
    if result.method not in _METHODS:
        raise E0WorkflowError("unsupported E0 method")
    if test_inputs.structure_ids != metadata.structure_ids:
        raise E0WorkflowError("method test structure IDs differ")
    if not shared_inputs_path.is_file() or not shared_manifest_path.is_file():
        raise E0WorkflowError("shared input artifacts are missing")
    corrected = np.asarray(result.corrected_total, dtype=np.float64)
    if (
        corrected.shape != (len(test_inputs.structure_ids),)
        or not np.isfinite(corrected).all()
    ):
        raise E0WorkflowError("corrected totals are invalid")
    artifact_method = result.artifact.get("method")
    artifact_totals = result.artifact.get("corrected_total")
    if (
        artifact_method != result.method
        or not isinstance(artifact_totals, torch.Tensor)
        or not np.array_equal(
            artifact_totals.detach().cpu().to(torch.float64).numpy(), corrected
        )
    ):
        raise E0WorkflowError("correction artifact content differs")
    return {
        "schema_version": 1,
        "formula_version": E0_FORMULA_VERSION,
        "method": result.method,
        "structure_count": len(test_inputs.structure_ids),
        "correction_id": stable_id(_jsonable_artifact(result.artifact)),
        "shared_sha256": {
            "shared_inputs.pt": sha256_file(shared_inputs_path),
            "manifest.json": sha256_file(shared_manifest_path),
        },
        "source_evaluations": _source_hashes(sources),
    }


def _derived_for_sources(
    *,
    method_root: Path,
    result: E0MethodResult,
    test_inputs: EnergyInputs,
    sources: Mapping[int, EnergyEvaluationSource],
    shared_inputs_path: Path,
    shared_manifest_path: Path,
    correction_path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    predictions_by_order: dict[int, dict[str, Any]] = {}
    metrics_by_key: dict[str, dict[str, Any]] = {}
    for order in range(1, 9):
        source = sources[order]
        identity = dict(source.predictions["identity"])
        derived = derive_energy_predictions(
            source.predictions,
            expected_identity=identity,
            corrected_total=result.corrected_total,
            reference_total=test_inputs.reference_total,
            num_atoms=test_inputs.num_atoms,
            thresholds=source.thresholds,
        )
        metrics = energy_metrics_payload(derived, source.representatives)
        publish_derived_evaluation(
            DerivedEvaluationPaths.from_root(
                method_root / "energy" / f"order{order}"
            ),
            predictions=derived,
            metrics=metrics,
            method=result.method,
            head_key=source.head_key,
            input_hashes={
                "source_predictions.pt": sha256_file(source.predictions_path),
                "source_manifest.json": sha256_file(source.manifest_path),
                "shared_inputs.pt": sha256_file(shared_inputs_path),
                "correction.pt": sha256_file(correction_path),
            },
        )
        predictions_by_order[order] = derived
        metrics_by_key[source.head_key] = metrics["branches"]["energy"]
    return predictions_by_order, metrics_by_key


def _error_statistics(values: np.ndarray) -> dict[str, int | float]:
    bound = np.asarray(values, dtype=np.float64)
    if bound.ndim != 1 or bound.size == 0 or not np.isfinite(bound).all():
        raise E0WorkflowError("energy error statistics input is invalid")
    return {
        "sample_count": int(bound.size),
        "mean_abs_error_per_atom": float(np.mean(bound)),
        "median_abs_error_per_atom": float(np.median(bound)),
        "rmse_abs_error_per_atom": float(np.sqrt(np.mean(np.square(bound)))),
        "max_abs_error_per_atom": float(np.max(bound)),
    }


def _energy_error_summary(
    result: E0MethodResult, test_inputs: EnergyInputs
) -> dict[str, dict[str, int | float]]:
    raw = energy_errors_per_atom(
        reference_total=test_inputs.reference_total,
        prediction_total=test_inputs.raw_total,
        num_atoms=test_inputs.num_atoms,
    )
    corrected = energy_errors_per_atom(
        reference_total=test_inputs.reference_total,
        prediction_total=result.corrected_total,
        num_atoms=test_inputs.num_atoms,
    )
    return {
        "raw": _error_statistics(raw),
        "corrected": _error_statistics(corrected),
    }


def _per_structure_rows(
    *,
    result: E0MethodResult,
    test_inputs: EnergyInputs,
    metadata: TestMetadata,
    predictions_by_order: Mapping[int, Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    corrected_error = energy_errors_per_atom(
        reference_total=test_inputs.reference_total,
        prediction_total=result.corrected_total,
        num_atoms=test_inputs.num_atoms,
    )
    correction = result.artifact["correction"].detach().cpu().numpy()
    fields = [
        "source_index",
        "structure_id",
        "formula",
        "num_atoms",
        "reference_total",
        "raw_total",
        "correction",
        "corrected_total",
        "corrected_error_per_atom",
    ]
    for order in range(1, 9):
        fields.extend(
            [f"energy_order{order}_expected_error", f"energy_order{order}_label"]
        )
    rows: list[dict[str, Any]] = []
    for index, structure_id in enumerate(test_inputs.structure_ids):
        row: dict[str, Any] = {
            "source_index": int(metadata.source_indices[index]),
            "structure_id": structure_id,
            "formula": metadata.formulas[index],
            "num_atoms": int(test_inputs.num_atoms[index]),
            "reference_total": float(test_inputs.reference_total[index]),
            "raw_total": float(test_inputs.raw_total[index]),
            "correction": float(correction[index]),
            "corrected_total": float(result.corrected_total[index]),
            "corrected_error_per_atom": float(corrected_error[index]),
        }
        for order in range(1, 9):
            branch = predictions_by_order[order]["energy"]
            row[f"energy_order{order}_expected_error"] = float(
                branch["expected_errors"][index]
            )
            row[f"energy_order{order}_label"] = int(branch["labels"][index])
        rows.append(row)
    return fields, rows


def _summary_csv(
    errors: Mapping[str, Mapping[str, int | float]],
    metrics: Mapping[str, Mapping[str, int | float]],
) -> bytes:
    rows: list[dict[str, Any]] = []
    for key, values in errors.items():
        for metric, value in values.items():
            rows.append(
                {"section": "energy_error", "key": key, "metric": metric, "value": value}
            )
    for key, values in metrics.items():
        for metric, value in values.items():
            rows.append(
                {"section": "confidence_head", "key": key, "metric": metric, "value": value}
            )
    return _csv_bytes(("section", "key", "metric", "value"), rows)


def publish_e0_method(
    *,
    output_root: Path,
    result: E0MethodResult,
    test_inputs: EnergyInputs,
    metadata: TestMetadata,
    energy_sources: Mapping[int, EnergyEvaluationSource],
    shared_inputs_path: Path,
    shared_manifest_path: Path,
) -> Path:
    """Publish one complete, immutable, release-ready E0 method directory."""
    root = Path(output_root) / result.method
    correction_path = root / "correction.pt"
    summary_json = root / "summary_metrics.json"
    summary_csv = root / "summary_metrics.csv"
    structure_csv = root / "per_structure.csv"
    manifest_path = root / "manifest.json"
    base = _method_base(
        result=result,
        test_inputs=test_inputs,
        metadata=metadata,
        sources=energy_sources,
        shared_inputs_path=shared_inputs_path,
        shared_manifest_path=shared_manifest_path,
    )
    if manifest_path.exists():
        manifest = _read_json(manifest_path, "E0 method manifest")
        if {key: manifest.get(key) for key in base} != base:
            raise E0WorkflowError("E0 method identity differs")
        outputs = manifest.get("output_sha256")
        if type(outputs) is not dict:
            raise E0WorkflowError("E0 method output hashes differ")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if set(outputs) != actual:
            raise E0WorkflowError("E0 method output file set differs")
        for relative, digest in outputs.items():
            if type(digest) is not str or _HEX64.fullmatch(digest) is None:
                raise E0WorkflowError("E0 method output hashes differ")
            if sha256_file(root / relative) != digest:
                raise E0WorkflowError("E0 method output hashes differ")
        return manifest_path
    if root.exists() and any(root.iterdir()):
        raise E0WorkflowError("E0 method evidence is partial")
    root.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(correction_path, result.artifact)
    if not _payload_equal(load_torch_artifact(correction_path), result.artifact):
        raise E0WorkflowError("persisted correction artifact differs")
    predictions, metrics = _derived_for_sources(
        method_root=root,
        result=result,
        test_inputs=test_inputs,
        sources=energy_sources,
        shared_inputs_path=shared_inputs_path,
        shared_manifest_path=shared_manifest_path,
        correction_path=correction_path,
    )
    fields, rows = _per_structure_rows(
        result=result,
        test_inputs=test_inputs,
        metadata=metadata,
        predictions_by_order=predictions,
    )
    _write_bytes(structure_csv, _csv_bytes(fields, rows))
    error_summary = _energy_error_summary(result, test_inputs)
    summary = {**base, "energy_error": error_summary, "orders": metrics}
    atomic_json_dump(summary_json, summary)
    _write_bytes(summary_csv, _summary_csv(error_summary, metrics))
    outputs = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    atomic_json_dump(manifest_path, {**base, "output_sha256": outputs})
    return publish_e0_method(
        output_root=output_root,
        result=result,
        test_inputs=test_inputs,
        metadata=metadata,
        energy_sources=energy_sources,
        shared_inputs_path=shared_inputs_path,
        shared_manifest_path=shared_manifest_path,
    )


def _load_external_cache(
    config: ExternalInferenceConfig,
    *,
    build_missing_inputs: bool,
) -> CacheManifest:
    if build_missing_inputs:
        run_build_external_cache(config)
    cache = load_complete_cache(
        external_cache_root(config),
        expected_cache_id=external_cache_id(config),
    )
    if set(cache.splits) != {config.cache.split}:
        raise E0WorkflowError("external cache split set differs")
    shards = cache.splits[config.cache.split]
    structures = sum(shard.num_structures for shard in shards)
    atoms = sum(shard.num_atoms for shard in shards)
    if (
        structures != config.dataset.expected_structures
        or atoms != config.dataset.expected_atoms
    ):
        raise E0WorkflowError("external cache counts differ from configuration")
    return cache


def _load_external_evaluation(
    *,
    config: ExternalInferenceConfig,
    cache: CacheManifest,
    head_key: str,
    test_inputs: EnergyInputs,
) -> tuple[Any, dict[str, Any], Path, Path]:
    training = resolve_head_config(config, head_key)
    inputs = load_evaluation_inputs(training)
    paths = external_evaluation_paths(config, head_key)
    hashes = evaluation_input_hashes(inputs)
    hashes["cache_manifest.json"] = sha256_file(
        cache.root / "cache_manifest.json"
    )
    if not validate_or_reuse_evaluation(paths, inputs.identity, hashes):
        raise E0WorkflowError(f"test evaluation is missing: {head_key}")
    predictions = validate_prediction_payload(
        load_torch_artifact(paths.predictions),
        expected_identity=inputs.identity,
    )
    expected_branch = "force" if head_key == "force" else "energy"
    if tuple(predictions["enabled_branches"]) != (expected_branch,):
        raise E0WorkflowError(f"test evaluation branch differs: {head_key}")
    if tuple(predictions["structure_ids"]) != test_inputs.structure_ids:
        raise E0WorkflowError(
            f"test evaluation structure IDs differ: {head_key}"
        )
    offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(test_inputs.num_atoms, dtype=np.int64),
        )
    )
    if not torch.equal(
        predictions["structure_offsets"], torch.from_numpy(offsets)
    ):
        raise E0WorkflowError(
            f"test evaluation structure offsets differ: {head_key}"
        )
    return inputs, predictions, paths.predictions, paths.manifest


def _load_test_evaluations(
    *,
    config: ExternalInferenceConfig,
    cache: CacheManifest,
    test_inputs: EnergyInputs,
    build_missing_inputs: bool,
) -> _TestEvaluationSources:
    keys = ("force", *(f"energy_order{order}" for order in range(1, 9)))
    if build_missing_inputs:
        for key in keys:
            run_evaluate_external(config, key)
    force_context, force_predictions, force_path, force_manifest = (
        _load_external_evaluation(
            config=config,
            cache=cache,
            head_key="force",
            test_inputs=test_inputs,
        )
    )
    energy_sources: dict[int, EnergyEvaluationSource] = {}
    for order in range(1, 9):
        head_key = f"energy_order{order}"
        inputs, predictions, predictions_path, manifest_path = (
            _load_external_evaluation(
                config=config,
                cache=cache,
                head_key=head_key,
                test_inputs=test_inputs,
            )
        )
        if set(inputs.binning.branches) != {"energy"}:
            raise E0WorkflowError(
                f"energy binning branches differ: {head_key}"
            )
        binning = inputs.binning.branches["energy"]
        energy_sources[order] = EnergyEvaluationSource(
            head_key=head_key,
            predictions=predictions,
            predictions_path=predictions_path,
            manifest_path=manifest_path,
            thresholds=binning.thresholds.detach().cpu().to(torch.float64),
            representatives=binning.representatives.detach().cpu().to(
                torch.float64
            ),
        )
    return _TestEvaluationSources(
        force_context=force_context,
        force_predictions=force_predictions,
        force_predictions_path=force_path,
        force_manifest_path=force_manifest,
        energy_sources=energy_sources,
    )


def _prepare_e0_sources(config: E0PostprocessConfig) -> _PreparedE0Sources:
    validation_cache = _load_external_cache(
        config.validation,
        build_missing_inputs=config.build_missing_inputs,
    )
    test_cache = _load_external_cache(
        config.test,
        build_missing_inputs=config.build_missing_inputs,
    )
    loaded = load_frozen_backbone(
        config.test.force_config.checkpoint,
        device="cpu",
    )
    supported = tuple(loaded.identity.atomic_numbers)
    raw_validation = collect_energy_inputs(
        validation_cache,
        split=config.validation.cache.split,
        batch_size=config.validation.head_batch_size,
        supported_atomic_numbers=supported,
    )
    test = collect_energy_inputs(
        test_cache,
        split=config.test.cache.split,
        batch_size=config.test.head_batch_size,
        supported_atomic_numbers=supported,
    )
    validation = isolate_validation_inputs(raw_validation, test)
    metadata = load_test_metadata(
        path=config.test.dataset.path,
        expected_sha256=config.test.dataset.expected_sha256,
        supported_atomic_numbers=supported,
        expected_structure_ids=test.structure_ids,
        atomization_energy_key=config.atomization_energy_key,
        source_index_path=config.test.dataset.source_index_path,
    )
    model_e0 = extract_model_e0(
        loaded.model,
        atomic_numbers=supported,
        selected_head=loaded.identity.selected_head,
    )
    evaluations = _load_test_evaluations(
        config=config.test,
        cache=test_cache,
        test_inputs=test,
        build_missing_inputs=config.build_missing_inputs,
    )
    return _PreparedE0Sources(
        validation=validation.inputs,
        test=test,
        metadata=metadata,
        supported_atomic_numbers=supported,
        model_e0=model_e0,
        input_files={
            "e0_config.yaml": config.source_path,
            "validation_config.yaml": config.validation.source_path,
            "test_config.yaml": config.test.source_path,
            "validation_cache_manifest.json": (
                validation_cache.root / "cache_manifest.json"
            ),
            "test_cache_manifest.json": (
                test_cache.root / "cache_manifest.json"
            ),
            "checkpoint.model": config.test.force_config.checkpoint.path,
        },
        evaluations=evaluations,
        excluded_validation_structure_ids=validation.excluded_structure_ids,
        excluded_validation_structure_indices=(
            validation.excluded_structure_indices
        ),
    )


def _derived_density_sources(
    *,
    output_root: Path,
    method: str,
    energy_sources: Mapping[int, EnergyEvaluationSource],
) -> dict[str, tuple[EnergyEvaluationSource, dict[str, Any], Path]]:
    loaded: dict[
        str, tuple[EnergyEvaluationSource, dict[str, Any], Path]
    ] = {}
    for order in range(1, 9):
        source = energy_sources[order]
        paths = DerivedEvaluationPaths.from_root(
            Path(output_root) / method / "energy" / f"order{order}"
        )
        if not all(path.is_file() for path in paths.outputs):
            raise E0WorkflowError(
                f"derived energy evaluation is missing: {method}/order{order}"
            )
        identity = dict(source.predictions["identity"])
        predictions = validate_prediction_payload(
            load_torch_artifact(paths.predictions),
            expected_identity=identity,
        )
        loaded[f"energy_order{order}"] = (
            source,
            predictions,
            paths.manifest,
        )
    return loaded


def run_e0_postprocess(
    config: E0PostprocessConfig,
    *,
    plot: bool = False,
) -> E0PostprocessOutputs:
    """Reuse committed inference to publish both E0 methods and optional plots."""
    if not isinstance(config, E0PostprocessConfig):
        raise TypeError("config must be an E0PostprocessConfig")
    if type(plot) is not bool:
        raise TypeError("plot must be a boolean")
    if tuple(config.methods) != ("e0_replace", "e0_reestimate"):
        raise E0WorkflowError("E0 method order differs")
    prepared = _prepare_e0_sources(config)
    shared = publish_shared_inputs(
        output_root=config.output_root,
        validation=prepared.validation,
        test=prepared.test,
        metadata=prepared.metadata,
        supported_atomic_numbers=prepared.supported_atomic_numbers,
        model_e0=prepared.model_e0,
        input_files=prepared.input_files,
        force_predictions_path=prepared.evaluations.force_predictions_path,
        force_manifest_path=prepared.evaluations.force_manifest_path,
    )
    corrections = compute_e0_corrections(
        validation=prepared.validation,
        test=prepared.test,
        metadata=prepared.metadata,
        model_e0=prepared.model_e0,
        supported_atomic_numbers=prepared.supported_atomic_numbers,
        excluded_validation_structure_ids=(
            prepared.excluded_validation_structure_ids
        ),
        excluded_validation_structure_indices=(
            prepared.excluded_validation_structure_indices
        ),
    )
    if tuple(corrections) != tuple(config.methods):
        raise E0WorkflowError("computed E0 method order differs")
    method_manifests: dict[str, Path] = {}
    for method in config.methods:
        method_manifests[method] = publish_e0_method(
            output_root=config.output_root,
            result=corrections[method],
            test_inputs=prepared.test,
            metadata=prepared.metadata,
            energy_sources=prepared.evaluations.energy_sources,
            shared_inputs_path=shared.inputs,
            shared_manifest_path=shared.manifest,
        )
    plot_manifests: dict[str, Path] = {}
    if plot:
        from .plot_density_suite import (
            publish_density_suite_from_predictions,
        )

        plot_manifests["force"] = publish_density_suite_from_predictions(
            dataset="force",
            plot_root=config.plot_root,
            loaded={
                "force": (
                    prepared.evaluations.force_context,
                    prepared.evaluations.force_predictions,
                    prepared.evaluations.force_manifest_path,
                )
            },
            repo_root=REPOSITORY_ROOT,
        )
        for method in config.methods:
            plot_manifests[method] = (
                publish_density_suite_from_predictions(
                    dataset=method,
                    plot_root=config.plot_root,
                    loaded=_derived_density_sources(
                        output_root=config.output_root,
                        method=method,
                        energy_sources=prepared.evaluations.energy_sources,
                    ),
                    repo_root=REPOSITORY_ROOT,
                )
            )
    return E0PostprocessOutputs(
        shared_manifest=shared.manifest,
        method_manifests=method_manifests,
        plot_manifests=plot_manifests,
    )


__all__ = [
    "DerivedEvaluationPaths",
    "E0MethodResult",
    "E0PostprocessOutputs",
    "E0WorkflowError",
    "EnergyEvaluationSource",
    "EnergyInputs",
    "ValidationIsolation",
    "SharedArtifactPaths",
    "TestMetadata",
    "collect_energy_inputs",
    "compute_e0_corrections",
    "derive_energy_predictions",
    "energy_metrics_payload",
    "extract_model_e0",
    "isolate_validation_inputs",
    "load_test_metadata",
    "publish_e0_method",
    "publish_derived_evaluation",
    "publish_shared_inputs",
    "run_e0_postprocess",
]
