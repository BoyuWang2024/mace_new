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
from ..binning import labels_from_thresholds
from ..cache import CacheManifest, iter_cache_batches
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
    return supported


def compute_e0_corrections(
    *,
    validation: EnergyInputs,
    test: EnergyInputs,
    metadata: TestMetadata,
    model_e0: Sequence[float] | np.ndarray,
    supported_atomic_numbers: Sequence[int],
) -> dict[str, E0MethodResult]:
    """Compute both corrections while fitting reestimate on validation only."""
    supported = _aligned_inputs(validation, test, metadata, supported_atomic_numbers)
    checkpoint_e0 = np.asarray(model_e0, dtype=np.float64)
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


__all__ = [
    "DerivedEvaluationPaths",
    "E0MethodResult",
    "E0WorkflowError",
    "EnergyEvaluationSource",
    "EnergyInputs",
    "SharedArtifactPaths",
    "TestMetadata",
    "collect_energy_inputs",
    "compute_e0_corrections",
    "derive_energy_predictions",
    "energy_metrics_payload",
    "extract_model_e0",
    "load_test_metadata",
    "publish_e0_method",
    "publish_derived_evaluation",
    "publish_shared_inputs",
]
