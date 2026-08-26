"""Publication workflow for LLPR energy-reference postprocessing."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml
from ase.data import chemical_symbols
from ase.io import iread

from .artifacts import atomic_json_dump, atomic_torch_save, load_torch_artifact, sha256_file
from .calibration import CholeskyQuadraticForm
from .checkpoint import LoadedCheckpoint, load_checkpoint
from .config import LLPRConfig, load_config
from .curvature_source import load_curvature_source
from .data import build_dataset, iter_samples
from .energy_reference import (
    apply_direct_test_atomic_baseline,
    apply_model_aware_correction,
    calibrate_energy_alpha,
    fit_model_aware_reestimation,
)
from .inference import ENERGY_FIELDS, summarize_variant, validate_q
from .observables import compute_energy_jacobian
from .readout import discover_readout_layout
from .validation import validate_publication_root

_VARIANTS = ("he", "hf", "hef")
_METHODS = ("direct_test_atomic_baseline", "model_aware_val_fit")
_FILES = ("energy.csv", "force_components.csv", "force_structure.csv", "summary.json")
_FORCE_FILES = ("force_components.csv", "force_structure.csv")


@dataclass(frozen=True)
class EnergyReferenceWorkflowConfig:
    llpr_config: Path
    source_publication_root: Path
    output: Path
    methods: tuple[str, ...]


@dataclass(frozen=True)
class ModelContext:
    loaded: LoadedCheckpoint
    model_e0: np.ndarray


@dataclass(frozen=True)
class DatasetMetadata:
    structure_ids: tuple[str, ...]
    num_atoms: np.ndarray
    composition: np.ndarray
    reference_total: np.ndarray
    atomization_total: np.ndarray | None


@dataclass(frozen=True)
class ValidationEnergyData:
    metadata: DatasetMetadata
    prediction_total: np.ndarray
    q_by_variant: Mapping[str, np.ndarray]


ValidationCollector = Callable[[LLPRConfig, ModelContext, Mapping[str, Any]], ValidationEnergyData]


def _path(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    result = Path(value)
    return (base / result).resolve() if not result.is_absolute() else result.resolve()


def load_energy_reference_config(path: Path) -> EnergyReferenceWorkflowConfig:
    source = Path(path).resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"could not read energy-reference config: {source}") from error
    fields = {"llpr_config", "source_publication_root", "output", "methods"}
    if not isinstance(document, Mapping) or set(document) != fields:
        raise ValueError("energy-reference config must contain exactly llpr_config, source_publication_root, output, and methods")
    if not isinstance(document["methods"], list) or tuple(document["methods"]) != _METHODS:
        raise ValueError(f"methods must equal {list(_METHODS)}")
    base = source.parent
    result = EnergyReferenceWorkflowConfig(
        llpr_config=_path(base, document["llpr_config"], "llpr_config"),
        source_publication_root=_path(base, document["source_publication_root"], "source_publication_root"),
        output=_path(base, document["output"], "output"),
        methods=tuple(document["methods"]),
    )
    if result.output.is_relative_to(result.source_publication_root):
        raise ValueError("output must be outside source_publication_root")
    return result


# Backward-compatible name used by the command-line entry point and older
# automation wrappers. Both names resolve to the same strict config parser.
load_e0_config = load_energy_reference_config


def _rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(ENERGY_FIELDS):
                raise ValueError(f"CSV schema mismatch for {path}")
            result = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"could not read {path}") from error
    if not result or any(None in row for row in result):
        raise ValueError(f"CSV rows are invalid: {path}")
    return result


def _float(rows: Sequence[Mapping[str, str]], field: str, source: str) -> np.ndarray:
    try:
        result = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source} {field} is invalid") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{source} {field} is non-finite")
    return result


def _ids(atoms: Any, index: int) -> str:
    for key in ("structure_id", "config_id", "id"):
        if atoms.info.get(key) is not None and str(atoms.info[key]):
            return str(atoms.info[key])
    return str(index)


def _energy(atoms: Any, index: int) -> float:
    results = getattr(getattr(atoms, "calc", None), "results", {})
    for value in (atoms.info.get("REF_energy"), atoms.info.get("energy"), results.get("energy")):
        if value is not None:
            array = np.asarray(value, dtype=np.float64)
            if array.shape == () and np.isfinite(array).all():
                return float(array)
    raise ValueError(f"dataset structure {index} is missing energy")


def _atomization(atoms: Any, index: int) -> float:
    for key in ("atomization_energy", "REF_atomization_energy"):
        if key in atoms.info:
            array = np.asarray(atoms.info[key], dtype=np.float64)
            if array.shape == () and np.isfinite(array).all():
                return float(array)
    raise ValueError(f"dataset structure {index} is missing atomization_energy")


def _metadata(path: Path, atomic_numbers: Sequence[int], rows: int, atomization: bool) -> DatasetMetadata:
    table = {int(number): column for column, number in enumerate(atomic_numbers)}
    ids: list[str] = []
    counts: list[int] = []
    composition: list[np.ndarray] = []
    reference: list[float] = []
    atom_values: list[float] = []
    try:
        for index, atoms in enumerate(iread(path, index=":")):
            if index >= rows:
                break
            count = np.zeros(len(table), dtype=np.float64)
            for number in atoms.numbers:
                if int(number) not in table:
                    raise ValueError(f"dataset structure {index} contains unsupported element {number}")
                count[table[int(number)]] += 1.0
            ids.append(_ids(atoms, index))
            counts.append(len(atoms))
            composition.append(count)
            reference.append(_energy(atoms, index))
            if atomization:
                atom_values.append(_atomization(atoms, index))
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"could not stream dataset metadata: {path}") from error
    if len(ids) != rows or len(set(ids)) != rows:
        raise ValueError("dataset/publication structure count or ID mismatch")
    return DatasetMetadata(
        tuple(ids),
        np.asarray(counts, dtype=np.int64),
        np.stack(composition),
        np.asarray(reference, dtype=np.float64),
        np.asarray(atom_values, dtype=np.float64) if atomization else None,
    )


def _model_e0(loaded: LoadedCheckpoint) -> np.ndarray:
    heads = tuple(str(value) for value in loaded.model.heads)
    values = getattr(getattr(loaded.model, "atomic_energies_fn", None), "atomic_energies", None)
    if not isinstance(values, torch.Tensor) or loaded.identity.selected_head not in heads:
        raise ValueError("checkpoint atomic energies are missing")
    matrix = torch.atleast_2d(values.detach().cpu()).to(torch.float64)
    expected = (len(heads), len(loaded.identity.atomic_numbers))
    if tuple(matrix.shape) != expected or not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"checkpoint atomic energies must have finite shape {expected}")
    return np.asarray(matrix[heads.index(loaded.identity.selected_head)].numpy())


def _load_context(config: LLPRConfig) -> ModelContext:
    loaded = load_checkpoint(
        config.checkpoint,
        torch.device("cpu"),
        selected_head=config.selected_head,
        expected_readout_size=config.expected_readout_size,
    )
    return ModelContext(loaded, _model_e0(loaded))


def _records(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        value = identity["calibration"]["records"]
    except (KeyError, TypeError) as error:
        raise ValueError("raw calibration records are missing") from error
    if not isinstance(value, Mapping) or set(value) != set(_VARIANTS):
        raise ValueError("raw calibration records are invalid")
    return value


def _collect_validation(config: LLPRConfig, context: ModelContext, raw_identity: Mapping[str, Any]) -> ValidationEnergyData:
    loaded = context.loaded
    layout = discover_readout_layout(loaded.model, expected_size=config.expected_readout_size)
    curvature = load_curvature_source(config, loaded.identity, layout)
    records = _records(raw_identity)
    solvers = {
        variant: CholeskyQuadraticForm(curvature.variants[variant], float(records[variant]["energy"]["ridge"]))
        for variant in _VARIANTS
    }
    dataset = build_dataset(
        config.calibration.path,
        config.calibration.expected_sha256,
        loaded.identity.atomic_numbers,
        loaded.identity.r_max,
        loaded.identity.selected_head,
    )
    limit = config.runtime.effective_consumer_max_structures
    count = dataset.size if limit is None else min(dataset.size, limit)
    metadata = _metadata(config.calibration.path, loaded.identity.atomic_numbers, count, False)
    predictions: list[float] = []
    q = {variant: [] for variant in _VARIANTS}
    for index, sample in enumerate(iter_samples(dataset, torch.device("cpu"), loaded.identity.dtype, max_structures=limit)):
        if (sample.structure_id, sample.num_atoms) != (metadata.structure_ids[index], int(metadata.num_atoms[index])):
            raise ValueError("validation sample order differs from metadata")
        jacobian = compute_energy_jacobian(loaded.model, sample.batch, layout)
        predictions.append(jacobian.energy_total)
        for variant in _VARIANTS:
            value = validate_q(solvers[variant].q(jacobian.g_energy.reshape(1, -1)), structure_index=index, target="energy")
            q[variant].append(float(value[0]))
    if len(predictions) != count:
        raise ValueError("validation energy count mismatch")
    return ValidationEnergyData(metadata, np.asarray(predictions), {key: np.asarray(value) for key, value in q.items()})


def _copy(source: Path, destination: Path) -> str:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"publication input is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    digest = sha256_file(source)
    if sha256_file(destination) != digest:
        raise ValueError(f"publication copy SHA mismatch: {source}")
    return digest


def _paths() -> tuple[str, ...]:
    return ("progress.pt",) + tuple(f"{variant}/{name}" for variant in _VARIANTS for name in _FILES)


def _snapshot(source: Path, destination: Path) -> dict[str, str]:
    hashes = {name: _copy(source / name, destination / name) for name in _paths()}
    if (source / "manifest.json").is_file():
        _copy(source / "manifest.json", destination / "manifest.json")
    validate_publication_root(destination)
    if hashes != {name: sha256_file(source / name) for name in _paths()}:
        raise ValueError("raw publication changed while being snapshotted")
    return hashes


def _q_digest(rows: Sequence[Mapping[str, str]]) -> str:
    values = [[row["structure_id"], int(row["num_atoms"]), float(row["q"])] for row in rows]
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _raw_rows(root: Path) -> dict[str, list[dict[str, str]]]:
    result = {variant: _rows(root / variant / "energy.csv") for variant in _VARIANTS}
    baseline = result["he"]
    keys = [(row["structure_id"], int(row["num_atoms"])) for row in baseline]
    refs = _float(baseline, "reference", "he/energy.csv")
    if len(set(keys)) != len(keys) or any(number <= 0 for _, number in keys):
        raise ValueError("raw energy keys are invalid")
    for variant in _VARIANTS[1:]:
        rows = result[variant]
        if [(row["structure_id"], int(row["num_atoms"])) for row in rows] != keys or not np.allclose(_float(rows, "reference", variant), refs, rtol=1e-12, atol=1e-12):
            raise ValueError("raw energy variants are not aligned")
    return result


def _align(metadata: DatasetMetadata, rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    first = rows["he"]
    if metadata.structure_ids != tuple(row["structure_id"] for row in first) or not np.array_equal(metadata.num_atoms, np.asarray([int(row["num_atoms"]) for row in first])):
        raise ValueError("test structure_id/num_atoms differs from raw publication")
    expected = metadata.reference_total / metadata.num_atoms
    for variant in _VARIANTS:
        if not np.allclose(_float(rows[variant], "reference", variant), expected, rtol=1e-11, atol=1e-10):
            raise ValueError("test reference differs from raw publication")


def _write_energy(path: Path, rows: Sequence[Mapping[str, str]], totals: np.ndarray, alpha: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENERGY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row, total in zip(rows, totals):
            prediction = float(total) / int(row["num_atoms"])
            residual = float(row["reference"]) - prediction
            variance = float(alpha) ** 2 * float(row["q"])
            values = (prediction, residual, variance, math.sqrt(variance))
            if not all(math.isfinite(value) for value in values):
                raise ValueError("derived energy contains non-finite values")
            derived = dict(row)
            derived.update(dict(zip(("prediction", "residual", "variance", "std"), values)))
            writer.writerow(derived)


def _effective(raw: Mapping[str, Any], alpha: Mapping[str, float], rows: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in _VARIANTS:
        result[variant] = {}
        for target in ("energy", "forces"):
            record = {field: raw[variant][target][field] for field in ("ridge_mode", "ridge", "alpha", "rows")}
            if target == "energy":
                record.update(alpha=float(alpha[variant]), rows=int(rows))
            result[variant][target] = record
    return result


def _provenance(
    method: str,
    raw_identity: Mapping[str, Any],
    hashes: Mapping[str, str],
    raw_rows: Mapping[str, Sequence[Mapping[str, str]]],
    destination: Path,
    context: ModelContext,
    effective: Mapping[str, Any],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    raw_q = {variant: _q_digest(raw_rows[variant]) for variant in _VARIANTS}
    derived_q = {variant: _q_digest(_rows(destination / variant / "energy.csv")) for variant in _VARIANTS}
    force_keys = {f"{variant}/{name}" for variant in _VARIANTS for name in _FORCE_FILES}
    identity = context.loaded.identity
    result: dict[str, Any] = {
        "method": method,
        "method_version": "1",
        "raw_publication": {
            "progress_sha256": hashes["progress.pt"],
            "files_sha256": {key: value for key, value in hashes.items() if key != "progress.pt"},
        },
        "checkpoint_sha256": identity.sha256,
        "test_dataset_sha256": raw_identity["test"]["sha256"],
        "atomic_numbers": list(identity.atomic_numbers),
        "chemical_symbols": [chemical_symbols[number] for number in identity.atomic_numbers],
        "model_e0": context.model_e0.tolist(),
        "energy_q": {"unchanged_from_raw": True, "raw_sha256": raw_q, "derived_sha256": derived_q},
        "force_files": {
            "unchanged_from_raw": True,
            "raw_sha256": {key: hashes[key] for key in force_keys},
            "derived_sha256": {key: sha256_file(destination / key) for key in force_keys},
        },
        "effective_calibration_records": effective,
        "method_details": dict(details),
    }
    if method == "direct_test_atomic_baseline":
        result["label_usage"] = {
            "evaluation_protocol": "test_informed_oracle",
            "alpha_calibration_split": "test",
            "label_fields": ["energy", "atomization_energy"],
        }
    else:
        result["validation_dataset_sha256"] = raw_identity["calibration"]["identity"]["calibration"]["sha256"]
        result["label_usage"] = {
            "fit_split": "validation",
            "alpha_calibration_split": "validation",
            "label_fields": ["energy"],
            "test_labels_used": False,
            "atomization_energy_used": False,
        }
    return result


def _derive(
    method: str,
    destination: Path,
    raw_root: Path,
    progress: Mapping[str, Any],
    hashes: Mapping[str, str],
    raw_rows: Mapping[str, Sequence[Mapping[str, str]]],
    corrected: Mapping[str, np.ndarray],
    alpha: Mapping[str, float],
    alpha_rows: int,
    context: ModelContext,
    details: Mapping[str, Any],
) -> None:
    raw_identity = progress["identity"]
    effective = _effective(_records(raw_identity), alpha, alpha_rows)
    for variant in _VARIANTS:
        target = destination / variant
        target.mkdir(parents=True)
        for name in _FORCE_FILES:
            _copy(raw_root / variant / name, target / name)
        _write_energy(target / "energy.csv", raw_rows[variant], corrected[variant], alpha[variant])
        raw_summary = json.loads((raw_root / variant / "summary.json").read_text(encoding="utf-8"))
        record = effective[variant]
        summary = summarize_variant(
            target,
            variant=variant,
            ridge_mode=record["energy"]["ridge_mode"],
            ridge=float(record["energy"]["ridge"]),
            energy_alpha=float(record["energy"]["alpha"]),
            force_alpha=float(record["forces"]["alpha"]),
            cholesky_diagnostics=raw_summary["cholesky_diagnostics"],
        )
        atomic_json_dump(target / "summary.json", summary)
    provenance = _provenance(method, raw_identity, hashes, raw_rows, destination, context, effective, details)
    derived_progress = dict(progress)
    derived_progress["identity"] = {**raw_identity, "energy_reference": provenance}
    derived_progress["csv_offsets"] = {
        f"{variant}/{name}": (destination / variant / name).stat().st_size
        for variant in _VARIANTS
        for name in ("energy.csv", *_FORCE_FILES)
    }
    atomic_torch_save(destination / "progress.pt", derived_progress)
    validate_publication_root(destination)


def _validate_context(context: ModelContext, raw_identity: Mapping[str, Any]) -> None:
    checkpoint = raw_identity.get("checkpoint")
    identity = context.loaded.identity
    if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != identity.sha256 or checkpoint.get("atomic_numbers") != list(identity.atomic_numbers) or checkpoint.get("selected_head") != identity.selected_head:
        raise ValueError("checkpoint and raw publication identities differ")


def run_energy_reference_workflow(
    config: EnergyReferenceWorkflowConfig | Path,
    *,
    validation_collector: ValidationCollector | None = None,
    model_context_loader: Callable[[LLPRConfig], ModelContext] | None = None,
) -> Mapping[str, Path]:
    workflow = load_energy_reference_config(config) if isinstance(config, (str, Path)) else config
    if workflow.methods != _METHODS:
        raise ValueError(f"methods must equal {list(_METHODS)}")
    source, output = workflow.source_publication_root.resolve(), workflow.output.resolve()
    if not source.is_dir() or output.exists():
        raise ValueError("source must exist and output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    llpr = load_config(workflow.llpr_config)
    context = (model_context_loader or _load_context)(llpr)
    collector = validation_collector or _collect_validation
    with TemporaryDirectory(prefix=f".{source.name}.e0-raw-", dir=source.parent) as raw_directory:
        raw_root = Path(raw_directory)
        hashes = _snapshot(source, raw_root)
        progress = load_torch_artifact(raw_root / "progress.pt")
        if not isinstance(progress, Mapping) or progress.get("status") != "complete" or not isinstance(progress.get("identity"), Mapping):
            raise ValueError("raw progress must be complete")
        raw_identity = progress["identity"]
        _validate_context(context, raw_identity)
        if sha256_file(llpr.test.path) != raw_identity["test"]["sha256"]:
            raise ValueError("test dataset SHA differs from raw publication")
        validation_sha = raw_identity["calibration"]["identity"]["calibration"][
            "sha256"
        ]
        if sha256_file(llpr.calibration.path) != validation_sha:
            raise ValueError("validation dataset SHA differs from raw publication")
        rows = _raw_rows(raw_root)
        test = _metadata(llpr.test.path, context.loaded.identity.atomic_numbers, len(rows["he"]), True)
        _align(test, rows)
        assert test.atomization_total is not None

        direct: dict[str, np.ndarray] = {}
        direct_alpha: dict[str, float] = {}
        for variant in _VARIANTS:
            raw_total = _float(rows[variant], "prediction", variant) * test.num_atoms
            direct[variant] = apply_direct_test_atomic_baseline(
                raw_total=raw_total,
                reference_total=test.reference_total,
                atomization_total=test.atomization_total,
                composition=test.composition,
                model_e0=context.model_e0,
            )
            direct_alpha[variant] = calibrate_energy_alpha(
                reference_total=test.reference_total,
                prediction_total=direct[variant],
                num_atoms=test.num_atoms,
                q=_float(rows[variant], "q", variant),
                min_q=llpr.curvature.min_q,
            )

        validation = collector(llpr, context, raw_identity)
        fit = fit_model_aware_reestimation(
            composition=validation.metadata.composition,
            reference_total=validation.metadata.reference_total,
            raw_total=validation.prediction_total,
            model_e0=context.model_e0,
        )
        model: dict[str, np.ndarray] = {}
        model_alpha: dict[str, float] = {}
        for variant in _VARIANTS:
            model_alpha[variant] = calibrate_energy_alpha(
                reference_total=validation.metadata.reference_total,
                prediction_total=fit.corrected_validation_total,
                num_atoms=validation.metadata.num_atoms,
                q=validation.q_by_variant[variant],
                min_q=llpr.curvature.min_q,
            )
            model[variant] = apply_model_aware_correction(
                raw_total=_float(rows[variant], "prediction", variant) * test.num_atoms,
                composition=test.composition,
                delta_e0=fit.delta_e0,
            )

        direct_details = {
            "baseline": "per_structure_mad_atomic_baseline",
            "formula": "raw_total - composition @ model_e0 + (reference_total - atomization_total)",
            "structure_count": len(test.structure_ids),
            "composition_shape": list(test.composition.shape),
        }
        model_details = {
            "solver": "numpy.linalg.lstsq",
            "objective": "total_energy_least_squares",
            "application_sign": "positive",
            "rcond": None,
            "matrix_shape": list(validation.metadata.composition.shape),
            "rank": fit.rank,
            "singular_values": fit.singular_values.tolist(),
            "residual_norm": fit.residual_norm,
            "delta_e0": fit.delta_e0.tolist(),
            "new_e0": fit.new_e0.tolist(),
        }
        with TemporaryDirectory(prefix=f".{output.name}.e0-staging-", dir=output.parent) as directory:
            staging = Path(directory) / "publication"
            staging.mkdir()
            _derive("direct_test_atomic_baseline", staging / "direct_test_atomic_baseline", raw_root, progress, hashes, rows, direct, direct_alpha, len(test.structure_ids), context, direct_details)
            _derive("model_aware_val_fit", staging / "model_aware_val_fit", raw_root, progress, hashes, rows, model, model_alpha, len(validation.prediction_total), context, model_details)
            if hashes != {name: sha256_file(source / name) for name in _paths()}:
                raise ValueError("raw publication changed during derivation")
            os.replace(staging, output)
    return {method: output / method for method in _METHODS}
