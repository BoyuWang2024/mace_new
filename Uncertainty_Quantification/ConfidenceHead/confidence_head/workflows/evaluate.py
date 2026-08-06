"""Evaluate committed ConfidenceHead best checkpoints on cache-v2 test data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..artifacts import load_torch_artifact
from ..binning import BinningArtifact, labels_from_thresholds
from ..cache import CacheManifest, iter_cache_batches, load_complete_cache
from ..checkpoint import (
    BEST_CHECKPOINT_KEYS,
    _validate_saved_model,
    _validate_versions,
)
from ..config import ConfidenceHeadConfig
from ..evaluation_artifacts import (
    EVALUATION_FORMULA_VERSION,
    EVALUATION_INPUT_FILES,
    EVALUATION_SCHEMA_VERSION,
    EvaluationConflictError,
    EvaluationPaths,
    commit_evaluation,
    validate_or_reuse_evaluation,
)
from ..identity import sha256_file
from ..metrics import branch_metrics, expected_errors
from ..model import MultiBranchConfidenceModel
from ..run_naming import make_run_tag
from .train import (
    TRAINING_FORMULA_VERSION,
    TRAINING_SCHEMA_VERSION,
    _ROOT_ENTRIES,
    _feature_dtype,
)


_IDENTITY_KEYS = {"run_id", "experiment_id", "cache_id", "binning_id"}
_RUN_IDENTITY_KEYS = {
    "schema_version",
    "formula_version",
    *_IDENTITY_KEYS,
    "run_tag",
    "enabled_branches",
}
_TRAINING_SUPPORT_PATHS = {
    "source.yaml": ("config", "source.yaml"),
    "resolved.json": ("config", "resolved.json"),
    "run_identity.json": ("identity", "run_identity.json"),
    "environment.json": ("identity", "environment.json"),
    "git.json": ("identity", "git.json"),
    "binning.pt": ("binning", "binning.pt"),
    "binning_manifest.json": ("binning", "binning_manifest.json"),
    "events.jsonl": ("run", "events.jsonl"),
    "best.pt": ("run", "best.pt"),
    "last.pt": ("run", "last.pt"),
    "training_summary.json": ("run", "training_summary.json"),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX12 = re.compile(r"^[0-9a-f]{12}$")


class EvaluationInputError(RuntimeError):
    """A completed training run cannot be safely used for evaluation."""


@dataclass(frozen=True)
class EvaluationInputs:
    """Snapshot-bound inputs for one completed training run."""

    config: ConfidenceHeadConfig
    cache: CacheManifest
    binning: BinningArtifact
    run_root: Path
    run_dir: Path
    identity: dict[str, str]


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise EvaluationInputError(f"{where} is unreadable: {error}") from error
    if type(value) is not dict:
        raise EvaluationInputError(f"{where} must contain a JSON object")
    return value


def _sha(value: object, *, where: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise EvaluationInputError(f"{where} must be a SHA-256 value")
    return value


def _identity(value: object, *, where: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != _IDENTITY_KEYS:
        raise EvaluationInputError(f"{where} keys differ")
    return {name: _sha(value[name], where=f"{where}.{name}") for name in sorted(value)}


def _enabled(config: ConfidenceHeadConfig) -> tuple[str, ...]:
    return tuple(
        branch
        for branch, enabled in (
            ("force", config.force_enabled),
            ("energy", config.energy_enabled),
        )
        if enabled
    )


def _discover_run_root(config: ConfidenceHeadConfig) -> Path:
    runs_root = config.run.output_root / config.run.name_prefix / "runs"
    prefix = f"{make_run_tag(config)}-"
    if not runs_root.is_dir():
        raise EvaluationInputError(f"training runs root is missing: {runs_root}")
    matches = sorted(
        path
        for path in runs_root.iterdir()
        if path.is_dir()
        and path.name.startswith(prefix)
        and _HEX12.fullmatch(path.name[len(prefix):]) is not None
    )
    if len(matches) != 1:
        raise EvaluationInputError(
            f"expected exactly one completed run matching {prefix}*, found {len(matches)}"
        )
    return matches[0]


def _load_snapshot_identity(
    config: ConfidenceHeadConfig, run_root: Path
) -> dict[str, str]:
    unknown = {path.name for path in run_root.iterdir()} - _ROOT_ENTRIES
    if unknown:
        raise EvaluationInputError(
            f"run root contains unknown entries: {sorted(unknown)}"
        )
    source = run_root / "config" / "source.yaml"
    if not source.is_file() or source.read_bytes() != config.source_path.read_bytes():
        raise EvaluationInputError("source config differs from immutable training snapshot")
    payload = _read_json(
        run_root / "identity" / "run_identity.json",
        where="run identity snapshot",
    )
    if set(payload) != _RUN_IDENTITY_KEYS:
        raise EvaluationInputError("run identity snapshot keys differ")
    if payload["schema_version"] != TRAINING_SCHEMA_VERSION:
        raise EvaluationInputError("run identity schema_version differs")
    if payload["formula_version"] != TRAINING_FORMULA_VERSION:
        raise EvaluationInputError("run identity formula_version differs")
    if payload["run_tag"] != make_run_tag(config):
        raise EvaluationInputError("run identity tag differs from config")
    if payload["enabled_branches"] != list(_enabled(config)):
        raise EvaluationInputError("run identity enabled branches differ")
    return _identity(
        {name: payload[name] for name in _IDENTITY_KEYS},
        where="run identity",
    )


def _validate_training_completion(
    run_root: Path, identity: Mapping[str, str]
) -> None:
    run_dir = run_root / "run"
    validation_path = run_dir / "training_validation.json"
    manifest_path = run_dir / "training_manifest.json"
    if validation_path.exists() != manifest_path.exists():
        raise EvaluationInputError("training validation and manifest are partial")
    if not validation_path.is_file():
        raise EvaluationInputError("training validation is missing")
    validation = _read_json(validation_path, where="training validation")
    manifest = _read_json(manifest_path, where="training manifest")
    for where, payload in (
        ("training validation", validation),
        ("training manifest", manifest),
    ):
        if payload.get("schema_version") != TRAINING_SCHEMA_VERSION:
            raise EvaluationInputError(f"{where} schema_version differs")
        if payload.get("formula_version") != TRAINING_FORMULA_VERSION:
            raise EvaluationInputError(f"{where} formula_version differs")
        if any(payload.get(name) != identity[name] for name in _IDENTITY_KEYS):
            raise EvaluationInputError(f"{where} identity differs")
    if validation.get("valid") is not True:
        raise EvaluationInputError("training validation is not valid")
    if manifest.get("validation_filename") != "training_validation.json":
        raise EvaluationInputError("training manifest validation filename differs")
    validation_hash = sha256_file(validation_path)
    if manifest.get("validation_sha256") != validation_hash:
        raise EvaluationInputError("training validation sha256 differs")

    validation_hashes = validation.get("file_sha256")
    manifest_hashes = manifest.get("file_sha256")
    expected_support_names = set(_TRAINING_SUPPORT_PATHS)
    if type(validation_hashes) is not dict or set(validation_hashes) != expected_support_names:
        raise EvaluationInputError("training validation support hashes differ")
    expected_manifest_names = expected_support_names | {"training_validation.json"}
    if type(manifest_hashes) is not dict or set(manifest_hashes) != expected_manifest_names:
        raise EvaluationInputError("training manifest support hashes differ")
    expected_manifest_hashes = dict(validation_hashes)
    expected_manifest_hashes["training_validation.json"] = validation_hash
    if manifest_hashes != expected_manifest_hashes:
        raise EvaluationInputError("training manifest and validation hashes differ")
    for name, parts in _TRAINING_SUPPORT_PATHS.items():
        path = run_root.joinpath(*parts)
        if not path.is_file():
            raise EvaluationInputError(f"training support artifact is missing: {name}")
        if _sha(validation_hashes[name], where=f"training hash {name}") != sha256_file(path):
            raise EvaluationInputError(f"training support artifact sha256 differs: {name}")


def load_evaluation_inputs(config: ConfidenceHeadConfig) -> EvaluationInputs:
    """Resolve a completed run by its immutable snapshot, not current Git HEAD."""
    if not isinstance(config, ConfidenceHeadConfig):
        raise TypeError("config must be a ConfidenceHeadConfig")
    if config.model.force.target_mode != "atom_mean" and config.force_enabled:
        raise EvaluationInputError("publication evaluation requires force atom_mean")
    run_root = _discover_run_root(config)
    identity = _load_snapshot_identity(config, run_root)
    _validate_training_completion(run_root, identity)

    cache_root = (
        config.run.output_root
        / config.run.name_prefix
        / "cache"
        / identity["cache_id"]
    )
    cache = load_complete_cache(
        cache_root,
        expected_cache_id=identity["cache_id"],
        expected_allow_cross_split_duplicates=config.profile == "smoke_test",
    )
    target_mode = config.model.force.target_mode if config.force_enabled else None
    try:
        binning = BinningArtifact.from_payload(
            load_torch_artifact(run_root / "binning" / "binning.pt"),
            force_target_mode=target_mode,
        )
    except Exception as error:
        raise EvaluationInputError(f"binning artifact is invalid: {error}") from error
    if (
        binning.cache_id != identity["cache_id"]
        or binning.experiment_id != identity["experiment_id"]
        or binning.binning_id != identity["binning_id"]
        or binning.run_id != identity["run_id"]
        or binning.algorithm != config.binning.algorithm
        or set(binning.branches) != set(_enabled(config))
    ):
        raise EvaluationInputError("cache/binning/training identity chain differs")
    return EvaluationInputs(
        config=config,
        cache=cache,
        binning=binning,
        run_root=run_root,
        run_dir=run_root / "run",
        identity=identity,
    )


def evaluation_input_hashes(inputs: EvaluationInputs) -> dict[str, str]:
    """Hash the exact immutable inputs bound by an evaluation manifest."""
    paths = {
        "best.pt": inputs.run_dir / "best.pt",
        "training_validation.json": inputs.run_dir / "training_validation.json",
        "training_manifest.json": inputs.run_dir / "training_manifest.json",
        "binning.pt": inputs.run_root / "binning" / "binning.pt",
        "binning_manifest.json": inputs.run_root / "binning" / "binning_manifest.json",
        "cache_manifest.json": inputs.cache.root / "cache_manifest.json",
    }
    if set(paths) != set(EVALUATION_INPUT_FILES):
        raise EvaluationInputError("evaluation input file contract differs")
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise EvaluationInputError(f"evaluation inputs are missing: {sorted(missing)}")
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _load_best_model(inputs: EvaluationInputs, dtype: torch.dtype) -> MultiBranchConfidenceModel:
    model = MultiBranchConfidenceModel.from_config(
        inputs.config, feature_dim=640
    ).to(device=torch.device("cpu"), dtype=dtype)
    best = load_torch_artifact(inputs.run_dir / "best.pt")
    if set(best) != set(BEST_CHECKPOINT_KEYS):
        raise EvaluationInputError("best checkpoint keys differ")
    try:
        _validate_versions(best)
        state = _validate_saved_model(best, model)
    except ValueError as error:
        raise EvaluationInputError(f"best checkpoint is invalid: {error}") from error
    if any(best.get(name) != value for name, value in inputs.identity.items()):
        raise EvaluationInputError("best checkpoint identity differs")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _test_counts(cache: CacheManifest) -> tuple[int, int]:
    shards = cache.splits.get("test", ())
    structures = sum(shard.num_structures for shard in shards)
    atoms = sum(shard.num_atoms for shard in shards)
    if structures <= 0 or atoms <= 0:
        raise EvaluationInputError("test cache is empty")
    return structures, atoms


def _collect_predictions(
    inputs: EvaluationInputs,
    model: MultiBranchConfidenceModel,
    *,
    dtype: torch.dtype,
) -> dict[str, Any]:
    enabled = _enabled(inputs.config)
    structure_ids: list[str] = []
    structure_offsets = [0]
    collected_logits: dict[str, list[torch.Tensor]] = {
        name: [] for name in enabled
    }
    collected_errors: dict[str, list[torch.Tensor]] = {
        name: [] for name in enabled
    }
    with torch.inference_mode():
        for batch in iter_cache_batches(
            inputs.cache,
            "test",
            batch_size=inputs.config.trainer.batch_size,
        ):
            features = batch.scalar_features.to(device="cpu", dtype=dtype)
            offsets = batch.atom_offsets.to(device="cpu", dtype=torch.int64)
            outputs = model(features, offsets)
            if set(outputs) != set(enabled):
                raise EvaluationInputError("model output branches differ")
            structure_ids.extend(batch.structure_id)
            for count in batch.num_atoms.tolist():
                structure_offsets.append(structure_offsets[-1] + int(count))
            if "force" in enabled:
                force_errors = (
                    batch.force_prediction.to(torch.float64)
                    - batch.force_reference.to(torch.float64)
                ).abs().mean(dim=-1)
                collected_errors["force"].append(force_errors.cpu())
                collected_logits["force"].append(outputs["force"].detach().cpu())
            if "energy" in enabled:
                energy_errors = (
                    batch.energy_prediction.to(torch.float64)
                    - batch.energy_reference.to(torch.float64)
                ).abs() / batch.num_atoms.to(torch.float64)
                collected_errors["energy"].append(energy_errors.cpu())
                collected_logits["energy"].append(outputs["energy"].detach().cpu())

    expected_structures, expected_atoms = _test_counts(inputs.cache)
    if len(structure_ids) != expected_structures:
        raise EvaluationInputError("test structure count differs from cache manifest")
    if structure_offsets[-1] != expected_atoms:
        raise EvaluationInputError("test atom count differs from cache manifest")

    predictions: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "formula_version": EVALUATION_FORMULA_VERSION,
        "split": "test",
        "identity": dict(inputs.identity),
        "enabled_branches": enabled,
        "structure_ids": tuple(structure_ids),
        "structure_offsets": torch.tensor(structure_offsets, dtype=torch.int64),
        "force_target_mode": (
            inputs.config.model.force.target_mode
            if inputs.config.force_enabled
            else None
        ),
    }
    for branch in enabled:
        logits = torch.cat(collected_logits[branch], dim=0)
        errors = torch.cat(collected_errors[branch], dim=0).to(torch.float64)
        binning = inputs.binning.branches[branch]
        predictions[branch] = {
            "logits": logits,
            "labels": labels_from_thresholds(errors, binning.thresholds),
            "errors": errors,
            "expected_errors": expected_errors(logits, binning.representatives),
        }
    return predictions


def _metrics_payload(
    inputs: EvaluationInputs, predictions: Mapping[str, Any]
) -> dict[str, Any]:
    branches: dict[str, dict[str, int | float]] = {}
    for branch in predictions["enabled_branches"]:
        payload = predictions[branch]
        branches[branch] = branch_metrics(
            payload["logits"],
            payload["labels"],
            payload["errors"],
            inputs.binning.branches[branch].representatives,
        )
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "formula_version": EVALUATION_FORMULA_VERSION,
        "split": "test",
        "identity": dict(inputs.identity),
        "branches": branches,
    }


def run_evaluate(config: ConfidenceHeadConfig) -> Path:
    """Evaluate one completed run on test cache using CPU and best.pt only."""
    inputs = load_evaluation_inputs(config)
    paths = EvaluationPaths.from_run_root(inputs.run_root)
    input_hashes = evaluation_input_hashes(inputs)
    try:
        if validate_or_reuse_evaluation(paths, inputs.identity, input_hashes):
            return paths.manifest
    except EvaluationConflictError:
        raise
    dtype = _feature_dtype(inputs.cache, config.trainer.batch_size)
    model = _load_best_model(inputs, dtype)
    predictions = _collect_predictions(inputs, model, dtype=dtype)
    metrics = _metrics_payload(inputs, predictions)
    return commit_evaluation(paths, predictions, metrics, input_hashes)
