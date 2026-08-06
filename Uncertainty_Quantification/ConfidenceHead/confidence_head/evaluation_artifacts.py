"""Immutable schemas and manifests for test-set ConfidenceHead evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .artifacts import atomic_json_dump, atomic_torch_save, load_torch_artifact
from .identity import sha256_file
from .metrics import validate_metric_payload


EVALUATION_SCHEMA_VERSION = 1
EVALUATION_FORMULA_VERSION = "mace_confidence_head_test_evaluation_v1"
EVALUATION_INPUT_FILES = (
    "best.pt",
    "training_validation.json",
    "training_manifest.json",
    "binning.pt",
    "binning_manifest.json",
    "cache_manifest.json",
)
_IDENTITY_KEYS = {"run_id", "experiment_id", "cache_id", "binning_id"}
_BRANCH_KEYS = {"logits", "labels", "errors", "expected_errors"}
_BASE_PREDICTION_KEYS = {
    "schema_version",
    "formula_version",
    "split",
    "identity",
    "enabled_branches",
    "structure_ids",
    "structure_offsets",
    "force_target_mode",
}
_METRICS_KEYS = {
    "schema_version",
    "formula_version",
    "split",
    "identity",
    "branches",
}
_MANIFEST_KEYS = {
    "schema_version",
    "formula_version",
    "split",
    "identity",
    "input_sha256",
    "output_sha256",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


class EvaluationConflictError(RuntimeError):
    """Evaluation evidence is partial, invalid, or bound to other inputs."""


@dataclass(frozen=True)
class EvaluationPaths:
    """Canonical per-run evaluation artifact paths."""

    run_root: Path
    predictions: Path
    metrics: Path
    manifest: Path

    @classmethod
    def from_run_root(cls, run_root: Path) -> "EvaluationPaths":
        root = Path(run_root)
        run = root / "run"
        return cls(
            run_root=root,
            predictions=run / "test_predictions.pt",
            metrics=run / "test_metrics.json",
            manifest=run / "evaluation_manifest.json",
        )

    @property
    def outputs(self) -> tuple[Path, Path, Path]:
        return (self.predictions, self.metrics, self.manifest)


def _exact_mapping(value: object, keys: set[str], where: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise EvaluationConflictError(f"{where} must be a mapping")
    if set(value) != keys:
        raise EvaluationConflictError(f"{where} keys differ")
    return value


def _identity(value: object, *, where: str = "identity") -> dict[str, str]:
    mapping = _exact_mapping(value, _IDENTITY_KEYS, where)
    result: dict[str, str] = {}
    for name in sorted(_IDENTITY_KEYS):
        item = mapping[name]
        if type(item) is not str or _HEX64.fullmatch(item) is None:
            raise EvaluationConflictError(f"{where}.{name} must be a SHA-256 id")
        result[name] = item
    return result


def _expected_identity(value: object) -> dict[str, str]:
    try:
        return _identity(value, where="expected identity")
    except EvaluationConflictError:
        raise
    except Exception as error:
        raise EvaluationConflictError(f"expected identity is invalid: {error}") from error


def _hashes(
    value: object, *, expected_names: set[str], where: str
) -> dict[str, str]:
    mapping = _exact_mapping(value, expected_names, where)
    result: dict[str, str] = {}
    for name in sorted(expected_names):
        digest = mapping[name]
        if type(digest) is not str or _HEX64.fullmatch(digest) is None:
            raise EvaluationConflictError(f"{where}.{name} must be SHA-256")
        result[name] = digest
    return result


def _enabled_branches(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise EvaluationConflictError("enabled_branches must be a non-empty tuple")
    if any(type(item) is not str for item in value):
        raise EvaluationConflictError("enabled_branches entries must be strings")
    canonical = tuple(name for name in ("force", "energy") if name in value)
    if value != canonical or len(set(value)) != len(value):
        raise EvaluationConflictError("enabled_branches differ from supported branches")
    return value


def _structure_ids(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise EvaluationConflictError("structure_ids must be a non-empty tuple")
    if any(type(item) is not str or not item for item in value):
        raise EvaluationConflictError("structure_ids entries must be non-empty strings")
    if len(set(value)) != len(value):
        raise EvaluationConflictError("structure_ids must be unique")
    return value


def _structure_offsets(value: object, *, structures: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise EvaluationConflictError("structure_offsets must be a tensor")
    if value.ndim != 1 or value.numel() != structures + 1:
        raise EvaluationConflictError(
            "structure_offsets length must equal structure count plus one"
        )
    if value.dtype not in _INTEGER_DTYPES:
        raise EvaluationConflictError("structure_offsets must use an integer dtype")
    offsets = value.detach().to(device="cpu", dtype=torch.int64)
    if int(offsets[0]) != 0:
        raise EvaluationConflictError("structure_offsets must start at zero")
    if not bool(torch.all(offsets[1:] > offsets[:-1])):
        raise EvaluationConflictError("structure_offsets must be strictly increasing")
    return offsets


def _branch_payload(
    value: object, *, branch: str, expected_samples: int
) -> dict[str, torch.Tensor]:
    mapping = _exact_mapping(value, _BRANCH_KEYS, f"{branch} branch")
    logits = mapping["logits"]
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or logits.shape[0] != expected_samples
        or logits.shape[1] < 2
        or not logits.dtype.is_floating_point
    ):
        raise EvaluationConflictError(
            f"{branch} logits sample/bin shape or dtype differs"
        )
    logits = logits.detach().to(device="cpu")
    if not bool(torch.isfinite(logits).all()):
        raise EvaluationConflictError(f"{branch} logits must be finite")
    bins = int(logits.shape[1])

    labels = mapping["labels"]
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or labels.numel() != expected_samples
        or labels.dtype not in _INTEGER_DTYPES
    ):
        raise EvaluationConflictError(f"{branch} labels sample shape or dtype differs")
    labels = labels.detach().to(device="cpu", dtype=torch.int64)
    if bool(torch.any(labels < 0)) or bool(torch.any(labels >= bins)):
        raise EvaluationConflictError(f"{branch} labels contain invalid bins")

    vectors: dict[str, torch.Tensor] = {}
    for name in ("errors", "expected_errors"):
        item = mapping[name]
        if (
            not isinstance(item, torch.Tensor)
            or item.ndim != 1
            or item.numel() != expected_samples
            or not item.dtype.is_floating_point
        ):
            raise EvaluationConflictError(
                f"{branch} {name} sample shape or dtype differs"
            )
        bound = item.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(bound).all()):
            raise EvaluationConflictError(f"{branch} {name} must be finite")
        if bool(torch.any(bound < 0)):
            raise EvaluationConflictError(f"{branch} {name} must be non-negative")
        vectors[name] = bound
    return {
        "logits": logits,
        "labels": labels,
        "errors": vectors["errors"],
        "expected_errors": vectors["expected_errors"],
    }


def validate_prediction_payload(
    value: object, *, expected_identity: Mapping[str, str]
) -> dict[str, Any]:
    """Validate one complete test prediction artifact without trusting filenames."""
    identity = _expected_identity(dict(expected_identity))
    if type(value) is not dict:
        raise EvaluationConflictError("prediction payload must be a mapping")
    enabled = _enabled_branches(value.get("enabled_branches"))
    present_branches = {"force", "energy"} & set(value)
    if present_branches != set(enabled):
        raise EvaluationConflictError("prediction enabled branches differ from payload")

    expected_keys = _BASE_PREDICTION_KEYS | set(enabled)
    mapping = _exact_mapping(value, expected_keys, "prediction payload")
    if mapping["schema_version"] != EVALUATION_SCHEMA_VERSION:
        raise EvaluationConflictError("prediction schema_version differs")
    if mapping["formula_version"] != EVALUATION_FORMULA_VERSION:
        raise EvaluationConflictError("prediction formula_version differs")
    if mapping["split"] != "test":
        raise EvaluationConflictError("prediction split must be test")
    if _identity(mapping["identity"]) != identity:
        raise EvaluationConflictError("prediction identity differs")
    structure_ids = _structure_ids(mapping["structure_ids"])
    offsets = _structure_offsets(
        mapping["structure_offsets"], structures=len(structure_ids)
    )
    force_target_mode = mapping["force_target_mode"]
    if "force" in enabled:
        if force_target_mode != "atom_mean":
            raise EvaluationConflictError("force_target_mode must be atom_mean")
    elif force_target_mode is not None:
        raise EvaluationConflictError("force_target_mode must be null without force")

    result = dict(mapping)
    result["identity"] = identity
    result["structure_ids"] = structure_ids
    result["structure_offsets"] = offsets
    if "force" in enabled:
        result["force"] = _branch_payload(
            mapping["force"], branch="force", expected_samples=int(offsets[-1])
        )
    if "energy" in enabled:
        result["energy"] = _branch_payload(
            mapping["energy"],
            branch="energy",
            expected_samples=len(structure_ids),
        )
    return result


def _validate_metrics_payload(
    value: object,
    *,
    expected_identity: Mapping[str, str],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    mapping = _exact_mapping(value, _METRICS_KEYS, "metrics payload")
    if mapping["schema_version"] != EVALUATION_SCHEMA_VERSION:
        raise EvaluationConflictError("metrics schema_version differs")
    if mapping["formula_version"] != EVALUATION_FORMULA_VERSION:
        raise EvaluationConflictError("metrics formula_version differs")
    if mapping["split"] != "test":
        raise EvaluationConflictError("metrics split must be test")
    identity = _expected_identity(dict(expected_identity))
    if _identity(mapping["identity"]) != identity:
        raise EvaluationConflictError("metrics identity differs")
    branches = mapping["branches"]
    expected_branches = set(predictions["enabled_branches"])
    branch_mapping = _exact_mapping(branches, expected_branches, "metrics branches")
    validated: dict[str, dict[str, int | float]] = {}
    for branch in sorted(expected_branches):
        try:
            metrics = validate_metric_payload(branch_mapping[branch])
        except ValueError as error:
            raise EvaluationConflictError(
                f"{branch} metrics are invalid: {error}"
            ) from error
        samples = int(predictions[branch]["errors"].numel())
        if metrics["sample_count"] != samples:
            raise EvaluationConflictError(
                f"{branch} metrics sample_count differs from predictions"
            )
        validated[branch] = metrics
    result = dict(mapping)
    result["identity"] = identity
    result["branches"] = validated
    return result


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise EvaluationConflictError(f"{where} is unreadable: {error}") from error
    if type(value) is not dict:
        raise EvaluationConflictError(f"{where} must contain a JSON object")
    return value


def _manifest_payload(
    paths: EvaluationPaths,
    identity: Mapping[str, str],
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "formula_version": EVALUATION_FORMULA_VERSION,
        "split": "test",
        "identity": _expected_identity(dict(identity)),
        "input_sha256": _hashes(
            dict(input_hashes),
            expected_names=set(EVALUATION_INPUT_FILES),
            where="input hashes",
        ),
        "output_sha256": {
            paths.predictions.name: sha256_file(paths.predictions),
            paths.metrics.name: sha256_file(paths.metrics),
        },
    }



def _validate_manifest(
    value: object,
    *,
    paths: EvaluationPaths,
    expected_identity: Mapping[str, str],
    expected_input_hashes: Mapping[str, str],
) -> None:
    mapping = _exact_mapping(value, _MANIFEST_KEYS, "evaluation manifest")
    if mapping["schema_version"] != EVALUATION_SCHEMA_VERSION:
        raise EvaluationConflictError("evaluation manifest schema_version differs")
    if mapping["formula_version"] != EVALUATION_FORMULA_VERSION:
        raise EvaluationConflictError("evaluation manifest formula_version differs")
    if mapping["split"] != "test":
        raise EvaluationConflictError("evaluation manifest split must be test")
    if _identity(mapping["identity"]) != _expected_identity(dict(expected_identity)):
        raise EvaluationConflictError("evaluation manifest identity differs")
    actual_inputs = _hashes(
        mapping["input_sha256"],
        expected_names=set(EVALUATION_INPUT_FILES),
        where="evaluation manifest input_sha256",
    )
    expected_inputs = _hashes(
        dict(expected_input_hashes),
        expected_names=set(EVALUATION_INPUT_FILES),
        where="expected input hashes",
    )
    if actual_inputs != expected_inputs:
        raise EvaluationConflictError("evaluation manifest input hashes differ")
    recorded_outputs = _hashes(
        mapping["output_sha256"],
        expected_names={paths.predictions.name, paths.metrics.name},
        where="evaluation manifest output_sha256",
    )
    for path in (paths.predictions, paths.metrics):
        if recorded_outputs[path.name] != sha256_file(path):
            raise EvaluationConflictError(
                f"{path.name} sha256 differs from evaluation manifest"
            )


def validate_or_reuse_evaluation(
    paths: EvaluationPaths,
    expected_identity: Mapping[str, str],
    expected_input_hashes: Mapping[str, str],
) -> bool:
    """Return true for a complete identical evaluation, false for no evidence."""
    evidence = {path: path.exists() for path in paths.outputs}
    if not any(evidence.values()):
        return False
    if not all(evidence.values()):
        raise EvaluationConflictError(
            "evaluation outputs are partial; preserve evidence and investigate"
        )
    _validate_manifest(
        _read_json(paths.manifest, where="evaluation manifest"),
        paths=paths,
        expected_identity=expected_identity,
        expected_input_hashes=expected_input_hashes,
    )
    predictions = validate_prediction_payload(
        load_torch_artifact(paths.predictions),
        expected_identity=expected_identity,
    )
    _validate_metrics_payload(
        _read_json(paths.metrics, where="test metrics"),
        expected_identity=expected_identity,
        predictions=predictions,
    )
    return True


def commit_evaluation(
    paths: EvaluationPaths,
    predictions: Mapping[str, Any],
    metrics: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> Path:
    """Atomically write data artifacts and commit their manifest last."""
    if any(path.exists() for path in paths.outputs):
        raise EvaluationConflictError(
            "evaluation outputs already exist; validate or preserve partial evidence"
        )
    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity(predictions.get("identity"), where="prediction identity")
    validated_predictions = validate_prediction_payload(
        dict(predictions), expected_identity=identity
    )
    validated_metrics = _validate_metrics_payload(
        dict(metrics),
        expected_identity=identity,
        predictions=validated_predictions,
    )
    _hashes(
        dict(input_hashes),
        expected_names=set(EVALUATION_INPUT_FILES),
        where="input hashes",
    )

    atomic_torch_save(paths.predictions, dict(predictions))
    validate_prediction_payload(
        load_torch_artifact(paths.predictions), expected_identity=identity
    )
    atomic_json_dump(paths.metrics, validated_metrics)
    reloaded_metrics = _read_json(paths.metrics, where="new test metrics")
    _validate_metrics_payload(
        reloaded_metrics,
        expected_identity=identity,
        predictions=validated_predictions,
    )
    atomic_json_dump(
        paths.manifest,
        _manifest_payload(paths, identity, input_hashes),
    )
    _validate_manifest(
        _read_json(paths.manifest, where="new evaluation manifest"),
        paths=paths,
        expected_identity=identity,
        expected_input_hashes=input_hashes,
    )
    return paths.manifest
