"""Evaluate one completed best checkpoint against a shared external cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..binning import labels_from_thresholds
from ..cache import CacheManifest, iter_cache_batches, load_complete_cache
from ..config import ConfidenceHeadConfig
from ..evaluation_artifacts import (
    EvaluationPaths,
    commit_evaluation,
    validate_or_reuse_evaluation,
)
from ..external_config import ExternalInferenceConfig
from ..identity import sha256_file
from ..metrics import expected_errors
from .build_external_cache import external_cache_id, external_cache_root
from .evaluate import (
    _load_best_model,
    _metrics_payload,
    evaluation_input_hashes,
    load_evaluation_inputs,
)
from .train import _feature_dtype


class ExternalEvaluationInputError(RuntimeError):
    """External cache or requested production head is invalid."""


def resolve_head_config(
    config: ExternalInferenceConfig, head_key: str
) -> ConfidenceHeadConfig:
    if head_key == "force":
        return config.force_config
    if head_key.startswith("energy_order"):
        suffix = head_key.removeprefix("energy_order")
        if suffix.isdigit() and int(suffix) in config.energy_configs:
            return config.energy_configs[int(suffix)]
    raise ExternalEvaluationInputError(f"unsupported head key: {head_key}")


def _paths(config: ExternalInferenceConfig, head_key: str) -> EvaluationPaths:
    root = config.output_root / config.dataset.name / "evaluations" / head_key
    return EvaluationPaths(
        run_root=root,
        predictions=root / "predictions.pt",
        metrics=root / "metrics.json",
        manifest=root / "evaluation_manifest.json",
    )


def _collect(
    inputs,
    model,
    cache: CacheManifest,
    *,
    split: str,
    batch_size: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    enabled = tuple(
        branch
        for branch, active in (
            ("force", inputs.config.force_enabled),
            ("energy", inputs.config.energy_enabled),
        )
        if active
    )
    structure_ids: list[str] = []
    structure_offsets = [0]
    logits = {branch: [] for branch in enabled}
    errors = {branch: [] for branch in enabled}
    with torch.inference_mode():
        for batch in iter_cache_batches(cache, split, batch_size):
            features = batch.scalar_features.to(device="cpu", dtype=dtype)
            offsets = batch.atom_offsets.to(device="cpu", dtype=torch.int64)
            outputs = model(features, offsets)
            if set(outputs) != set(enabled):
                raise ExternalEvaluationInputError("model output branches differ")
            structure_ids.extend(batch.structure_id)
            for count in batch.num_atoms.tolist():
                structure_offsets.append(structure_offsets[-1] + int(count))
            if "force" in enabled:
                errors["force"].append(
                    (batch.force_prediction.to(torch.float64)
                     - batch.force_reference.to(torch.float64)).abs().mean(dim=-1)
                )
                logits["force"].append(outputs["force"].detach().cpu())
            if "energy" in enabled:
                errors["energy"].append(
                    (batch.energy_prediction.to(torch.float64)
                     - batch.energy_reference.to(torch.float64)).abs()
                    / batch.num_atoms.to(torch.float64)
                )
                logits["energy"].append(outputs["energy"].detach().cpu())
    expected_structures = sum(
        shard.num_structures for shard in cache.splits[split]
    )
    expected_atoms = sum(shard.num_atoms for shard in cache.splits[split])
    if len(structure_ids) != expected_structures or structure_offsets[-1] != expected_atoms:
        raise ExternalEvaluationInputError("external prediction counts differ")
    predictions: dict[str, Any] = {
        "schema_version": 1,
        "formula_version": "mace_confidence_head_test_evaluation_v1",
        "split": "test",
        "identity": dict(inputs.identity),
        "enabled_branches": enabled,
        "structure_ids": tuple(structure_ids),
        "structure_offsets": torch.tensor(structure_offsets, dtype=torch.int64),
        "force_target_mode": (
            inputs.config.model.force.target_mode if inputs.config.force_enabled else None
        ),
    }
    for branch in enabled:
        branch_logits = torch.cat(logits[branch], dim=0)
        branch_errors = torch.cat(errors[branch], dim=0).to(torch.float64)
        bins = inputs.binning.branches[branch]
        predictions[branch] = {
            "logits": branch_logits,
            "labels": labels_from_thresholds(branch_errors, bins.thresholds),
            "errors": branch_errors,
            "expected_errors": expected_errors(branch_logits, bins.representatives),
        }
    return predictions


def run_evaluate_external(
    config: ExternalInferenceConfig, head_key: str
) -> Path:
    """Run one existing best.pt on the committed external inference split."""
    training_config = resolve_head_config(config, head_key)
    inputs = load_evaluation_inputs(training_config)
    cache = load_complete_cache(
        external_cache_root(config), expected_cache_id=external_cache_id(config)
    )
    if set(cache.splits) != {config.cache.split}:
        raise ExternalEvaluationInputError("external cache split set differs")
    paths = _paths(config, head_key)
    input_hashes = evaluation_input_hashes(inputs)
    input_hashes["cache_manifest.json"] = sha256_file(
        cache.root / "cache_manifest.json"
    )
    if validate_or_reuse_evaluation(paths, inputs.identity, input_hashes):
        return paths.manifest
    dtype = _feature_dtype(cache, config.head_batch_size)
    model = _load_best_model(inputs, dtype)
    predictions = _collect(
        inputs,
        model,
        cache,
        split=config.cache.split,
        batch_size=config.head_batch_size,
        dtype=dtype,
    )
    metrics = _metrics_payload(inputs, predictions)
    return commit_evaluation(paths, predictions, metrics, input_hashes)
