"""Unweighted LLPR curvature construction with crash-safe resume."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    atomic_json_dump,
    atomic_torch_save,
    canonical_json,
    load_torch_artifact,
    require_identity,
)
from .checkpoint import CheckpointIdentity, load_checkpoint
from .config import LLPRConfig
from .data import DatasetHandle, build_dataset, iter_samples
from .observables import compute_structure_jacobians
from .readout import ReadoutLayout, discover_readout_layout


def curvature_variants(g_energy: Tensor, g_forces: Tensor) -> dict[str, Tensor]:
    """Return one structure's unweighted energy and force Gram matrices."""
    energy = g_energy.detach().to(device="cpu", dtype=torch.float64)
    forces = g_forces.detach().to(device="cpu", dtype=torch.float64)
    if energy.ndim != 1:
        raise ValueError("g_energy must be one-dimensional")
    if forces.ndim != 2 or forces.shape[1] != energy.numel():
        raise ValueError("g_forces must have shape (components, parameters)")
    he = torch.outer(energy, energy)
    hf = forces.T @ forces
    return {"he": he, "hf": hf, "hef": he + hf}


def run_root(config: LLPRConfig, checkpoint_sha: str) -> Path:
    """Return the stable run root for a checkpoint and experiment."""
    return config.output_root / config.experiment / checkpoint_sha[:12]


def _checkpoint_metadata(identity: CheckpointIdentity) -> dict[str, Any]:
    return {
        "sha256": identity.sha256,
        "model_class": identity.model_class,
        "heads": list(identity.heads),
        "selected_head": identity.selected_head,
        "r_max": identity.r_max,
        "atomic_numbers": list(identity.atomic_numbers),
        "dtype": str(identity.dtype),
    }


def _dataset_metadata(dataset: DatasetHandle) -> dict[str, Any]:
    return {
        "sha256": dataset.sha256,
        "identity": dataset.identity,
        "size": dataset.size,
        "atomic_numbers": list(dataset.atomic_numbers),
        "r_max": dataset.r_max,
        "head": dataset.head,
    }


def _build_identity(
    config: LLPRConfig,
    checkpoint: CheckpointIdentity,
    dataset: DatasetHandle,
    layout: ReadoutLayout,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "checkpoint": _checkpoint_metadata(checkpoint),
        "build": _dataset_metadata(dataset),
        "readout": layout.metadata(),
        "curvature": {
            "energy": "outer(d(E/N)/dtheta, d(E/N)/dtheta)",
            "forces": "G_F.T @ G_F (unweighted)",
            "variants": list(config.curvature.variants),
        },
        "limits": {
            "max_structures": config.runtime.max_structures,
            "max_force_components_per_structure": (
                config.runtime.max_force_components_per_structure
            ),
        },
    }


def _new_progress(identity: Mapping[str, Any], size: int) -> dict[str, Any]:
    return {
        "identity": dict(identity),
        "status": "in_progress",
        "next_index": 0,
        "structures": 0,
        "components": 0,
        "he": torch.zeros((size, size), dtype=torch.float64, device="cpu"),
        "hf": torch.zeros((size, size), dtype=torch.float64, device="cpu"),
    }


def _validate_progress(progress: Mapping[str, Any], size: int) -> None:
    if progress.get("status") not in ("in_progress", "complete"):
        raise ValueError("curvature progress has an invalid status")
    for field in ("next_index", "structures", "components"):
        value = progress.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"curvature progress {field} must be non-negative")
    for field in ("he", "hf"):
        matrix = progress.get(field)
        if (
            not isinstance(matrix, Tensor)
            or matrix.shape != (size, size)
            or matrix.device.type != "cpu"
            or matrix.dtype != torch.float64
            or not torch.isfinite(matrix).all()
        ):
            raise ValueError(
                f"curvature progress {field} must be CPU float64 with shape "
                f"({size}, {size})"
            )


def _matching_identity(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    return canonical_json(actual) == canonical_json(expected)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("complete curvature diagnostics are invalid strict JSON") from error


def _require_scale_aware_symmetry(matrix: Tensor, variant: str) -> None:
    scale = max(1.0, float(matrix.abs().max()))
    tolerance = 64.0 * torch.finfo(torch.float64).eps * scale
    asymmetry = float((matrix - matrix.T).abs().max())
    if asymmetry > tolerance:
        raise ValueError(
            f"complete curvature {variant} matrix must be symmetric"
        )


def _load_complete(
    artifact_path: Path,
    diagnostics_path: Path,
    progress: Mapping[str, Any],
    identity: Mapping[str, Any],
    size: int,
    target_structures: int,
) -> None:
    if not artifact_path.is_file() or not diagnostics_path.is_file():
        raise ValueError("complete curvature cache is missing an artifact")
    if (
        progress.get("status") != "complete"
        or progress.get("next_index") != target_structures
        or progress.get("structures") != target_structures
    ):
        raise ValueError("complete curvature progress count mismatch")

    artifact = load_torch_artifact(artifact_path)
    if not isinstance(artifact, Mapping):
        raise ValueError("complete curvature artifact must be a mapping")
    require_identity(artifact.get("identity", {}), identity)
    if artifact.get("status") != "complete":
        raise ValueError("complete curvature artifact has an invalid status")
    if artifact.get("structures") != progress.get("structures") or artifact.get(
        "components"
    ) != progress.get("components"):
        raise ValueError("complete curvature artifact count mismatch")

    source_variants = artifact.get("variants")
    if not isinstance(source_variants, Mapping) or set(source_variants) != {
        "he",
        "hf",
        "hef",
    }:
        raise ValueError("complete curvature artifact variants are invalid")
    variants: dict[str, Tensor] = {}
    for variant in ("he", "hf", "hef"):
        matrix = source_variants[variant]
        if (
            not isinstance(matrix, Tensor)
            or matrix.shape != (size, size)
            or matrix.device.type != "cpu"
            or matrix.dtype != torch.float64
            or not torch.isfinite(matrix).all()
        ):
            raise ValueError(f"complete curvature {variant} matrix is invalid")
        variants[variant] = matrix
        _require_scale_aware_symmetry(matrix, variant)
    if not torch.equal(variants["he"], progress["he"]) or not torch.equal(
        variants["hf"], progress["hf"]
    ):
        raise ValueError("complete curvature artifact disagrees with progress")
    if not torch.equal(variants["hef"], variants["he"] + variants["hf"]):
        raise ValueError("complete curvature hef must equal he plus hf")

    diagnostics = _load_strict_json(diagnostics_path)
    expected_diagnostics = {
        "identity": dict(identity),
        "status": "complete",
        "structures": progress["structures"],
        "components": progress["components"],
        "matrix_size": size,
        "dtype": "torch.float64",
    }
    if canonical_json(diagnostics) != canonical_json(expected_diagnostics):
        raise ValueError("complete curvature diagnostics mismatch")


def run_build(config: LLPRConfig) -> Path:
    """Build, resume, or reuse the unweighted LLPR base curvature."""
    requested_device = torch.device(config.runtime.device)
    loaded = load_checkpoint(
        config.checkpoint,
        torch.device("cpu"),
        selected_head=config.selected_head,
        expected_readout_size=config.expected_readout_size,
    )
    layout = discover_readout_layout(loaded.model)
    dataset = build_dataset(
        config.build.path,
        config.build.expected_sha256,
        loaded.identity.atomic_numbers,
        loaded.identity.r_max,
        loaded.identity.selected_head,
    )
    identity = _build_identity(config, loaded.identity, dataset, layout)
    target_structures = dataset.size
    if config.runtime.max_structures is not None:
        target_structures = min(target_structures, config.runtime.max_structures)

    curvature_dir = run_root(config, loaded.identity.sha256) / "curvature"
    progress_path = curvature_dir / "progress.pt"
    artifact_path = curvature_dir / "base_curvature.pt"
    diagnostics_path = curvature_dir / "diagnostics.json"

    internal_artifacts = (artifact_path, diagnostics_path)
    progress: dict[str, Any] | None = None
    if not progress_path.exists() and any(path.exists() for path in internal_artifacts):
        raise ValueError("curvature artifact exists without progress")
    if progress_path.exists():
        candidate = load_torch_artifact(progress_path)
        if not isinstance(candidate, Mapping):
            raise ValueError("curvature progress must be a mapping")
        candidate_identity = candidate.get("identity", {})
        if not isinstance(candidate_identity, Mapping):
            raise ValueError("curvature progress identity mismatch")
        require_identity(candidate_identity, identity)
        _validate_progress(candidate, layout.size)
        if candidate.get("status") == "complete":
            _load_complete(
                artifact_path,
                diagnostics_path,
                candidate,
                identity,
                layout.size,
                target_structures,
            )
            return artifact_path
        if config.runtime.resume:
            progress = dict(candidate)

    loaded.model.to(requested_device)
    if progress is None:
        progress = _new_progress(identity, layout.size)
        atomic_torch_save(progress_path, progress)

    if progress["next_index"] > target_structures:
        raise ValueError("curvature progress next_index exceeds the build limit")

    remaining = target_structures - progress["next_index"]
    if remaining > 0:
        samples = iter_samples(
            dataset,
            device=requested_device,
            dtype=loaded.identity.dtype,
            start_index=progress["next_index"],
            max_structures=remaining,
        )
        for sample in samples:
            jacobians = compute_structure_jacobians(
                model=loaded.model,
                batch=sample.batch,
                layout=layout,
                force_component_chunk_size=(
                    config.runtime.force_component_chunk_size
                ),
                max_force_components=(
                    config.runtime.max_force_components_per_structure
                ),
            )
            contribution = curvature_variants(
                jacobians.g_energy,
                jacobians.g_forces,
            )
            progress["he"].add_(contribution["he"])
            progress["hf"].add_(contribution["hf"])
            progress["next_index"] = sample.index + 1
            progress["structures"] += 1
            progress["components"] += int(jacobians.g_forces.shape[0])
            if (
                progress["structures"]
                % config.runtime.save_every_structures
                == 0
            ):
                atomic_torch_save(progress_path, progress)

    variants = {
        "he": progress["he"],
        "hf": progress["hf"],
        "hef": progress["he"] + progress["hf"],
    }
    artifact = {
        "identity": identity,
        "status": "complete",
        "structures": progress["structures"],
        "components": progress["components"],
        "variants": variants,
    }
    atomic_torch_save(artifact_path, artifact)
    atomic_json_dump(
        diagnostics_path,
        {
            "identity": identity,
            "status": "complete",
            "structures": progress["structures"],
            "components": progress["components"],
            "matrix_size": layout.size,
            "dtype": "torch.float64",
        },
    )
    progress["status"] = "complete"
    atomic_torch_save(progress_path, progress)
    return artifact_path
