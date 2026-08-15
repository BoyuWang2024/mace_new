"""Selection and validation of canonical LLPR curvature sources."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    load_torch_artifact,
    require_identity,
    sha256_file,
)
from .checkpoint import CheckpointIdentity
from .config import LLPRConfig
from .curvature import run_root
from .readout import ReadoutLayout


_VARIANTS = ("he", "hf", "hef")


@dataclass(frozen=True)
class LoadedCurvature:
    """A validated curvature artifact together with immutable provenance."""

    path: Path
    sha256: str
    identity: Mapping[str, Any]
    variants: Mapping[str, Tensor]


def _checkpoint_metadata(checkpoint: CheckpointIdentity) -> dict[str, Any]:
    return {
        "sha256": checkpoint.sha256,
        "model_class": checkpoint.model_class,
        "heads": list(checkpoint.heads),
        "selected_head": checkpoint.selected_head,
        "r_max": checkpoint.r_max,
        "atomic_numbers": list(checkpoint.atomic_numbers),
        "dtype": str(checkpoint.dtype),
    }


def _require_canonical_identity(
    identity: Mapping[str, Any],
    config: LLPRConfig,
    checkpoint: CheckpointIdentity,
    layout: ReadoutLayout,
) -> None:
    expected_fields = {
        "schema_version",
        "formula_version",
        "checkpoint",
        "build",
        "readout",
        "curvature",
        "limits",
    }
    if set(identity) != expected_fields:
        raise ValueError("curvature artifact identity fields are invalid")
    if (
        identity["schema_version"] != SCHEMA_VERSION
        or identity["formula_version"] != FORMULA_VERSION
    ):
        raise ValueError("curvature artifact identity version is invalid")
    checkpoint_identity = identity["checkpoint"]
    if not isinstance(checkpoint_identity, Mapping):
        raise ValueError("curvature artifact checkpoint identity is invalid")
    require_identity(checkpoint_identity, _checkpoint_metadata(checkpoint))

    build = identity["build"]
    build_fields = {
        "sha256",
        "identity",
        "size",
        "atomic_numbers",
        "r_max",
        "head",
    }
    if not isinstance(build, Mapping) or set(build) != build_fields:
        raise ValueError("curvature artifact build identity is invalid")
    build_sha256 = build["sha256"]
    if (
        not isinstance(build_sha256, str)
        or len(build_sha256) != 64
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in build_sha256
        )
    ):
        raise ValueError("curvature artifact build SHA256 is invalid")
    if (
        config.build.expected_sha256 is not None
        and build_sha256.lower() != config.build.expected_sha256.lower()
    ):
        raise ValueError("curvature artifact build SHA256 mismatch")
    build_size = build["size"]
    if (
        isinstance(build_size, bool)
        or not isinstance(build_size, int)
        or build_size <= 0
    ):
        raise ValueError("curvature artifact build size is invalid")
    build_identity = build["identity"]
    if (
        not isinstance(build_identity, str)
        or len(build_identity) != 16
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in build_identity
        )
    ):
        raise ValueError("curvature artifact build data identity is invalid")
    if (
        build["atomic_numbers"] != list(checkpoint.atomic_numbers)
        or build["r_max"] != checkpoint.r_max
        or build["head"] != checkpoint.selected_head
    ):
        raise ValueError("curvature artifact build model identity mismatch")

    readout = identity["readout"]
    if not isinstance(readout, Mapping):
        raise ValueError("curvature artifact readout identity is invalid")
    require_identity(readout, layout.metadata())

    expected_curvature = {
        "energy": "outer(d(E/N)/dtheta, d(E/N)/dtheta)",
        "forces": "G_F.T @ G_F (unweighted)",
        "variants": list(_VARIANTS),
    }
    if identity["curvature"] != expected_curvature:
        raise ValueError("curvature artifact formula identity is invalid")
    expected_limits = {
        "max_structures": config.runtime.max_structures,
        "max_force_components_per_structure": (
            config.runtime.max_force_components_per_structure
        ),
    }
    if identity["limits"] != expected_limits:
        raise ValueError("curvature artifact limits identity mismatch")


def curvature_source_path(config: LLPRConfig, checkpoint_sha256: str) -> Path:
    """Select an explicit shared artifact or the run-local default."""
    external = config.artifacts.curvature
    if external is not None:
        return external.path
    return run_root(config, checkpoint_sha256) / "curvature" / "base_curvature.pt"


def load_curvature_source(
    config: LLPRConfig,
    checkpoint: CheckpointIdentity,
    layout: ReadoutLayout,
) -> LoadedCurvature:
    """Load a selected artifact only after its file identity is verified."""
    path = curvature_source_path(config, checkpoint.sha256)
    try:
        file_status = path.lstat()
    except OSError as error:
        raise ValueError(f"curvature artifact must be a regular file: {path}") from error
    if not stat.S_ISREG(file_status.st_mode):
        raise ValueError(f"curvature artifact must be a regular file: {path}")
    if file_status.st_size <= 0:
        raise ValueError(f"curvature artifact must be non-empty: {path}")
    actual_sha256 = sha256_file(path)
    external = config.artifacts.curvature
    expected_sha256 = None if external is None else external.expected_sha256
    if (
        expected_sha256 is not None
        and actual_sha256.lower() != expected_sha256.lower()
    ):
        raise ValueError(
            "curvature artifact SHA256 mismatch: "
            f"expected {expected_sha256}, actual {actual_sha256}"
        )

    artifact = load_torch_artifact(path)
    if sha256_file(path) != actual_sha256:
        raise ValueError("curvature artifact changed while it was being loaded")
    if not isinstance(artifact, Mapping):
        raise ValueError("curvature artifact must be a mapping")
    artifact_fields = {"identity", "status", "structures", "components", "variants"}
    if set(artifact) != artifact_fields:
        raise ValueError("curvature artifact fields are invalid")
    if artifact["status"] != "complete":
        raise ValueError("curvature artifact has an invalid status")

    identity = artifact["identity"]
    if not isinstance(identity, Mapping):
        raise ValueError("curvature artifact identity must be a mapping")
    _require_canonical_identity(identity, config, checkpoint, layout)
    structures = artifact["structures"]
    components = artifact["components"]
    if isinstance(structures, bool) or not isinstance(structures, int) or structures <= 0:
        raise ValueError("curvature artifact structures count is invalid")
    if isinstance(components, bool) or not isinstance(components, int) or components <= 0:
        raise ValueError("curvature artifact components count is invalid")
    build = identity["build"]
    if not isinstance(build, Mapping):
        raise ValueError("curvature artifact build identity is invalid")
    target_structures = int(build["size"])
    if config.runtime.max_structures is not None:
        target_structures = min(target_structures, config.runtime.max_structures)
    if structures != target_structures:
        raise ValueError("curvature artifact structures count mismatch")

    source_variants = artifact.get("variants")
    if not isinstance(source_variants, Mapping) or set(source_variants) != set(_VARIANTS):
        raise ValueError("curvature artifact variants are invalid")
    variants: dict[str, Tensor] = {}
    for variant in _VARIANTS:
        matrix = source_variants[variant]
        if (
            not isinstance(matrix, Tensor)
            or matrix.shape != (layout.size, layout.size)
            or matrix.device.type != "cpu"
            or matrix.dtype != torch.float64
            or not torch.isfinite(matrix).all()
        ):
            raise ValueError(f"curvature artifact {variant} matrix is invalid")
        scale = max(1.0, float(matrix.abs().max()))
        tolerance = 64.0 * torch.finfo(torch.float64).eps * scale
        if float((matrix - matrix.T).abs().max()) > tolerance:
            raise ValueError(f"curvature artifact {variant} matrix must be symmetric")
        variants[variant] = matrix
    if not torch.equal(variants["hef"], variants["he"] + variants["hf"]):
        raise ValueError("curvature artifact hef must equal he plus hf")

    return LoadedCurvature(
        path=path,
        sha256=actual_sha256,
        identity=identity,
        variants=variants,
    )
