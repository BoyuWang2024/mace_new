"""Dataset-size-independent signatures for canonical BootStrapping runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import HardFailure
from .validation import validate_run


def _npz_signature(path: Path) -> dict[str, dict[str, Any]]:
    """Describe array contracts while excluding sample and atom counts."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {
                name: {
                    "dtype_kind": archive[name].dtype.kind,
                    "rank": archive[name].ndim,
                    "trailing_shape": list(archive[name].shape[1:]),
                }
                for name in sorted(archive.files)
            }
    except (OSError, ValueError) as error:
        raise HardFailure(f"could not inspect canonical NPZ {path}: {error}") from error


def _manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HardFailure(f"could not read run manifest {path}: {error}") from error
    if not isinstance(document, dict):
        raise HardFailure(f"run manifest root must be an object: {path}")
    return document


def core_run_schema_signature(root: str | Path) -> dict[str, Any]:
    """Return the result-format contract shared by small and full migrations.

    The signature intentionally excludes member count, structure/atom counts,
    hashes, byte sizes, and numerical values. It is therefore suitable for
    proving that a small migration exercises the same public storage schema as
    a full migration without claiming that their scientific data are equal.
    """
    run = Path(root).expanduser().resolve()
    validate_run(run)
    manifest = _manifest(run / "run_manifest.json")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise HardFailure("run manifest metadata must be an object")
    modes = metadata.get("parameter_modes")
    splits = metadata.get("splits")
    if not isinstance(modes, list) or not modes or not isinstance(splits, list) or not splits:
        raise HardFailure("run manifest modes/splits are invalid")

    member_root = run / "members" / "member_000"
    member_artifacts = sorted(
        path.relative_to(member_root).as_posix()
        for path in member_root.rglob("*")
        if path.is_file()
    )
    first_split = str(splits[0])
    first_mode = str(modes[0])
    analysis_root = run / "analysis" / first_split / first_mode

    return {
        "schema": manifest.get("schema"),
        "modes": modes,
        "splits": splits,
        "member_artifacts": member_artifacts,
        "targets": _npz_signature(run / "predictions" / first_split / "targets.npz"),
        "member_prediction": _npz_signature(
            run / "predictions" / first_split / "members" / "member_000" / f"{first_mode}.npz"
        ),
        "ensemble": _npz_signature(run / "ensemble" / first_split / f"{first_mode}.npz"),
        "uncertainty": _npz_signature(run / "uncertainty" / first_split / f"{first_mode}.npz"),
        "analysis_files": sorted(path.name for path in analysis_root.iterdir() if path.is_file()),
    }
