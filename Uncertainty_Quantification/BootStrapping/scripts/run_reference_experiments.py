"""Run dual MAD E0 post-processing on canonical member prediction arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..bootstrap.dataset_compatibility import filter_supported_structures, extract_targets
from ..bootstrap.errors import HardFailure
from ..bootstrap.reference_experiments import run_reference_experiments
from ._cli import run_cli


def _path(value: Any, base: Path, location: str) -> Path:
    if not isinstance(value, str):
        raise HardFailure(f"{location} must be a path string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise HardFailure(f"could not load reference config {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise HardFailure("reference config schema_version must be 1")
    required = {"model", "validation", "test", "output_root"}
    if set(document) != required:
        raise HardFailure(f"reference config keys must be {sorted(required)}")
    return document


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise HardFailure(f"could not load array artifact {path}: {error}") from error


def _section(document: dict[str, Any], name: str, base: Path) -> dict[str, Any]:
    value = document[name]
    if not isinstance(value, dict):
        raise HardFailure(f"{name} must be a mapping")
    required = {"targets", "matrix", "members"}
    if set(value) != required:
        raise HardFailure(f"{name} keys must be {sorted(required)}")
    members = value["members"]
    if not isinstance(members, list) or len(members) != 8 or not all(isinstance(item, str) for item in members):
        raise HardFailure(f"{name}.members must contain exactly 8 paths")
    return {
        "targets": _path(value["targets"], base, f"{name}.targets"),
        "matrix": _path(value["matrix"], base, f"{name}.matrix"),
        "members": [_path(item, base, f"{name}.members") for item in members],
    }


def _load_matrix(path: Path) -> np.ndarray:
    array = _load_npz(path)
    if set(array) != {"matrix"}:
        raise HardFailure(f"matrix artifact must contain only matrix: {path}")
    matrix = np.asarray(array["matrix"], dtype=float)
    if matrix.ndim != 2:
        raise HardFailure("composition matrix must be two-dimensional")
    return matrix


def _run(config_path: Path) -> Path:
    document = _load_config(config_path)
    base = config_path.parent
    model = document["model"]
    if not isinstance(model, dict) or set(model) != {"atomic_numbers", "e0"}:
        raise HardFailure("model must contain atomic_numbers and e0")
    atomic_numbers = model["atomic_numbers"]
    e0 = np.asarray(model["e0"], dtype=float)
    if not isinstance(atomic_numbers, list) or len(atomic_numbers) != e0.size:
        raise HardFailure("model atomic_numbers and e0 lengths differ")
    validation = _section(document, "validation", base)
    test = _section(document, "test", base)
    validation_targets = _load_npz(validation["targets"])
    test_targets = _load_npz(test["targets"])
    output = run_reference_experiments(
        test_members=test["members"],
        test_targets=test_targets,
        validation_members=validation["members"],
        validation_targets=validation_targets,
        test_matrix=_load_matrix(test["matrix"]),
        validation_matrix=_load_matrix(validation["matrix"]),
        model_atomic_numbers=[int(value) for value in atomic_numbers],
        model_e0=e0,
        output_root=_path(document["output_root"], base, "output_root"),
        metadata={"config_path": str(config_path.resolve())},
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    return run_cli(lambda: print(_run(arguments.config.expanduser().resolve())))


if __name__ == "__main__":
    raise SystemExit(main())
