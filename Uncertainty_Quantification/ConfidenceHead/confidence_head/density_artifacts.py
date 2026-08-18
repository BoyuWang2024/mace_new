"""Deterministic publication artifacts for ConfidenceHead density plots."""
from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .artifacts import atomic_json_dump
from .density_plot import DensityGrid, DensityMetrics, DensityPlotSettings, ErrorPairs
from .identity import sha256_file


class PlotConflictError(RuntimeError):
    """Raised when publication would mix or overwrite incompatible artifacts."""


_POINT_COLUMNS = (
    "dataset_index",
    "structure_id",
    "atom_index",
    "expected_error",
    "actual_error",
    "log10_expected_error",
    "log10_actual_error",
)
_DENSITY_COLUMNS = (
    "x_center",
    "y_center",
    "density",
    "contour_level",
    "grid_x_index",
    "grid_y_index",
)


def _float(value: float) -> str:
    return format(float(value), ".16g")


def _csv_bytes(columns: tuple[str, ...], rows: list[tuple[Any, ...]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def points_csv_bytes(pairs: ErrorPairs) -> bytes:
    """Serialize every valid pair in stable input order."""
    rows: list[tuple[Any, ...]] = []
    for index, (structure_id, expected, actual, lx, ly) in enumerate(
        zip(
            pairs.sample_ids,
            pairs.expected,
            pairs.actual,
            pairs.expected_log10,
            pairs.actual_log10,
            strict=True,
        )
    ):
        atom_index = "" if pairs.atom_indices is None else pairs.atom_indices[index]
        rows.append(
            (
                index,
                structure_id,
                atom_index,
                _float(expected),
                _float(actual),
                _float(lx),
                _float(ly),
            )
        )
    return _csv_bytes(_POINT_COLUMNS, rows)


def _cell_contour_level(value: float, levels: tuple[float, ...]) -> str:
    eligible = [level for level in levels if level <= value]
    return "" if not eligible else _float(max(eligible))


def density_csv_bytes(grid: DensityGrid) -> bytes:
    """Serialize the complete density grid in x-major order."""
    if grid.density.shape != (len(grid.x_centers), len(grid.y_centers)):
        raise ValueError("density shape does not match grid centers")
    rows: list[tuple[Any, ...]] = []
    levels = tuple(float(level) for level in grid.contour_levels)
    for x_index, x_center in enumerate(grid.x_centers):
        for y_index, y_center in enumerate(grid.y_centers):
            value = float(grid.density[x_index, y_index])
            rows.append(
                (
                    _float(x_center),
                    _float(y_center),
                    _float(value),
                    _cell_contour_level(value, levels),
                    x_index,
                    y_index,
                )
            )
    return _csv_bytes(_DENSITY_COLUMNS, rows)


def _mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def build_audit_payload(
    *,
    dataset: str,
    source_manifest_sha256: str,
    pairs: ErrorPairs,
    density: DensityGrid,
    metrics: DensityMetrics,
    settings: DensityPlotSettings,
    code: Any,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the complete JSON-safe audit record for one plot group."""
    return {
        "schema_version": 1,
        "dataset": dataset,
        "task": pairs.task,
        "order": pairs.order,
        "unit": pairs.unit,
        "coordinates": {
            "x": "expected_error",
            "y": "actual_absolute_error",
            "scale": "log10",
        },
        "source_manifest_sha256": source_manifest_sha256,
        "filtering": dict(pairs.audit),
        "metrics": _mapping(metrics),
        "log10_range": {
            "expected": [float(np.min(pairs.expected_log10)), float(np.max(pairs.expected_log10))],
            "actual": [float(np.min(pairs.actual_log10)), float(np.max(pairs.actual_log10))],
        },
        "contour_levels": [float(value) for value in density.contour_levels],
        "plot_settings": _mapping(settings),
        "code_identity": _mapping(code),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_artifact_set(root: Path, files: Mapping[str, bytes]) -> list[Path]:
    """Atomically publish an all-new set or byte-identically reuse a full set."""
    directory = Path(root)
    if not files:
        raise ValueError("artifact set must not be empty")
    names = sorted(files)
    if any(Path(name).name != name for name in names):
        raise ValueError("artifact names must be plain file names")
    destinations = [directory / name for name in names]
    existing = [path.exists() for path in destinations]
    if any(existing):
        if not all(existing):
            raise PlotConflictError(f"partial artifact set exists in {directory}")
        mismatched = [
            path.name
            for path in destinations
            if path.read_bytes() != bytes(files[path.name])
        ]
        if mismatched:
            raise PlotConflictError(
                f"artifact content conflict in {directory}: {', '.join(mismatched)}"
            )
        return destinations

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        for destination in destinations:
            _atomic_write(destination, bytes(files[destination.name]))
            written.append(destination)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return destinations


def write_plot_manifest(
    root: Path,
    *,
    dataset: str,
    source_manifests: Mapping[str, str],
) -> Path:
    """Write or identically reuse a manifest over all other regular files."""
    directory = Path(root)
    manifest_path = directory / "plot_manifest.json"
    artifacts = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != manifest_path and not path.name.startswith(".")
    }
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "source_manifests": dict(sorted(source_manifests.items())),
        "artifacts": artifacts,
    }
    encoded = canonical_json_bytes(payload)
    if manifest_path.exists():
        if manifest_path.read_bytes() != encoded:
            raise PlotConflictError(f"manifest conflict: {manifest_path}")
        return manifest_path
    atomic_json_dump(manifest_path, payload)
    return manifest_path
