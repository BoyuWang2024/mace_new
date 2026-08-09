"""Atomic orchestration of the complete FGE plotting suite."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path

from .artifacts import atomic_write_json
from .errors import HardFailure
from .plot_compare import render_comparisons
from .plot_data import load_plot_run
from .plot_single import FigureRecord, render_single_run
from .plot_style import PlotConfig


def _audit(records: tuple[FigureRecord, ...], staging: Path, labels: tuple[str, ...]) -> dict:
    return {
        "schema_version": "fge.plot-audit.v1",
        "status": "PASS",
        "experiments": list(labels),
        "logical_figure_count": len(records),
        "png_count": len(tuple(staging.rglob("*.png"))),
        "pdf_count": len(tuple(staging.rglob("*.pdf"))),
        "figures": [
            {
                "experiment": record.experiment,
                "branch": record.branch,
                "logical_name": record.logical_name,
                "png": record.png.relative_to(staging).as_posix(),
                "pdf": record.pdf.relative_to(staging).as_posix(),
                "excluded": record.excluded,
            }
            for record in records
        ],
    }


def render_all_figures(
    result_roots: Mapping[str, Path],
    output_root: Path,
    *,
    config: PlotConfig = PlotConfig(),
) -> Path:
    if len(result_roots) != 4:
        raise HardFailure("the publication suite requires exactly four experiments")
    if len(set(result_roots)) != len(result_roots):
        raise HardFailure("experiment labels must be unique")
    output_root = Path(output_root)
    if output_root.exists():
        raise HardFailure(f"refusing to overwrite existing figure directory: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.staging-{uuid.uuid4().hex}")
    if staging.exists():
        raise HardFailure(f"staging path already exists: {staging}")

    try:
        runs = {label: load_plot_run(Path(root)) for label, root in result_roots.items()}
        records: list[FigureRecord] = []
        for label, run in runs.items():
            if run.name != label:
                run = type(run)(name=label, root=run.root, branches=run.branches)
            records.extend(render_single_run(run, staging, config))
        records.extend(render_comparisons(runs, staging, config))
        frozen = tuple(records)
        png_count = len(tuple(staging.rglob("*.png")))
        pdf_count = len(tuple(staging.rglob("*.pdf")))
        if len(frozen) != 43 or png_count != 43 or pdf_count != 43:
            raise HardFailure(
                f"figure contract mismatch: logical={len(frozen)}, png={png_count}, pdf={pdf_count}"
            )
        if any(path.stat().st_size == 0 for path in staging.rglob("*.*")):
            raise HardFailure("one or more generated files are empty")
        atomic_write_json(staging / "plot_audit.json", _audit(frozen, staging, tuple(runs)))
        os.replace(staging, output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_root

