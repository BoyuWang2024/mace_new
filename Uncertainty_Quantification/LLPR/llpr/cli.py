"""Command-line orchestration for the deterministic LLPR workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .calibration import run_calibrate
from .config import load_config
from .curvature import run_build
from .inference import run_evaluate
from .plotting import run_plot
from .validation import run_validate


_COMPUTING_STAGES = ("build", "calibrate", "evaluate", "validate")


@dataclass(frozen=True)
class PlotConfig:
    publication_root: Path
    output_dir: Path
    selected: tuple[tuple[str, str], ...]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _plot_path(source_dir: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return (source_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _load_plot_config(path: Path) -> PlotConfig:
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8") as handle:
        document = _mapping(yaml.safe_load(handle), "plot config")
    expected_fields = {"publication_root", "output_dir", "selected"}
    if set(document) != expected_fields:
        raise ValueError(
            "plot config must contain exactly publication_root, output_dir, selected"
        )
    raw_selected = document["selected"]
    if not isinstance(raw_selected, list):
        raise ValueError("plot config selected must be a list")
    selected: list[tuple[str, str]] = []
    for index, item in enumerate(raw_selected):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(value, str) for value in item)
        ):
            raise ValueError(
                f"plot config selected[{index}] must be [variant, target]"
            )
        selected.append((item[0], item[1]))
    source_dir = source_path.parent
    return PlotConfig(
        publication_root=_plot_path(
            source_dir, document["publication_root"], "publication_root"
        ),
        output_dir=_plot_path(source_dir, document["output_dir"], "output_dir"),
        selected=tuple(selected),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m Uncertainty_Quantification.LLPR.llpr",
        description="MACE LLPR deterministic workflow",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (*_COMPUTING_STAGES, "plot", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, type=Path)
    return parser


def _run_computing_stage(command: str, config_path: Path) -> None:
    config = load_config(config_path)
    if command == "build":
        run_build(config)
    elif command == "calibrate":
        run_calibrate(config)
    elif command == "evaluate":
        run_evaluate(config)
    elif command == "validate":
        run_validate(config)
    else:
        raise ValueError(f"unknown computing stage: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch one requested workflow operation."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "plot":
        config = _load_plot_config(arguments.config)
        run_plot(
            config.publication_root,
            output_dir=config.output_dir,
            selected=config.selected,
        )
    elif arguments.command == "run":
        config = load_config(arguments.config)
        run_build(config)
        run_calibrate(config)
        run_evaluate(config)
        run_validate(config)
    else:
        _run_computing_stage(arguments.command, arguments.config)
    return 0
