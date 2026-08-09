"""Explicit script entry point for plotting existing FGE results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.plot_workflow import render_all_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot validated canonical FGE results")
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="experiment label and canonical result root; provide exactly four times",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    labels = [pair[0] for pair in arguments.result]
    if len(set(labels)) != len(labels):
        raise HardFailure("duplicate --result label")
    roots = {label: Path(path) for label, path in arguments.result}
    render_all_figures(roots, arguments.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
