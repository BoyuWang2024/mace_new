from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.plot_data import load_dataset_plot_run
from Uncertainty_Quantification.FGE.fge.plot_workflow import render_dataset_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one dataset-scoped FGE figure suite"
    )
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument(
        "--result",
        nargs=2,
        action="append",
        metavar=("LABEL", "DERIVED_DATASET_ROOT"),
        required=True,
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.result) != 4:
        raise HardFailure("plot_dataset requires exactly four --result entries")
    roots = {label: Path(root) for label, root in args.result}
    if len(roots) != 4:
        raise HardFailure("plot_dataset experiment labels must be unique")
    runs = {label: load_dataset_plot_run(root) for label, root in roots.items()}
    print(
        render_dataset_figures(
            runs,
            args.output_root,
            dataset=args.dataset_label,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
