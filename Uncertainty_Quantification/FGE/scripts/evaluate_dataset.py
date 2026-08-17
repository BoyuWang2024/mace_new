from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from Uncertainty_Quantification.FGE.fge.config import load_config
from Uncertainty_Quantification.FGE.fge.dataset_evaluation import evaluate_dataset
from Uncertainty_Quantification.FGE.fge.derived_artifacts import DerivedLayout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one sharded FGE dataset prediction"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--outputs-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    layout = DerivedLayout(args.outputs_root, config.project_name, args.dataset_label)
    print(evaluate_dataset(config, layout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
