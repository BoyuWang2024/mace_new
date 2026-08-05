from __future__ import annotations

import argparse
from collections.abc import Sequence

from Uncertainty_Quantification.FGE.fge.config import load_config
from Uncertainty_Quantification.FGE.fge.evaluation import evaluate_prediction
from Uncertainty_Quantification.FGE.fge.preflight import run_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只读评估 FGE canonical prediction")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    run_preflight(config, "evaluate")
    evaluate_prediction(config, config.output_dir)
    print(config.output_dir / "evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
