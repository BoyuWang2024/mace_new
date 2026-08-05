from __future__ import annotations

import argparse
from collections.abc import Sequence

from Uncertainty_Quantification.FGE.fge.config import load_config
from Uncertainty_Quantification.FGE.fge.prediction import predict_members


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 FGE canonical raw prediction")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(predict_members(load_config(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
