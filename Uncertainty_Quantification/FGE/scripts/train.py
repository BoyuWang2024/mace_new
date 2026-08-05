from __future__ import annotations

import argparse
from collections.abc import Sequence

from Uncertainty_Quantification.FGE.fge.config import load_config
from Uncertainty_Quantification.FGE.fge.training import train_fge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 FGE raw/EMA 成员")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(train_fge(load_config(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
