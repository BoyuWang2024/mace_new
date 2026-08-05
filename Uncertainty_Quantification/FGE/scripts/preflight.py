from __future__ import annotations

import argparse
from collections.abc import Sequence

from Uncertainty_Quantification.FGE.fge.config import load_config
from Uncertainty_Quantification.FGE.fge.preflight import run_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行一个 FGE 阶段的只读预检查")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("train", "predict", "evaluate"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    report = run_preflight(config, args.stage)
    print(config.output_dir / "preflight" / f"{report['stage']}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
