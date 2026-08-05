from __future__ import annotations

import argparse
from collections.abc import Sequence

from Uncertainty_Quantification.FGE.fge.config import load_config
from Uncertainty_Quantification.FGE.fge.validation import validate_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="独立验证完整 FGE 结果树")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    validate_result(config, config.output_dir)
    print(config.output_dir / "result_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
