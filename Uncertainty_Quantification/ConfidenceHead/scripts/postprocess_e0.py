"""Command-line entry point for MAD-r2SCAN E0 postprocessing."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Uncertainty_Quantification.ConfidenceHead.confidence_head.external_config import (
    load_e0_postprocess_config,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.postprocess_e0 import (
    run_e0_postprocess,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Postprocess existing ConfidenceHead inference with two E0 methods"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="publish continuous density plots after both methods complete",
    )
    arguments = parser.parse_args(argv)
    run_e0_postprocess(
        load_e0_postprocess_config(arguments.config),
        plot=arguments.plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
