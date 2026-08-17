"""Command-line entry point for CPU external ConfidenceHead evaluation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Uncertainty_Quantification.ConfidenceHead.confidence_head.external_config import load_external_config
from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.evaluate_external import run_evaluate_external


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one head on external cache")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--head-key", required=True)
    arguments = parser.parse_args(argv)
    run_evaluate_external(load_external_config(arguments.config), arguments.head_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
