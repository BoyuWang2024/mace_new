"""Thin command-line entry point for ConfidenceHead training."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import (
    load_config,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.train import (
    run_train,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ConfidenceHead from cache")
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_train(load_config(arguments.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
