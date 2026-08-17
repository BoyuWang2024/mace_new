"""Command-line entry point for auditable MAD element filtering."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.prepare_external_dataset import (
    prepare_compatible_dataset,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter unsupported MAD structures")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument(
        "--unsupported-atomic-number", type=int, action="append", required=True
    )
    arguments = parser.parse_args(argv)
    prepare_compatible_dataset(
        source_path=arguments.source,
        output_path=arguments.output,
        unsupported_atomic_numbers=arguments.unsupported_atomic_number,
        expected_source_sha256=arguments.source_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
