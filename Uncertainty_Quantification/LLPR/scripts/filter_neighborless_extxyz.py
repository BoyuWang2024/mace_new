"""Command-line entrypoint for filtering extxyz structures before MAD inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Uncertainty_Quantification.LLPR.llpr.dataset_filter import (  # noqa: E402
    filter_neighborless_extxyz,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--cutoff", type=float, required=True)
    arguments = parser.parse_args()
    report = filter_neighborless_extxyz(
        arguments.input, arguments.output, arguments.audit, cutoff=arguments.cutoff
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
