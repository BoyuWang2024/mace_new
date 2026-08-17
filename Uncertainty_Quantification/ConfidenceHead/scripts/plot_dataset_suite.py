"""Command-line entry point for stable per-dataset plot publication."""

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
from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.plot_dataset_suite import run_plot_dataset_suite, run_plot_test_dataset_suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot one ConfidenceHead dataset")
    parser.add_argument("--source", choices=("external", "test"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--plot-root", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.source == "external":
        if arguments.config is None:
            parser.error("--config is required for external source")
        run_plot_dataset_suite(load_external_config(arguments.config))
    else:
        if arguments.config_dir is None or arguments.plot_root is None:
            parser.error("--config-dir and --plot-root are required for test source")
        run_plot_test_dataset_suite(arguments.config_dir, arguments.plot_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
