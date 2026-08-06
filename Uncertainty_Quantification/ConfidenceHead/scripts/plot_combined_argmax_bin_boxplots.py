"""Thin command-line entry point for combined argmax-bin PDFs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.plot_combined_argmax_bin_boxplots import (
    run_plot_combined_argmax_bin_boxplots,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.production_matrix import (
    discover_production_matrix,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build combined Force and Energy argmax-bin PDFs"
    )
    parser.add_argument("--config-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    force_config, energy_configs = discover_production_matrix(arguments.config_dir)
    run_plot_combined_argmax_bin_boxplots(force_config, energy_configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
