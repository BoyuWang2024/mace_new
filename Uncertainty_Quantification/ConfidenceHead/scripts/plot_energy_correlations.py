"""Thin command-line entry point for Energy order correlations."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.plot_energy_correlations import (
    run_plot_energy_correlations,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.production_matrix import (
    discover_production_matrix,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot Energy order 1-8 correlations without bootstrap CI"
    )
    parser.add_argument("--config-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    _, energy_configs = discover_production_matrix(arguments.config_dir)
    run_plot_energy_correlations(energy_configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
