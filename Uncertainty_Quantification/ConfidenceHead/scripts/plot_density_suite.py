"""CLI for publishing ConfidenceHead continuous density plots."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Uncertainty_Quantification.ConfidenceHead.confidence_head.external_config import load_external_config
from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows.plot_density_suite import run_external_density_suite, run_test_density_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot existing ConfidenceHead evaluations")
    parser.add_argument("--source", choices=("external", "test"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--plot-root", type=Path)
    parser.add_argument("--dataset-name", default="matpes_test")
    args = parser.parse_args()
    if args.source == "external":
        if args.config is None:
            parser.error("--config is required for external source")
        run_external_density_suite(load_external_config(args.config), REPOSITORY_ROOT)
    else:
        if args.config_dir is None or args.plot_root is None:
            parser.error("--config-dir and --plot-root are required for test source")
        run_test_density_suite(args.config_dir, args.plot_root, REPOSITORY_ROOT, args.dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
