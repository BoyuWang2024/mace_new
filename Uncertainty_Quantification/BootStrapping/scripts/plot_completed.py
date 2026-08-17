"""Plot STD against absolute residual for completed MACE bootstrap results."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..bootstrap.completed_plotting import render_dataset_plots
from ..bootstrap.inference_config import load_inference_config
from ._cli import run_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dataset", required=True, choices=("matpes_test", "mad_test", "matpes_train")
    )
    parser.add_argument("--inference-root", type=Path)
    arguments = parser.parse_args(argv)
    config = load_inference_config(arguments.config)
    inference_root = (
        arguments.inference_root
        or config.source_path.parent.parent / "outputs" / "completed_inference"
    )
    return run_cli(
        lambda: print(
            render_dataset_plots(
                config, arguments.dataset, inference_root=inference_root
            )
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
