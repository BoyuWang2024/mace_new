"""Run resumable prediction and UQ for one completed-ensemble dataset."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..bootstrap.completed_pipeline import run_prediction_dataset
from ..bootstrap.inference_config import load_inference_config
from ._cli import run_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("mad_test", "matpes_train"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    arguments = parser.parse_args(argv)
    config = load_inference_config(arguments.config)
    if arguments.device is not None:
        config = replace(
            config,
            prediction=replace(config.prediction, device=arguments.device),
        )
    output_root = (
        arguments.output_root
        or config.source_path.parent.parent / "outputs" / "completed_inference"
    )
    return run_cli(
        lambda: print(
            run_prediction_dataset(
                config, config.datasets[arguments.dataset], output_root=output_root
            )
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
