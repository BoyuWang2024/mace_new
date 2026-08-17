from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from Uncertainty_Quantification.FGE.fge.config import load_config
from Uncertainty_Quantification.FGE.fge.dataset_prediction import predict_dataset
from Uncertainty_Quantification.FGE.fge.dataset_spec import DatasetSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable FGE prediction for one extxyz dataset"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--shard-size", required=True, type=int)
    stress = parser.add_mutually_exclusive_group(required=True)
    stress.add_argument("--compute-stress", action="store_true")
    stress.add_argument("--no-stress", dest="compute_stress", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    data = config.section("data")
    spec = DatasetSpec(
        label=args.dataset_label,
        path=args.data,
        energy_key=data["energy_key"],
        forces_key=data["forces_key"],
        stress_key=data["stress_key"],
        head_name=data["head_name"],
        compute_stress=args.compute_stress,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
    )
    print(predict_dataset(config, spec, args.outputs_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
