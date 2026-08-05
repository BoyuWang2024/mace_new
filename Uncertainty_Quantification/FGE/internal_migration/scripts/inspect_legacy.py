#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from internal_migration.migration.legacy_reader import read_legacy_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and summarize one legacy FGE result")
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    legacy = read_legacy_run(args.result)
    shape = legacy.prediction["energy_members"].shape
    atoms = legacy.prediction["forces_members"].shape[1]
    print(f"members={len(legacy.members)} structures={shape[1]} atoms={atoms}")
    print(f"base_metrics={legacy.base_metrics}")


if __name__ == "__main__":
    main()

