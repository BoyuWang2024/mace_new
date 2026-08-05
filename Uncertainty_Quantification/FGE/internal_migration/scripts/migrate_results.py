#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from fge.config import load_config
from internal_migration.migration.converter import convert_legacy_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one verified FGE result tree")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    result = convert_legacy_run(
        config=load_config(args.config),
        legacy_root=args.legacy_root,
        destination=args.destination,
        audit_path=args.audit,
    )
    print(result)


if __name__ == "__main__":
    main()

