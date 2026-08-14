"""Compare every migrated model, resume and array against its old source."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...scripts._cli import run_cli
from ..migration.legacy_reader import inspect_legacy_run
from ..migration.validation import validate_migrated_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args(argv)
    return run_cli(lambda: print(validate_migrated_run(inspect_legacy_run(arguments.source), arguments.destination)))


if __name__ == "__main__":
    raise SystemExit(main())
