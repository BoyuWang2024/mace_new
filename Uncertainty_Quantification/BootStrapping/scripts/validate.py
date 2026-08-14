"""Validate a canonical MACE BootStrapping run without loading models."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..bootstrap.validation import validate_run
from ._cli import run_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    arguments = parser.parse_args(argv)
    return run_cli(lambda: print(validate_run(arguments.run)))


if __name__ == "__main__":
    raise SystemExit(main())
