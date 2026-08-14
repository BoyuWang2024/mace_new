"""Build the allowlist-only public MACE BootStrapping source archive."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..bootstrap.release import build_release
from ._cli import run_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args(argv)
    package = Path(__file__).parents[1]
    return run_cli(lambda: print(build_release(package, arguments.destination)))


if __name__ == "__main__":
    raise SystemExit(main())
