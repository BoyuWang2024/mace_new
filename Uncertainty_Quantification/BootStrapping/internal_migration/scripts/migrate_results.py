"""Atomically adapt an authenticated old run without model computation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...scripts._cli import run_cli
from ..migration.converter import convert_legacy_run
from ..migration.legacy_reader import inspect_legacy_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("audit_root", type=Path)
    arguments = parser.parse_args(argv)

    def migrate() -> None:
        publication = convert_legacy_run(inspect_legacy_run(arguments.source), arguments.destination, arguments.audit_root)
        print({"destination": str(publication.destination), "written": publication.written, "members": publication.member_count})

    return run_cli(migrate)


if __name__ == "__main__":
    raise SystemExit(main())
