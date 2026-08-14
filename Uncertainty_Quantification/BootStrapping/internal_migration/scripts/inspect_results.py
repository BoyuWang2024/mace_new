"""Inspect and authenticate an old MACE BootStrapping run without writing."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...scripts._cli import run_cli
from ..migration.legacy_reader import inspect_legacy_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    arguments = parser.parse_args(argv)

    def inspect() -> None:
        audit = inspect_legacy_run(arguments.source)
        print({"members": len(audit.members), "seeds": [member.seed for member in audit.members], "models": len(audit.model_files), "resumes": len(audit.resume_files), "predictions": len(audit.member_predictions), "analyses": len(audit.analyses), "tracked_files": len(audit.source_snapshot)})

    return run_cli(inspect)


if __name__ == "__main__":
    raise SystemExit(main())
