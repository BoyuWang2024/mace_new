#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fge.errors import HardFailure
from fge.validation import schema_signature


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HardFailure(f"invalid JSON mapping: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit converted formal FGE results")
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    baseline = None
    for root in args.results:
        if _json(root / "validation.json").get("status") != "PASS":
            raise HardFailure(f"validation is not PASS: {root.name}")
        if _json(root / "result_manifest.json").get("status") != "PASS":
            raise HardFailure(f"result manifest is not PASS: {root.name}")
        raw = sorted((root / "training" / "members" / "raw").glob("*.model"))
        ema = sorted((root / "training" / "members" / "ema").glob("*.model"))
        if len(raw) != 8 or len(ema) != 8:
            raise HardFailure(f"member count mismatch: {root.name}")
        forbidden = [
            path
            for path in root.rglob("*")
            if path.name.lower() in {"wandb", "logs", "checkpoints"}
            or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
        ]
        if forbidden:
            raise HardFailure(f"forbidden artifacts exist: {root.name}")
        signature = schema_signature(root)
        if baseline is None:
            baseline = signature
        elif signature != baseline:
            raise HardFailure(f"schema signature mismatch: {root.name}")
        print(f"PASS {root.name}: raw=8 ema=8")


if __name__ == "__main__":
    main()
