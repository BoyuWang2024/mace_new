#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fge.validation import schema_signature


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a normalized FGE result schema")
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(json.dumps(schema_signature(args.result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

