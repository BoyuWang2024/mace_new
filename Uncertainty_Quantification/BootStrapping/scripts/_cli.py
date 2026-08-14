"""Shared command-line failure handling."""

from __future__ import annotations

import sys
from collections.abc import Callable

from ..bootstrap.errors import HardFailure


def run_cli(main: Callable[[], object]) -> int:
    """Run a command, rendering expected failures without masking programming errors."""

    try:
        main()
    except HardFailure as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0
