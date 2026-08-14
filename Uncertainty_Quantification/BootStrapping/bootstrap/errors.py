"""Expected domain failures for BootStrapping commands."""


class HardFailure(RuntimeError):
    """A user-actionable failure that should not hide unexpected tracebacks."""
