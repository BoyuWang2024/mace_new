"""Errors with stable meanings across every FGE stage."""


class FGEError(RuntimeError):
    """Base class for FGE workflow errors."""


class HardFailure(FGEError):
    """An integrity or numerical failure that invalidates the experiment."""


class ConfigError(HardFailure):
    """A strict configuration or publication-boundary violation."""
