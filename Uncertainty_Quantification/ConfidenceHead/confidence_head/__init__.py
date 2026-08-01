"""Strict configuration support for ConfidenceHead training."""

from .config import ConfidenceHeadConfig, load_config
from .errors import ConfigError

__all__ = ["ConfidenceHeadConfig", "ConfigError", "load_config"]
