"""ConfidenceHead-specific exceptions."""


class ConfigError(ValueError):
    """Raised when a ConfidenceHead configuration violates its contract."""


class DataContractError(ValueError):
    """Raised when a checkpoint or structure dataset violates its contract."""


class FeatureSchemaError(ValueError):
    """Raised when captured MACE features do not match the fixed schema."""
